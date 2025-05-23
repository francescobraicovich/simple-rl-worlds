import torch
import torch.nn as nn
from einops.layers.torch import Rearrange # Moved to top as per instructions
from .vit import ViT # Assuming vit.py is in the same directory

class StandardEncoderDecoder(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 input_channels, # Number of channels in the input image (e.g., 3 for RGB, 1 for grayscale)
                 action_dim, # Dimension of the action space (e.g., number of discrete actions or size of continuous vector)
                 action_emb_dim, # Dimension to embed actions into
                 latent_dim, # Dimension of ViT output (encoder_output_dim)
                 decoder_dim, # Dimension for the Transformer Decoder
                 decoder_depth,
                 decoder_heads,
                 decoder_mlp_dim,
                 output_channels, # Number of channels in the output image (should match input_channels)
                 output_image_size, # Output image height/width (should match input_image_size for s_t+1 prediction)
                 vit_depth,
                 vit_heads,
                 vit_mlp_dim,
                 vit_pool='cls',
                 vit_dropout=0.,
                 vit_emb_dropout=0.,
                 decoder_dropout=0.):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.output_image_size = output_image_size

        # Encoder (ViT)
        self.encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            channels=input_channels,
            num_classes=0, # We want the latent representation directly
            dim=latent_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            pool=vit_pool,
            dropout=vit_dropout,
            emb_dropout=vit_emb_dropout
        )

        # Action embedding
        # If action_dim is for discrete actions, nn.Embedding is suitable.
        # If continuous, a small MLP could be used. Assuming discrete for now.
        # For simplicity, let's use a linear layer for action embedding,
        # which works for both discrete (if one-hot encoded) or continuous.
        self.action_embedding = nn.Linear(action_dim, action_emb_dim)

        # The decoder needs to predict an image.
        # The ViT encoder outputs a single latent vector (e.g., CLS token).
        # The action is also a vector.
        # We need to combine these and feed them to a Transformer Decoder.
        # The Transformer Decoder typically expects a sequence.

        # Option 1: Concatenate latent_state and action_embedding, then project.
        # self.decoder_input_projection = nn.Linear(latent_dim + action_emb_dim, decoder_dim)
        
        # Option 2: Use latent_state as memory and action_embedding as the initial query/input to the decoder.
        # For predicting a sequence (pixels of an image), a common approach is to have learnable query tokens
        # that attend to the encoded state and action.

        # Let's try a simpler approach first:
        # Project the concatenated (latent_state + embedded_action) to decoder_dim
        # This combined representation will be the initial hidden state or input to the decoder.
        # The decoder then needs to output a sequence of patch embeddings for the target image.

        self.decoder_input_dim = latent_dim + action_emb_dim
        self.decoder_input_projection = nn.Linear(self.decoder_input_dim, decoder_dim)

        # Transformer Decoder
        # PyTorch's nn.TransformerDecoderLayer and nn.TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_mlp_dim,
            dropout=decoder_dropout,
            batch_first=True # Important!
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # To generate an image, the decoder needs to output a sequence of patch embeddings.
        # The number of patches is (image_size // patch_size) ** 2.
        num_output_patches = (self.output_image_size // patch_size) ** 2
        self.num_output_patches = num_output_patches
        
        # These will be the learnable query tokens for the decoder
        self.decoder_query_tokens = nn.Parameter(torch.randn(1, num_output_patches, decoder_dim))

        # Final layer to project decoder output to patch pixel values
        output_patch_dim = output_channels * patch_size * patch_size
        self.to_pixels = nn.Linear(decoder_dim, output_patch_dim)
        
        # Layer to reconstruct image from patches
        self.patch_to_image = nn.Sequential(
            Rearrange('(b h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      p1=patch_size, p2=patch_size, 
                      h=(self.output_image_size // patch_size), 
                      w=(self.output_image_size // patch_size),
                      c=output_channels)
        )


    def forward(self, current_state_img, action):
        # current_state_img: (b, c, h, w)
        # action: (b, action_dim) - could be one-hot for discrete or continuous vector
        from einops import rearrange # Added for the forward pass as it uses the rearrange function

        # 1. Encode current state
        latent_s_t = self.encoder(current_state_img)  # (b, latent_dim)

        # 2. Embed action
        embedded_action = self.action_embedding(action) # (b, action_emb_dim)

        # 3. Combine latent state and action to form decoder input context (memory)
        # The TransformerDecoder expects 'memory' to attend to.
        # Latent_s_t is (b, latent_dim), embedded_action is (b, action_emb_dim)
        # We can concatenate them and project, then unsqueeze to make it (b, 1, projected_dim)
        # This will serve as the 'memory' for the decoder.
        decoder_memory_input = torch.cat((latent_s_t, embedded_action), dim=-1) # (b, latent_dim + action_emb_dim)
        decoder_memory = self.decoder_input_projection(decoder_memory_input) # (b, decoder_dim)
        decoder_memory = decoder_memory.unsqueeze(1) # (b, 1, decoder_dim) - memory for the decoder

        # 4. Prepare query tokens for the decoder
        # These are learnable parameters that ask "what should each patch of the next image be?"
        batch_size = current_state_img.shape[0]
        query_tokens = self.decoder_query_tokens.repeat(batch_size, 1, 1) # (b, num_output_patches, decoder_dim)

        # 5. Pass through Transformer Decoder
        # query_tokens is the target sequence (what we want to generate)
        # decoder_memory is the memory (context from s_t and a_t)
        decoded_patches_representation = self.transformer_decoder(tgt=query_tokens, memory=decoder_memory)
        # Output: (b, num_output_patches, decoder_dim)

        # 6. Project to pixel values for each patch
        predicted_patches = self.to_pixels(decoded_patches_representation)
        # Output: (b, num_output_patches, output_patch_dim)

        # 7. Rearrange patches back into an image
        # Before rearranging, we need to ensure the shape is compatible.
        # Rearrange expects (b*h*w, p1*p2*c) or similar if batch is preserved.
        # Let's adjust to: b, num_patches, patch_dim -> (b * num_patches), patch_dim
        # then rearrange.
        # The current rearrange is set up for: (b h w) (p1 p2 c)
        # This means we need to reshape predicted_patches from (b, num_output_patches, output_patch_dim)
        # to (b * num_output_patches, output_patch_dim) and provide h_num_patches and w_num_patches.
        
        # Let h_num_patches = w_num_patches = self.output_image_size // self.patch_size
        # num_output_patches = h_num_patches * w_num_patches
        # So, reshape predicted_patches to (b, h_num_patches, w_num_patches, output_patch_dim)
        # then view as (b * h_num_patches * w_num_patches, output_patch_dim) for the Rearrange.
        
        # The Rearrange pattern `(b h w) (p1 p2 c)` means the input should be
        # `(batch_size * num_patches_h * num_patches_w, patch_height * patch_width * channels)`
        # Our `predicted_patches` is `(batch_size, num_total_patches, patch_dim)`.
        # We need to make `batch_size * num_total_patches` the first dimension.
        
        # The Rearrange `Rearrange('(b h w) (p1 p2 c) -> b c (h p1) (w p2)', ...)`
        # expects input of shape (B, D) where B = b*h*w and D = p1*p2*c.
        # Our predicted_patches is (batch_size, self.num_output_patches, output_patch_dim).
        # We need to make it (batch_size * self.num_output_patches, output_patch_dim).
        
        # Let's modify the Rearrange layer slightly for clarity or ensure the view is correct.
        # The current `self.patch_to_image` contains a Rearrange layer.
        # `Rearrange('(b ph pw) (p1 p2 c) -> b c (ph p1) (pw p2)', p1=ph, p2=pw, ph=num_patch_h, pw=num_patch_w)`
        # Input to this would be (batch_size * num_patches_h * num_patches_w, patch_dim)

        # Reconstruct image from patches
        # predicted_patches shape: (b, num_output_patches, output_patch_dim)
        # We need to pass it to a layer that assumes input like:
        # (batch, sequence_length, features) where sequence_length is num_patches
        # and features is patch_dim.
        
        # The `patch_to_image` layer expects:
        # Input: (B_prime, P_dim) where B_prime = b * (output_image_size/patch_size) * (output_image_size/patch_size)
        # and P_dim = patch_size * patch_size * channels
        # Our `predicted_patches` is (b, num_output_patches, output_patch_dim)
        # Reshape it:
        predicted_patches_reshaped = rearrange(predicted_patches, 'b n d -> (b n) d')

        # The Rearrange in patch_to_image is:
        # `Rearrange('(b_h_w) (p1 p2 c) -> b c (h p1) (w p2)'...
        # where it infers 'b' for the output from the first dimension of the input,
        # and 'h', 'w' are num_patches_high and num_patches_wide.
        # It should be: `Rearrange('(b np_h np_w) (p1 p2 c) -> b c (np_h p1) (np_w p2)'`
        # For this to work, the input to Rearrange should be `(batch_size * np_h * np_w, patch_dim)`.
        # This is what `predicted_patches_reshaped` is.
        
        # We need to make sure the `Rearrange` layer definition matches this.
        # The `h` and `w` in `(b h w)` of the Rearrange pattern are numbers of patches.
        num_patches_h = self.output_image_size // self.patch_size
        num_patches_w = self.output_image_size // self.patch_size

        # Update self.patch_to_image to use specific h, w for patch counts if not done already
        # The current definition is:
        # Rearrange('(b h w) (p1 p2 c) -> b c (h p1) (w p2)', 
        #           p1=patch_size, p2=patch_size, 
        #           h=(self.output_image_size // patch_size), 
        #           w=(self.output_image_size // patch_size),
        #           c=output_channels)
        # This seems correct. It expects the input to be compatible with (b*h*w) as the first dimension.
        
        predicted_s_t_plus_1 = self.patch_to_image(predicted_patches_reshaped)
        # Output: (b, output_channels, output_image_size, output_image_size)

        return predicted_s_t_plus_1

# Helper for Rearrange if not already globally available in this file.
# It's better to import it if it's used.
# from einops.layers.torch import Rearrange  # This was moved to the top.
# This was already in ViT, but good to ensure it's conceptually here too.
# The class definition of StandardEncoderDecoder uses Rearrange in self.patch_to_image.
# So it needs to be available at class definition time.
# Let's put the import at the top of the file. # Done.
