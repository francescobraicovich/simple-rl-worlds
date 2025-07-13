# Model Architectures

This document provides a rigorous and intuitive explanation of the core neural network components used in this project. Understanding these models is key to understanding the entire system.

---

## 1. Encoder (`VideoViT`)

**File:** `src/models/encoder.py`

### Goal & Intuition

The Encoder's job is to look at a short sequence of raw image frames from the game and compress it into a sequence of compact, meaningful latent vectors. Think of it as a summarizer. Instead of dealing with thousands of pixel values per frame, the rest of the system can work with a much smaller vector that captures the *essence* of what's happening in that frame (e.g., "the player is on the left, and an enemy is approaching from the right").

This project uses a Vision Transformer (ViT) adapted for video, which allows it to effectively identify and relate different objects and regions within each frame.

### Detailed Architecture

*   **Input:** A video clip tensor of shape `[Batch, Channels, Time, Height, Width]`. For example: `[64, 1, 6, 64, 64]` (a batch of 64 clips, each with 6 grayscale frames of 64x64 resolution).
*   **Output:** A sequence of latent vectors, one for each frame, with shape `[Batch, Time, Embedding Dimension]`. For example: `[64, 6, 128]`.

#### Key Component: `FactorizedPatchEmbed`

*   **Mechanism:** The first step is to break each frame into smaller patches. This is done using a 3D convolution where the kernel size along the time dimension is 1 (`kernel_size=(1, patch_h, patch_w)`). This is a clever way to apply a standard 2D convolution (for patching) to every frame in the sequence independently and efficiently. It's "factorized" because the spatial and temporal dimensions are handled separately at this stage.
*   **Data Flow:** It transforms the input tensor from `[B, C, T, H, W]` into a sequence of patch embeddings for each frame, resulting in a shape of `[B, T, Num_Patches, Embed_Dim]`.

#### Key Component: `TransformerBlock`

*   **Mechanism:** The core of the Encoder is a series of standard Transformer blocks. Each block consists of Multi-Head Self-Attention (MHSA) and an MLP. The key is that these blocks are applied to the patches of *each frame independently*. The model reshapes the tensor to `[Batch * Time, Num_Patches, Embed_Dim]` to process all frames as one large batch, which is highly efficient on GPUs.
*   **Rotary Position Embeddings (RoPE):** Instead of adding a separate positional embedding to the patches, this model uses RoPE. RoPE modifies the query and key vectors within the attention mechanism to encode *relative* positional information. This helps the model understand the spatial layout of the patches (e.g., this patch is to the left of that patch) without altering the patch embeddings themselves.

#### Forward Pass Walkthrough

1.  **Patching:** The input video clip `[B, C, T, H, W]` is passed to `FactorizedPatchEmbed`, which creates a sequence of patch embeddings for each frame, resulting in `[B, T, Num_Patches, Embed_Dim]`.
2.  **Batch Reshape:** The tensor is reshaped to `[B * T, Num_Patches, Embed_Dim]`.
3.  **Transformer Processing:** This large batch is processed by the stack of `TransformerBlock`s. Self-attention is computed only among the patches *within the same frame*.
4.  **Un-Reshape:** The tensor is reshaped back to `[B, T, Num_Patches, Embed_Dim]`.
5.  **Spatial Pooling (Crucial Step):** For each frame, the model computes the mean of all its patch embeddings (`x.mean(dim=2)`). This collapses the `Num_Patches` dimension, producing a single, fixed-size latent vector that summarizes the entire frame. The final output is `[B, T, Embed_Dim]`.

---

## 2. Predictor (`LatentDynamicsPredictor`)

**File:** `src/models/predictor.py`

### Goal & Intuition

The Predictor is the heart of the world model. Its purpose is to understand the **dynamics** of the environment in the compressed latent space. Given the history of what the world looked like (the sequence of latent vectors `z_0, ..., z_t` from the Encoder) and the action the agent took (`a_t`), the Predictor's job is to predict what the world will look like in the next moment (`z_{t+1}`).

### Detailed Architecture

*   **Input:** A sequence of latent vectors `[B, T, E]` from the Encoder and a batch of discrete actions `[B]`.
*   **Output:** A single predicted latent vector for the *next* frame, with shape `[B, 1, E]`.

#### Key Component: Action Embedding

*   The discrete action (e.g., integer `3` for "fire") is not suitable for a Transformer. An `nn.Embedding` layer converts this integer into a dense vector of the same dimension as the state embeddings, allowing it to be processed seamlessly.

#### Key Component: Causal Transformer

*   **Mechanism:** The Predictor is also a Transformer, but it uses **causal masking**. This is a critical feature. The causal mask ensures that when the model is making a prediction for a specific time step, it can only attend to (i.e., look at) the tokens that came before it in the sequence. This prevents the model from cheating by looking into the future.

#### Forward Pass Walkthrough

1.  **Action Embedding:** The action `a_t` is embedded into a vector.
2.  **Sequence Prepending:** This action vector is prepended to the sequence of state vectors from the Encoder. The input to the Transformer is now `[action_token, z_0, z_1, ..., z_t]`.
3.  **Causal Transformer Processing:** The full sequence is processed by the stack of causal `TransformerBlock`s.
4.  **Prediction (Crucial Step):** The model takes only the **very last output token** from the Transformer. Because of the causal masking and the sequence structure, this last token is the model's prediction for the element that would come next in the sequence—the latent state of the next frame, `z_{t+1}`.
5.  **Prediction Head:** This final token is passed through a `LayerNorm` and a `Linear` layer to produce the final predicted latent vector.

---

## 3. Decoder (`HybridConvTransformerDecoder`)

**File:** `src/models/decoder.py`

### Goal & Intuition

The Decoder's role is to translate a compressed latent vector back into a full image. This is essential for the Encoder-Decoder architecture, where the training signal is the difference between the reconstructed image and the real image. It also allows us to visualize what the model *thinks* the world looks like, which is useful for debugging and evaluation.

It is a "hybrid" model because it uses both convolutions (great for spatial tasks like image generation) and attention (great for incorporating global context).

### Detailed Architecture

*   **Input:** A single latent vector `[B, 1, E]`, typically from the Predictor.
*   **Output:** A reconstructed, single-channel image tensor `[B, 1, 1, H, W]`.

#### Key Component: `token_decoder`

*   This is a simple `Linear` layer that acts as a starting point. It takes the single input latent vector and projects it into a much larger vector that is then reshaped into a grid of `H_p x W_p` "spatial tokens." This creates a low-resolution blueprint of the image to be generated.

#### Key Component: `UpsampleBlock` & `MultiHeadCrossAttention`

*   **Mechanism:** The Decoder is a series of these blocks. Each block doubles the spatial resolution of the image being generated using a 3D transposed convolution. The magic happens in the **cross-attention** step.
*   **Cross-Attention:** In each block, the upsampled convolutional feature map (the **Query**) attends to the grid of `spatial_tokens` created at the very beginning (the **Key** and **Value**). This is incredibly powerful. It means that as the Decoder builds the image layer by layer, it can constantly look back at the original blueprint to ensure the details it's adding are consistent with the initial instruction (the latent vector).

#### Forward Pass Walkthrough

1.  **Token Expansion:** The input latent vector `[B, 1, E]` is passed through the `token_decoder` and reshaped into a grid of spatial tokens `[B, H_p * W_p, E]`.
2.  **Initialization:** These spatial tokens are used to initialize a very low-resolution 3D feature map `[B, C, 1, H_p, W_p]`.
3.  **Iterative Upsampling:** The feature map is passed through a series of `UpsampleBlock`s. In each block:
    a.  The spatial resolution is doubled via transposed convolution.
    b.  The upsampled features are refined using cross-attention with the initial spatial tokens.
4.  **Final Projection:** A final 1x1x1 convolution collapses the channel dimension to 1, producing the final grayscale image `[B, 1, 1, H, W]`.

---

## 4. Reward Predictor (`RewardPredictor`)

**File:** `src/models/reward_predictor.py`

### Goal & Intuition

This model has a simple but important job: predict the reward for a given transition. It looks at the latent representation of the state before the action (`z_t`) and the state after the action (`z_{t+1}`) and estimates the scalar reward value the agent received.

### Detailed Architecture

*   **Input:** Two latent vectors (or sequences of vectors), `z_t` and `z_{t+1}`, both of shape `[B, T, E]`.
*   **Output:** A single scalar reward value, shape `[B, 1, 1]`.

#### Key Component: `CLS` Token and Cross-Attention

*   **Mechanism:** This model uses a common pattern from Transformer literature for classification/regression tasks. A special, learnable `[CLS]` token is created. This token acts as an aggregator.
*   **Intuition:** The `[CLS]` token is like a detective. It is given the latent descriptions of the state before and after the transition, and its job is to query this information using attention to find the clues needed to figure out the reward. The final state of the `[CLS]` token after attention is a summary of the entire transition, which is then used for the final prediction.

#### Forward Pass Walkthrough

1.  **Projection & Concatenation:** The input latent vectors `z_t` and `z_{t+1}` are first projected with a `Linear` layer and then concatenated together into a single, long sequence.
2.  **Prepend `CLS` Token:** The learnable `[CLS]` token is prepended to this combined sequence.
3.  **Cross-Attention:** A `MultiheadAttention` layer is used where the `query` is the `[CLS]` token, and the `key` and `value` are the entire sequence (including the `CLS` token itself). This forces the `CLS` token to gather information from the entire transition.
4.  **MLP Head:** The output vector corresponding to the `[CLS]` token is isolated and passed through a small MLP, which regresses it to a single scalar value—the predicted reward.
