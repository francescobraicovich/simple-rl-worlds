import torch
import torch.nn.functional as F
import wandb
import time
from tqdm import tqdm
from src.utils.data_utils import ( # Assuming these are used for action processing like in reward_predictor_loop
    process_actions_continuous,
    process_actions_discrete
)

def train_larp_epoch(
    larp_model,
    base_model,
    optimizer_larp,
    train_dataloader,
    val_dataloader,
    loss_fn,
    device,
    action_dim,
    action_type,
    model_name_log_prefix, # e.g., "LARP (Enc-Dec)" or "LARP (JEPA)"
    num_epochs_larp, # Renamed from num_epochs to be specific
    log_interval_larp, # Renamed from log_interval
    early_stopping_patience,
    is_jepa_base_model, # True if base_model is JEPA, False for EncoderDecoder variants
    wandb_run,
    current_epoch_main_training # For logging epoch alignment if needed
):
    """
    Training loop for Look-Ahead Reward Predictor (LARP).
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    total_train_steps = 0

    for epoch in range(num_epochs_larp):
        larp_model.train()
        if base_model: # Base model (world model) should be in eval mode for feature extraction
            base_model.eval()

        train_loss_sum = 0.0
        num_train_batches = 0

        # Training phase
        progress_bar_train = tqdm(train_dataloader, desc=f"{model_name_log_prefix} Epoch {epoch+1}/{num_epochs_larp} [Train]", leave=False)
        for batch_idx, (s_t, a_t, r_t, s_t_plus_1, _, _) in enumerate(progress_bar_train): # Assuming data format
            s_t, a_t, r_t, s_t_plus_1 = s_t.to(device), a_t.to(device), r_t.to(device), s_t_plus_1.to(device)

            # Process actions
            if action_type == 'continuous':
                a_t_processed = process_actions_continuous(a_t, action_dim).float()
            elif action_type == 'discrete':
                a_t_processed = process_actions_discrete(a_t, action_dim).float()
            else:
                raise ValueError(f"Unsupported action_type: {action_type}")

            # Detach inputs to LARP to prevent gradients flowing back to the base_model
            with torch.no_grad():
                if is_jepa_base_model:
                    # JEPA base model
                    # Inputs: encoded state s_t, latent prediction of the next state, and the action embedding.
                    # base_model(s_t, a_t_processed, s_t_plus_1) returns:
                    # predicted_embedding (latent s_{t+1}), target_embedding (encoded s_{t+1}),
                    # online_s_t_representation (encoded s_t), target_s_t_representation (ema encoded s_t)
                    predicted_latent_s_t_plus_1, _, online_s_t_representation, _ = base_model(s_t, a_t_processed, s_t_plus_1)
                    action_embedding = base_model.action_embedding(a_t_processed)

                    input_to_larp = torch.cat((
                        online_s_t_representation.detach(),
                        predicted_latent_s_t_plus_1.detach(),
                        action_embedding.detach()
                    ), dim=1)

                else:  # Standard Encoder-Decoder or JEPAStyle Encoder-Decoder base model
                    encoded_s_t = base_model.encoder(s_t)
                    action_embedding = base_model.action_embedding(a_t_processed)

                    # Get predicted next state (image) from the base_model's forward pass
                    # For StandardEncoderDecoder and EncoderDecoderJEPAStyle, forward(s_t, a_t) gives predicted_s_t_plus_1
                    predicted_s_t_plus_1_img = base_model(s_t, a_t_processed)
                    predicted_s_t_plus_1_flat = predicted_s_t_plus_1_img.flatten(start_dim=1)

                    if hasattr(base_model, 'predictor') and hasattr(base_model, 'decoder') and callable(getattr(base_model, 'predictor')):
                        # This identifies EncoderDecoderJEPAStyle
                        # It has a .predictor module (distinct from JEPA's main predictor)
                        # LARP inputs: encoded s_t, predicted next state s_{t+1}, action embedding,
                        #              intermediate features after the predictor module (before the decoder)

                        # The predictor in EncoderDecoderJEPAStyle takes concatenated encoded_s_t and action_embedding
                        predictor_input_jepa_style = torch.cat((encoded_s_t, action_embedding), dim=-1)
                        intermediate_predictor_features = base_model.predictor(predictor_input_jepa_style)

                        input_to_larp = torch.cat((
                            encoded_s_t.detach(),
                            predicted_s_t_plus_1_flat.detach(),
                            action_embedding.detach(),
                            intermediate_predictor_features.detach()
                        ), dim=1)
                    else:  # Standard Encoder-Decoder
                        # Inputs: encoded state s_t, predicted next state s_{t+1}, and the embedded action.
                        input_to_larp = torch.cat((
                            encoded_s_t.detach(),
                            predicted_s_t_plus_1_flat.detach(),
                            action_embedding.detach()
                        ), dim=1)

            # Forward pass and training step for LARP
            optimizer_larp.zero_grad()
            pred_reward = larp_model(input_to_larp) # input_to_larp is already detached
            loss = loss_fn(pred_reward, r_t.unsqueeze(1).float()) # Ensure r_t matches pred_reward shape
            loss.backward()
            optimizer_larp.step()

            train_loss_sum += loss.item()
            num_train_batches += 1
            total_train_steps +=1

            if wandb_run and batch_idx % log_interval_larp == 0:
                wandb.log({
                    f"larp/{model_name_log_prefix}/train/loss_step": loss.item(),
                    f"larp/{model_name_log_prefix}/train/step": total_train_steps,
                    f"larp/{model_name_log_prefix}/epoch_progress_main": current_epoch_main_training # Log main training epoch
                })
            progress_bar_train.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_sum / num_train_batches if num_train_batches > 0 else 0

        # Validation phase
        larp_model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0
        progress_bar_val = tqdm(val_dataloader, desc=f"{model_name_log_prefix} Epoch {epoch+1}/{num_epochs_larp} [Val]", leave=False)
        with torch.no_grad():
            for s_t, a_t, r_t, s_t_plus_1, _, _ in progress_bar_val:
                s_t, a_t, r_t, s_t_plus_1 = s_t.to(device), a_t.to(device), r_t.to(device), s_t_plus_1.to(device)

                if action_type == 'continuous':
                    a_t_processed = process_actions_continuous(a_t, action_dim).float()
                else: # discrete
                    a_t_processed = process_actions_discrete(a_t, action_dim).float()

                # Feature extraction (same logic as in training, but all under torch.no_grad())
                if is_jepa_base_model:
                    predicted_latent_s_t_plus_1, _, online_s_t_representation, _ = base_model(s_t, a_t_processed, s_t_plus_1)
                    action_embedding = base_model.action_embedding(a_t_processed)
                    input_to_larp = torch.cat((online_s_t_representation, predicted_latent_s_t_plus_1, action_embedding), dim=1)
                else: # Encoder-Decoder variants
                    encoded_s_t = base_model.encoder(s_t)
                    action_embedding = base_model.action_embedding(a_t_processed)
                    predicted_s_t_plus_1_img = base_model(s_t, a_t_processed)
                    predicted_s_t_plus_1_flat = predicted_s_t_plus_1_img.flatten(start_dim=1)
                    if hasattr(base_model, 'predictor') and hasattr(base_model, 'decoder') and callable(getattr(base_model, 'predictor')): # JEPAStyle
                        predictor_input_jepa_style = torch.cat((encoded_s_t, action_embedding), dim=-1)
                        intermediate_predictor_features = base_model.predictor(predictor_input_jepa_style)
                        input_to_larp = torch.cat((encoded_s_t, predicted_s_t_plus_1_flat, action_embedding, intermediate_predictor_features), dim=1)
                    else: # Standard Encoder-Decoder
                        input_to_larp = torch.cat((encoded_s_t, predicted_s_t_plus_1_flat, action_embedding), dim=1)

                pred_reward = larp_model(input_to_larp)
                val_loss = loss_fn(pred_reward, r_t.unsqueeze(1).float())
                val_loss_sum += val_loss.item()
                num_val_batches += 1
                progress_bar_val.set_postfix(loss=val_loss.item())

        avg_val_loss = val_loss_sum / num_val_batches if num_val_batches > 0 else 0

        print(f"{model_name_log_prefix} - Epoch {epoch+1}/{num_epochs_larp}: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        if wandb_run:
            wandb.log({
                f"larp/{model_name_log_prefix}/train_epoch_avg/loss": avg_train_loss,
                f"larp/{model_name_log_prefix}/val_epoch_avg/loss": avg_val_loss,
                f"larp/{model_name_log_prefix}/epoch": current_epoch_main_training * num_epochs_larp + epoch, # Unique epoch counter for LARP
                f"larp/{model_name_log_prefix}/epoch_inner": epoch + 1 # Current LARP training epoch
            })

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Optionally save best model checkpoint here if needed for LARP specifically
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"{model_name_log_prefix} - Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Finished training {model_name_log_prefix}. Best Val Loss: {best_val_loss:.4f}")
