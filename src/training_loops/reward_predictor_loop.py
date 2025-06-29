import torch
import torch.nn.functional as F # For F.one_hot
# import os # Not used in this specific function directly
# import matplotlib.pyplot as plt # Not used in this specific function directly
# import numpy as np # Not used in this specific function directly
# import time # Not used in this specific function directly

"""Handles the training loop for the reward predictor model."""

def train_reward_mlp_epoch(
    reward_mlp_model, base_model, optimizer_reward_mlp, train_dataloader,
    val_dataloader,  # Added
    loss_fn, device, action_dim, action_type,
    model_name_log_prefix, num_epochs_reward_mlp, log_interval_reward_mlp,
    early_stopping_patience,  # Added
    is_jepa_base_model, # Boolean to differentiate input processing
    wandb_run,
    reward_plotter=None  # New parameter for reward plotting
):
    """
    Handles the training process for a reward MLP (Multi-Layer Perceptron) model
    over a specified number of epochs. It now includes validation and early stopping.

    This function trains an MLP to predict rewards based on embeddings/features
    extracted from a base model (either a Standard Encoder-Decoder or a JEPA model).
    It iterates through the training data, computes losses, performs backpropagation,
    and updates the optimizer. After each epoch, it evaluates the model on a
    validation set and implements early stopping if the validation loss does not
    improve for a specified number of epochs. Training progress (loss, learning rate,
    validation loss) is logged using Weights & Biases (wandb).

    Args:
        reward_mlp_model (torch.nn.Module): The reward MLP model to be trained.
        base_model (torch.nn.Module): The base model (StdEncDec or JEPA) used for
                                      feature extraction. It's set to eval mode.
        optimizer_reward_mlp (torch.optim.Optimizer): Optimizer for the reward MLP.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_fn (callable): The loss function for training the reward MLP.
        device (torch.device): The device (CPU or GPU) for computations.
        action_dim (int): Dimension of the action space.
        action_type (str): Type of action space ('discrete' or 'continuous').
        model_name_log_prefix (str): Prefix for logging (e.g., "Reward MLP (Enc-Dec)").
        num_epochs_reward_mlp (int): Number of epochs to train the reward MLP.
        log_interval_reward_mlp (int): Interval (in batches) for logging training.
        early_stopping_patience (int): Number of epochs to wait for improvement in
                                       validation loss before stopping.
        is_jepa_base_model (bool): Flag indicating if the base_model is a JEPA model.
                                   This affects how input features are derived.
        wandb_run (wandb.sdk.wandb_run.Run, optional): Active Weights & Biases run object.
        reward_plotter (RewardPlotter, optional): Plotter for creating reward scatter plots
                                                  during validation.

    Note:
        This function doesn't return any value but modifies the `reward_mlp_model`
        in place and logs metrics to wandb.
    """
    if not (reward_mlp_model and base_model and optimizer_reward_mlp and train_dataloader and val_dataloader):
        print(f"{model_name_log_prefix}: Components missing, skipping training.")
        return

    # Determine wandb prefix based on the base model type
    wandb_model_prefix = "JEPA" if is_jepa_base_model else "StdEncDec"

    # move model to device
    reward_mlp_model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_enabled = early_stopping_patience is not None and early_stopping_patience > 0

    for epoch in range(num_epochs_reward_mlp):
        reward_mlp_model.train()
        if base_model: base_model.eval() # Base model is used for feature extraction

        epoch_loss_reward_mlp = 0
        num_train_batches = len(train_dataloader)

        if num_train_batches == 0:
            print(f"{model_name_log_prefix} Epoch {epoch+1}: No training data. Skipping.")
            continue

        for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
            s_t, r_t, s_t_plus_1 = s_t.to(device), r_t.to(device).float().unsqueeze(1), s_t_plus_1.to(device)

            if action_type == 'discrete':
                if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
                a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
            else:
                a_t_processed = a_t.float().to(device)

            optimizer_reward_mlp.zero_grad()

            input_to_reward_mlp = None
            with torch.no_grad():
                if is_jepa_base_model:
                    # For JEPA, use s_t, a_t, s_t_plus_1 to get predictor embedding
                    _, _, online_s_t_representation, _ = base_model(s_t, a_t_processed, s_t_plus_1)
                    action_embedding = base_model.action_embedding(a_t_processed)
                else: # For StdEncDec
                    online_s_t_representation = base_model.encoder(s_t)
                    action_embedding = base_model.action_embedding(a_t_processed)
                
                input_to_reward_mlp = torch.cat((online_s_t_representation, action_embedding), dim=1)

                # detach to avoid gradients flowing back to the base model
                input_to_reward_mlp = input_to_reward_mlp.detach()

            if input_to_reward_mlp is None:
                print(f"{model_name_log_prefix} Epoch {epoch+1}, Batch {batch_idx+1}: Failed to get input from base model. Skipping batch.")
                continue

            pred_reward = reward_mlp_model(input_to_reward_mlp)
            loss_reward = loss_fn(pred_reward, r_t)

            loss_reward.backward()
            optimizer_reward_mlp.step()

            loss_reward_item = loss_reward.item()
            epoch_loss_reward_mlp += loss_reward_item

            if (batch_idx + 1) % log_interval_reward_mlp == 0:
                if wandb_run:
                    log_data_reward_batch = {
                        f"reward_mlp/{wandb_model_prefix}/train/Loss": loss_reward_item,
                        f"reward_mlp/{wandb_model_prefix}/train/Learning_Rate": optimizer_reward_mlp.param_groups[0]['lr']
                    }
                    # The step current_reward_mlp_global_step should align with f"{wandb_model_prefix}/reward_mlp/train/step"
                    wandb_run.log(log_data_reward_batch)

        avg_epoch_loss_reward_mlp = epoch_loss_reward_mlp / num_train_batches if num_train_batches > 0 else 0
        # This print statement will be moved and consolidated later

        # Validation phase
        reward_mlp_model.eval()
        epoch_loss_reward_mlp_val = 0
        avg_epoch_loss_val = 0 # Default to 0 if no validation
        val_loss_display = "N/A"
        
        # Collect predictions and true values for plotting
        all_true_rewards = []
        all_pred_rewards = []

        if val_dataloader and len(val_dataloader) > 0:
            num_val_batches = len(val_dataloader)
            with torch.no_grad():
                for batch_idx_val, (s_t_val, a_t_val, r_t_val, s_t_plus_1_val) in enumerate(val_dataloader):
                    s_t_val, r_t_val, s_t_plus_1_val = s_t_val.to(device), r_t_val.to(device).float().unsqueeze(1), s_t_plus_1_val.to(device)

                    if action_type == 'discrete':
                        if a_t_val.ndim == 1:
                            a_t_val = a_t_val.unsqueeze(1)
                        a_t_processed_val = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                    else:
                        a_t_processed_val = a_t_val.float().to(device)

                    input_to_reward_mlp_val = None
                    if is_jepa_base_model:
                        # For JEPA, use s_t, a_t, s_t_plus_1 to get predictor embedding
                        _, _, online_s_t_representation_val, _ = base_model(s_t_val, a_t_processed_val, s_t_plus_1_val)
                        action_embedding_val = base_model.action_embedding(a_t_processed_val)
                    else: # For StdEncDec
                        online_s_t_representation_val = base_model.encoder(s_t_val)
                        action_embedding_val = base_model.action_embedding(a_t_processed_val)

                    input_to_reward_mlp_val = torch.cat((online_s_t_representation_val, action_embedding_val), dim=1)

                    # detach to avoid gradients flowing back to the base model
                    input_to_reward_mlp_val = input_to_reward_mlp_val.detach()

                    if input_to_reward_mlp_val is None:
                        print(f"{model_name_log_prefix} Epoch {epoch+1}, Val Batch {batch_idx_val+1}: Failed to get input from base model. Skipping batch.")
                        continue

                    pred_reward_val = reward_mlp_model(input_to_reward_mlp_val)
                    loss_reward_val = loss_fn(pred_reward_val, r_t_val)
                    epoch_loss_reward_mlp_val += loss_reward_val.item()
                    
                    # Collect predictions and true rewards for plotting
                    all_true_rewards.append(r_t_val.cpu())
                    all_pred_rewards.append(pred_reward_val.cpu())

            avg_epoch_loss_val = epoch_loss_reward_mlp_val / num_val_batches if num_val_batches > 0 else 0
            val_loss_display = f"{avg_epoch_loss_val:.4f}"
            
            # Create reward scatter plot if plotter is provided
            if reward_plotter and all_true_rewards and all_pred_rewards:
                true_rewards_tensor = torch.cat(all_true_rewards, dim=0)
                pred_rewards_tensor = torch.cat(all_pred_rewards, dim=0)
                
                model_type = "JEPA" if is_jepa_base_model else "Encoder-Decoder"
                reward_plotter.plot_reward_scatter(
                    true_rewards=true_rewards_tensor,
                    predicted_rewards=pred_rewards_tensor,
                    epoch=epoch + 1,
                    model_name=model_type
                )

        # Consolidated print statement for epoch losses
        print(f"  Epoch {epoch+1}/{num_epochs_reward_mlp} Avg Train Loss: {avg_epoch_loss_reward_mlp:.4f}, Avg Val Loss: {val_loss_display}")

        if wandb_run:
            log_data_epoch = {
                f"reward_mlp/{wandb_model_prefix}/train_epoch_avg/Loss": avg_epoch_loss_reward_mlp
            }
            if val_dataloader and len(val_dataloader) > 0: # Only log val loss if it was calculated
                log_data_epoch[f"reward_mlp/{wandb_model_prefix}/val_epoch_avg/Loss"] = avg_epoch_loss_val
            
            # The step current_reward_mlp_epoch should align with f"{wandb_model_prefix}/reward_mlp/epoch"
            wandb_run.log(log_data_epoch) # This will commit train and (if available) val loss together

        # Early stopping logic
        if early_stopping_enabled and val_dataloader and len(val_dataloader) > 0:
            if avg_epoch_loss_val < best_val_loss:
                best_val_loss = avg_epoch_loss_val
                patience_counter = 0
                # Optional: Save best model checkpoint here
                # torch.save(reward_mlp_model.state_dict(), f"{model_name_log_prefix}_best.pth")
                # print(f"  Epoch {epoch+1}: New best validation loss: {best_val_loss:.4f}. Model saved.")
            else:
                patience_counter += 1
                print(f"  Epoch {epoch+1}: Val loss {avg_epoch_loss_val:.4f} did not improve from {best_val_loss:.4f}. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss for {early_stopping_patience} epochs.")
                break  # Exit the training loop

    print(f"{model_name_log_prefix} training finished.")
    # Optionally return last epoch's average loss or a status
    # For now, no explicit return value is critical for the flow.
