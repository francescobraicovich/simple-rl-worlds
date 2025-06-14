# Contents for src/training_engine.py
import torch
import torch.nn.functional as F # For F.one_hot
import os # For os.path.exists in early stopping save/load
import matplotlib.pyplot as plt
import numpy as np
import time
import wandb # For wandb.Image

# Note: Loss functions (mse_loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight)
# will be passed in via the 'losses_map' dictionary.
# Models (std_enc_dec, jepa_model, reward_mlp_enc_dec, reward_mlp_jepa, jepa_decoder) via 'models_map'.
# Optimizers (optimizer_std_enc_dec, etc.) via 'optimizers_map'.
# Dataloaders (train_dataloader, val_dataloader) via 'dataloaders_map'.
# Configs (early_stopping_config, enc_dec_mlp_config, jepa_mlp_config, main_config for num_epochs, log_interval) via 'config'.
# Device, action_dim, action_type also passed as arguments.

def _train_validate_model_epoch(
    model, optimizer, train_dataloader, val_dataloader,
    loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight,
    device, epoch_num, log_interval, action_dim, action_type,
    early_stopping_state, checkpoint_path, model_name_log_prefix,
    wandb_run,
    update_target_fn=None
):
    """
    Handles training and validation for one epoch for a given model.
    Logs metrics with the structure: model_name/phase/metric_name
    Returns updated early_stopping_state and epoch losses.
    """
    # === Training Phase ===
    model.train()
    if aux_loss_fn and hasattr(aux_loss_fn, 'train'):
        aux_loss_fn.train()
    model.to(device)

    epoch_train_loss_primary, epoch_train_loss_aux = 0, 0
    num_train_batches = len(train_dataloader) if train_dataloader else 0

    if num_train_batches == 0:
        print(f"{model_name_log_prefix} Epoch {epoch_num}: No training data. Skipping.")
        return early_stopping_state, 0, 0

    for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
        s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
        if action_type == 'discrete':
            if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
            a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
        else:
            a_t_processed = a_t.float().to(device)

        optimizer.zero_grad()

        current_loss_primary_item, current_loss_aux_item, total_loss_item = 0, 0, 0
        diff_metric = 0.0

        if model_name_log_prefix == "JEPA":
            # Dummy logic for JEPA model forward pass
            pred_emb, target_emb_detached, online_s_t_emb, _ = model(s_t, a_t_processed, s_t_plus_1)
            loss_primary = loss_fn(pred_emb, target_emb_detached)
            current_loss_primary_item = loss_primary.item()
            current_loss_aux = torch.tensor(0.0, device=device)
            if aux_loss_fn is not None and aux_loss_weight > 0:
                aux_term_s_t, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_emb)
                current_loss_aux = aux_term_s_t
                current_loss_aux_item = current_loss_aux.item()
            total_loss = loss_primary + current_loss_aux * aux_loss_weight
        else:  # Standard Encoder-Decoder
            predicted_s_t_plus_1 = model(s_t, a_t_processed)
            loss_primary = loss_fn(predicted_s_t_plus_1, s_t_plus_1)
            current_loss_primary_item = loss_primary.item()
            total_loss = loss_primary

        total_loss_item = total_loss.item()
        total_loss.backward()
        optimizer.step()

        if update_target_fn:
            update_target_fn()
            if model_name_log_prefix == "JEPA" and hasattr(model, 'online_encoder'):
                num_params = 0
                for p_online, p_target in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
                    diff_metric += F.mse_loss(p_online, p_target, reduction='sum').item()
                    num_params += p_online.numel()
                if num_params > 0: diff_metric /= num_params

        epoch_train_loss_primary += current_loss_primary_item
        epoch_train_loss_aux += current_loss_aux_item

        if (batch_idx + 1) % log_interval == 0:
            log_data = {
                f"{model_name_log_prefix}/train/Prediction_Loss": current_loss_primary_item,
                f"{model_name_log_prefix}/train/Total_Loss": total_loss_item,
                f"{model_name_log_prefix}/train/Learning_Rate": optimizer.param_groups[0]['lr']
            }
            if model_name_log_prefix == "JEPA":
                if aux_loss_fn and aux_loss_weight > 0:
                    log_data[f"{model_name_log_prefix}/train/Aux_Raw_Loss"] = current_loss_aux_item
                    log_data[f"{model_name_log_prefix}/train/Aux_Weighted_Loss"] = current_loss_aux_item * aux_loss_weight
                if hasattr(model, 'online_encoder'):
                    log_data[f"{model_name_log_prefix}/train/Encoder_Weight_Diff_MSE"] = diff_metric

            if wandb_run:
                current_global_step = (epoch_num - 1) * num_train_batches + batch_idx
                log_data[f"{model_name_log_prefix}/train/step"] = current_global_step
                wandb_run.log(log_data)

    avg_epoch_train_loss_primary = epoch_train_loss_primary / num_train_batches
    avg_epoch_train_loss_aux = epoch_train_loss_aux / num_train_batches if aux_loss_fn and aux_loss_weight > 0 else 0

    # === Validation and Epoch Summary Phase ===
    if val_dataloader:
        model.eval()
        if aux_loss_fn and hasattr(aux_loss_fn, 'eval'):
            aux_loss_fn.eval()

        epoch_val_loss_primary, epoch_val_loss_aux = 0, 0
        num_val_batches = len(val_dataloader)

        with torch.no_grad():
            for s_t_val, a_t_val, r_t_val, s_t_plus_1_val in val_dataloader:
                s_t_val, s_t_plus_1_val = s_t_val.to(device), s_t_plus_1_val.to(device)
                if action_type == 'discrete':
                    if a_t_val.ndim == 1: a_t_val = a_t_val.unsqueeze(1)
                    a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_val_processed = a_t_val.float().to(device)

                val_loss_primary_item, val_loss_aux_item = 0, 0

                if model_name_log_prefix == "JEPA":
                    pred_emb_val, target_emb_detached_val, online_s_t_emb_val, _ = model(s_t_val, a_t_val_processed, s_t_plus_1_val)
                    val_loss_primary = loss_fn(pred_emb_val, target_emb_detached_val)
                    val_loss_primary_item = val_loss_primary.item()
                    if aux_loss_fn is not None and aux_loss_weight > 0:
                        aux_term_s_t_val, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_emb_val)
                        val_loss_aux_item = aux_term_s_t_val.item()
                else: # Standard Encoder-Decoder
                    predicted_s_t_plus_1_val = model(s_t_val, a_t_val_processed)
                    val_loss_primary = loss_fn(predicted_s_t_plus_1_val, s_t_plus_1_val)
                    val_loss_primary_item = val_loss_primary.item()

                epoch_val_loss_primary += val_loss_primary_item
                epoch_val_loss_aux += val_loss_aux_item

        avg_val_loss_primary = epoch_val_loss_primary / num_val_batches if num_val_batches > 0 else float('inf')
        avg_val_loss_aux_raw = epoch_val_loss_aux / num_val_batches if num_val_batches > 0 and aux_loss_fn and aux_loss_weight > 0 else 0

        # Consolidated Epoch Logging
        log_epoch_summary = {}
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Prediction_Loss"] = avg_epoch_train_loss_primary
        log_epoch_summary[f"{model_name_log_prefix}/val/Prediction_Loss"] = avg_val_loss_primary
        current_total_val_loss = avg_val_loss_primary

        if model_name_log_prefix == "JEPA" and aux_loss_fn and aux_loss_weight > 0:
            avg_total_train_loss = avg_epoch_train_loss_primary + (avg_epoch_train_loss_aux * aux_loss_weight)
            log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Aux_Raw_Loss"] = avg_epoch_train_loss_aux
            log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Total_Loss"] = avg_total_train_loss
            
            current_total_val_loss += avg_val_loss_aux_raw * aux_loss_weight
            log_epoch_summary[f"{model_name_log_prefix}/val/Aux_Raw_Loss"] = avg_val_loss_aux_raw
            log_epoch_summary[f"{model_name_log_prefix}/val/Aux_Weighted_Loss"] = avg_val_loss_aux_raw * aux_loss_weight
            log_epoch_summary[f"{model_name_log_prefix}/val/Total_Loss"] = current_total_val_loss

        print(f"  {'Avg Train Loss':<22}: {avg_epoch_train_loss_primary:>8.4f} | {'Avg Val Loss':<22}: {avg_val_loss_primary:>8.4f}")
        if model_name_log_prefix == "JEPA" and aux_loss_fn and aux_loss_weight > 0:
            print(f"  {'Avg Train Aux Loss':<22}: {avg_epoch_train_loss_aux:>8.4f} | {'Avg Val Aux Loss':<22}: {avg_val_loss_aux_raw:>8.4f}")
            print(f"  {'Avg Train Total Loss':<22}: {avg_total_train_loss:>8.4f} | {'Avg Val Total Loss':<22}: {current_total_val_loss:>8.4f}")

        if wandb_run:
            log_epoch_summary[f"{model_name_log_prefix}/epoch"] = epoch_num
            wandb_run.log(log_epoch_summary)
            
        # Early Stopping Logic
        if current_total_val_loss < early_stopping_state['best_val_loss'] - early_stopping_state.get('delta', 0.001):
            early_stopping_state['best_val_loss'] = current_total_val_loss
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                early_stopping_state['epochs_no_improve'] = 0
                print(f"  Val loss improved. Saved model to {checkpoint_path}")
        else:
            early_stopping_state['epochs_no_improve'] += 1
            print(f"  No val improvement for {early_stopping_state['epochs_no_improve']} epochs.")
            if early_stopping_state['epochs_no_improve'] >= early_stopping_state.get('patience', 10):
                early_stopping_state['early_stop_flag'] = True
                print(f"  Early stopping triggered.")

    return early_stopping_state, avg_epoch_train_loss_primary, avg_epoch_train_loss_aux


def _train_reward_mlp_epoch(
    reward_mlp_model, base_model, optimizer_reward_mlp, train_dataloader,
    loss_fn, device, action_dim, action_type,
    model_name_log_prefix, num_epochs_reward_mlp, log_interval_reward_mlp,
    is_jepa_base_model, # Boolean to differentiate input processing
    wandb_run
):
    """
    Handles training for a reward MLP model for a specified number of epochs.
    """
    if not (reward_mlp_model and base_model and optimizer_reward_mlp and train_dataloader):
        print(f"{model_name_log_prefix}: Components missing, skipping training.")
        return

    print(f"\nStarting training for {model_name_log_prefix} for {num_epochs_reward_mlp} epochs...")

    # Determine wandb prefix based on the base model type
    wandb_model_prefix = "JEPA" if is_jepa_base_model else "StdEncDec"

    # move model to device
    reward_mlp_model.to(device)

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
                    pred_emb_for_reward, _, _, _ = base_model(s_t, a_t_processed, s_t_plus_1)
                    input_to_reward_mlp = pred_emb_for_reward.detach()
                else: # For StdEncDec
                    predicted_s_t_plus_1_for_reward = base_model(s_t, a_t_processed)
                    input_to_reward_mlp = predicted_s_t_plus_1_for_reward.view(predicted_s_t_plus_1_for_reward.size(0), -1).detach()

            if input_to_reward_mlp is None:
                print(f"{model_name_log_prefix} Epoch {epoch+1}, Batch {batch_idx+1}: Failed to get input from base model. Skipping batch.")
                continue

            pred_reward = reward_mlp_model(input_to_reward_mlp)
            loss_reward = loss_fn(pred_reward, r_t)

            loss_reward.backward()
            optimizer_reward_mlp.step()

            loss_reward_item = loss_reward.item()
            epoch_loss_reward_mlp += loss_reward_item

            current_reward_mlp_global_step = epoch * num_train_batches + batch_idx
            if (batch_idx + 1) % log_interval_reward_mlp == 0:
                print(f"  {model_name_log_prefix} Epoch {epoch+1}, Batch {batch_idx+1}/{num_train_batches}: Loss {loss_reward_item:.4f}")
                if wandb_run:
                    log_data_reward_batch = {
                        f"{wandb_model_prefix}/reward_mlp/train/Loss": loss_reward_item,
                        f"{wandb_model_prefix}/reward_mlp/train/Learning_Rate": optimizer_reward_mlp.param_groups[0]['lr']
                    }
                    # The step current_reward_mlp_global_step should align with f"{wandb_model_prefix}/reward_mlp/train/step"
                    wandb_run.log(log_data_reward_batch, step=current_reward_mlp_global_step)

        avg_epoch_loss_reward_mlp = epoch_loss_reward_mlp / num_train_batches if num_train_batches > 0 else 0
        print(f"--- {model_name_log_prefix} Epoch {epoch+1}/{num_epochs_reward_mlp} Summary: Avg Train Loss {avg_epoch_loss_reward_mlp:.4f} ---")
        if wandb_run:
            # Log using epoch + 1 to align with f"{wandb_model_prefix}/reward_mlp/epoch"
            wandb_run.log({
                f"{wandb_model_prefix}/reward_mlp/train_epoch_avg/Loss": avg_epoch_loss_reward_mlp
            }, step=epoch + 1)

    print(f"{model_name_log_prefix} training finished.")
    # Optionally return last epoch's average loss or a status
    # For now, no explicit return value is critical for the flow.

def _train_jepa_state_decoder(
    jepa_decoder_model, jepa_model, optimizer_jepa_decoder,
    train_dataloader, val_dataloader, loss_fn, device,
    action_dim, action_type,
    decoder_training_config, # This is jepa_decoder_training_config from the main function
    main_model_dir, # This is model_dir from the main function, for paths
    general_log_interval, # Fallback log interval
    wandb_run # New argument for wandb
):
    """
    Handles training for the JEPA State Decoder model.
    Returns the path to the best saved checkpoint, or None.
    """
    print("\nStarting JEPA State Decoder training...")

    # move model to device
    jepa_decoder_model.to(device)

    # Extract params from decoder_training_config
    num_epochs_decoder = decoder_training_config.get('num_epochs', 50)
    decoder_log_interval = decoder_training_config.get('log_interval', general_log_interval)

    early_stopping_specific_config = decoder_training_config.get('early_stopping', {})
    # Fallback to general patience/delta if not specifically in decoder_training_config.early_stopping
    # This requires access to the general patience/delta, or they should be passed in,
    # or defined within this scope if they are meant to be independent.
    # For now, let's assume patience/delta are in early_stopping_specific_config or have defaults.
    patience_decoder = early_stopping_specific_config.get('patience', 10) # Default if not in specific config
    delta_decoder = early_stopping_specific_config.get('delta', 0.001)   # Default if not in specific config

    decoder_cp_name = decoder_training_config.get('checkpoint_path', 'best_jepa_decoder.pth')
    # Construct full checkpoint path using main_model_dir
    checkpoint_path_decoder = os.path.join(main_model_dir, decoder_cp_name)

    # Plotting directory from config
    validation_plot_dir_config = decoder_training_config.get('validation_plot_dir', "validation_plots/decoder")
    validation_plot_dir_full = os.path.join(main_model_dir, validation_plot_dir_config) # Construct full path

    early_stopping_state_decoder = {
        'best_val_loss': float('inf'),
        'epochs_no_improve': 0,
        'early_stop_flag': False,
        'patience': patience_decoder,
        'delta': delta_decoder,
        'checkpoint_path': checkpoint_path_decoder # Full path
    }

    for epoch in range(num_epochs_decoder):
        if early_stopping_state_decoder['early_stop_flag']:
            print(f"JEPA State Decoder early stopping triggered before epoch {epoch+1}. Exiting decoder training loop.")
            break

        print(f"\n--- JEPA State Decoder Epoch {epoch+1}/{num_epochs_decoder} ---")
        if jepa_model: jepa_model.eval() # Main JEPA model provides embeddings
        jepa_decoder_model.train()

        epoch_loss_train = 0
        num_batches_train = len(train_dataloader) if train_dataloader else 0

        if num_batches_train == 0:
            print(f"JEPA Decoder Epoch {epoch+1} has no training data. Skipping training phase.")
        else:
            for batch_idx, (s_t, a_t, _, s_t_plus_1) in enumerate(train_dataloader):
                s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
                if action_type == 'discrete':
                    a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device) if a_t.ndim == 1 else F.one_hot(a_t.long(), num_classes=action_dim).float().to(device)
                    if a_t_processed.shape[0] != s_t.shape[0]:
                         a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_processed = a_t.float().to(device)

                optimizer_jepa_decoder.zero_grad()
                with torch.no_grad():
                    if not jepa_model: # Should be caught by caller, but good to check
                        print("Error: Main JEPA model is None for JEPA decoder training.")
                        early_stopping_state_decoder['early_stop_flag'] = True; break
                    pred_emb, _, _, _ = jepa_model(s_t, a_t_processed, s_t_plus_1)

                jepa_predictor_output = pred_emb.detach()
                reconstructed_s_t_plus_1 = jepa_decoder_model(jepa_predictor_output)
                loss = loss_fn(reconstructed_s_t_plus_1, s_t_plus_1)
                loss.backward()
                optimizer_jepa_decoder.step()
                epoch_loss_train += loss.item()

                current_decoder_global_step = epoch * num_batches_train + batch_idx
                if (batch_idx + 1) % decoder_log_interval == 0:
                    print(f"  JEPA Decoder Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches_train}: Train Loss {loss.item():.4f}")
                    if wandb_run:
                        log_data_decoder_batch = {
                            "JEPA_Decoder/train/Loss": loss.item(),
                            "JEPA_Decoder/train/Learning_Rate": optimizer_jepa_decoder.param_groups[0]['lr']
                        }
                        # step current_decoder_global_step aligns with "JEPA_Decoder/train/step"
                        wandb_run.log(log_data_decoder_batch, step=current_decoder_global_step)
            if early_stopping_state_decoder['early_stop_flag']: break # From error in batch loop

        avg_train_loss = epoch_loss_train / num_batches_train if num_batches_train > 0 else 0
        print(f"  Avg Train JEPA Decoder L (Epoch {epoch+1}): {avg_train_loss:.4f}")
        if wandb_run:
            wandb_run.log({
                "JEPA_Decoder/train_epoch_avg/Loss": avg_train_loss
            }, step=epoch + 1) # step epoch + 1 aligns with "JEPA_Decoder/epoch"

        # Validation Phase
        if val_dataloader:
            jepa_decoder_model.eval()
            if jepa_model: jepa_model.eval()

            epoch_loss_val = 0
            num_batches_val = len(val_dataloader)
            with torch.no_grad():
                for val_batch_idx, (s_t_val, a_t_val, _, s_t_plus_1_val) in enumerate(val_dataloader):
                    s_t_val, s_t_plus_1_val = s_t_val.to(device), s_t_plus_1_val.to(device)
                    if action_type == 'discrete':
                        a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device) if a_t_val.ndim == 1 else F.one_hot(a_t_val.long(), num_classes=action_dim).float().to(device)
                        if a_t_val_processed.shape[0] != s_t_val.shape[0]:
                           a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                    else:
                        a_t_val_processed = a_t_val.float().to(device)

                    if not jepa_model:
                         print("Error: Main JEPA model is None during JEPA decoder validation.")
                         early_stopping_state_decoder['early_stop_flag'] = True; break
                    pred_emb_val, _, _, _ = jepa_model(s_t_val, a_t_val_processed, s_t_plus_1_val)
                    jepa_predictor_output_val = pred_emb_val.detach()
                    reconstructed_s_t_plus_1_val = jepa_decoder_model(jepa_predictor_output_val)
                    val_loss = loss_fn(reconstructed_s_t_plus_1_val, s_t_plus_1_val)
                    epoch_loss_val += val_loss.item()

                    # Plotting logic
                    if val_batch_idx == 0 and decoder_training_config.get('enable_validation_plot', False):
                        os.makedirs(validation_plot_dir_full, exist_ok=True)
                        num_plot_samples = min(4, s_t_val.shape[0])
                        random_indices = np.random.choice(s_t_val.shape[0], num_plot_samples, replace=False) if s_t_val.shape[0] > num_plot_samples else range(s_t_val.shape[0])
                        for i in random_indices:
                            curr_img_np = s_t_val[i].cpu().numpy()
                            true_img_np = s_t_plus_1_val[i].cpu().numpy()
                            pred_img_np = reconstructed_s_t_plus_1_val[i].cpu().numpy()

                            # Process all three images consistently
                            processed_images = []
                            for img_np in [curr_img_np, true_img_np, pred_img_np]:
                                if img_np.shape[0] == 1 or img_np.shape[0] == 3: # C, H, W
                                    img_np = np.transpose(img_np, (1, 2, 0))
                                if img_np.shape[-1] == 1: # Grayscale, squeeze channel
                                    img_np = img_np.squeeze(axis=2)
                                if img_np.dtype == np.float32 or img_np.dtype == np.float64: # Clip if float
                                    img_np = np.clip(img_np, 0, 1)
                                processed_images.append(img_np)
                            
                            curr_img_processed, true_img_processed, pred_img_processed = processed_images

                            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                            axes[0].imshow(curr_img_processed); axes[0].set_title("Current State (s_t)"); axes[0].axis('off')
                            axes[1].imshow(true_img_processed); axes[1].set_title("True Next State (s_{t+1})"); axes[1].axis('off')
                            axes[2].imshow(pred_img_processed); axes[2].set_title("Predicted Next State (s'_{t+1})"); axes[2].axis('off')
                            
                            plot_filename = os.path.join(validation_plot_dir_full, f"epoch_{epoch+1}_sample_{i}_comparison.png")
                            plt.savefig(plot_filename)
                            if wandb_run:
                                try:
                                    wandb_run.log({f"JEPA_Decoder/val/Validation_Comparison_Sample_{i}": wandb.Image(fig)}, step=epoch + 1)
                                except Exception as e:
                                    print(f"Warning: Failed to log image to wandb: {e}")
                            plt.close(fig)
                        print(f"  JEPA Decoder: Saved {num_plot_samples} validation image samples to {validation_plot_dir_full}")
                if early_stopping_state_decoder['early_stop_flag']: break

                avg_val_loss = epoch_loss_val / num_batches_val if num_batches_val > 0 else float('inf')
                print(f"--- JEPA Decoder Epoch {epoch+1} Validation Summary ---")
                print(f"  Avg Val JEPA Decoder L: {avg_val_loss:.4f}")
                if wandb_run:
                    wandb_run.log({
                        "JEPA_Decoder/val/Loss": avg_val_loss
                    }, step=epoch + 1) # step epoch + 1 aligns with "JEPA_Decoder/epoch"

                if avg_val_loss < early_stopping_state_decoder['best_val_loss'] - early_stopping_state_decoder['delta']:
                    early_stopping_state_decoder['best_val_loss'] = avg_val_loss
                    if early_stopping_state_decoder['checkpoint_path']:
                        os.makedirs(os.path.dirname(early_stopping_state_decoder['checkpoint_path']), exist_ok=True)
                        torch.save(jepa_decoder_model.state_dict(), early_stopping_state_decoder['checkpoint_path'])
                    early_stopping_state_decoder['epochs_no_improve'] = 0
                    print(f"  JEPA Decoder: Val loss improved. Saved model to {early_stopping_state_decoder['checkpoint_path']}")
                else:
                    early_stopping_state_decoder['epochs_no_improve'] += 1
                    print(f"  JEPA Decoder: No val improvement for {early_stopping_state_decoder['epochs_no_improve']} epochs.")
                    if early_stopping_state_decoder['epochs_no_improve'] >= early_stopping_state_decoder['patience']:
                        early_stopping_state_decoder['early_stop_flag'] = True
                        print("  JEPA Decoder: Early stopping triggered.")
        else: # No validation dataloader
            print(f"--- JEPA Decoder Epoch {epoch+1} Training Summary (No Validation) ---")
            if early_stopping_state_decoder['checkpoint_path']: # Save last epoch if no validation
                os.makedirs(os.path.dirname(early_stopping_state_decoder['checkpoint_path']), exist_ok=True)
                torch.save(jepa_decoder_model.state_dict(), early_stopping_state_decoder['checkpoint_path'])
                print(f"  JEPA Decoder: Saved model from last epoch to {early_stopping_state_decoder['checkpoint_path']} (no validation set)")

        if early_stopping_state_decoder['early_stop_flag']: break

    print("JEPA State Decoder training finished.")

    # Load the best model if it was saved
    best_checkpoint_file = early_stopping_state_decoder['checkpoint_path']
    if best_checkpoint_file and os.path.exists(best_checkpoint_file):
        print(f"Loading best JEPA State Decoder model from {best_checkpoint_file}")
        jepa_decoder_model.load_state_dict(torch.load(best_checkpoint_file, map_location=device))
        return best_checkpoint_file
    elif best_checkpoint_file: # Path was set, but no file (e.g. no training/val epochs ran or never improved)
         print(f"JEPA State Decoder checkpoint {best_checkpoint_file} not found. Using model state as is.")
         return None
    return None


def run_training_epochs(
    models_map, optimizers_map, losses_map, dataloaders_map,
    device, config, action_dim, action_type,
    image_h_w, input_channels, # For reward MLP input calculation if needed
    std_enc_dec_loaded_successfully=False,
    jepa_loaded_successfully=False,
    wandb_run=None
):
    # Unpack from maps/config for convenience
    std_enc_dec = models_map.get('std_enc_dec')
    jepa_model = models_map.get('jepa')
    reward_mlp_enc_dec = models_map.get('reward_mlp_enc_dec')
    reward_mlp_jepa = models_map.get('reward_mlp_jepa')
    jepa_decoder = models_map.get('jepa_decoder')

    optimizer_std_enc_dec = optimizers_map.get('std_enc_dec')
    optimizer_jepa = optimizers_map.get('jepa')
    optimizer_reward_mlp_enc_dec = optimizers_map.get('reward_mlp_enc_dec')
    optimizer_reward_mlp_jepa = optimizers_map.get('reward_mlp_jepa')
    optimizer_jepa_decoder = optimizers_map.get('jepa_decoder')

    mse_loss_fn = losses_map['mse']
    aux_loss_fn = losses_map.get('aux_fn')
    aux_loss_name = losses_map.get('aux_name', "None")
    aux_loss_weight = losses_map.get('aux_weight', 0.0)

    train_dataloader = dataloaders_map['train']
    val_dataloader = dataloaders_map.get('val')

    # Configs
    early_stopping_config = config.get('early_stopping', {})
    model_dir = config.get('model_dir', 'trained_models/')
    patience = early_stopping_config.get('patience', 10)
    delta = early_stopping_config.get('delta', 0.001)

    # Checkpoint paths
    checkpoint_path_enc_dec = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_enc_dec', 'best_encoder_decoder.pth'))
    checkpoint_path_jepa = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_jepa', 'best_jepa.pth'))

    enc_dec_mlp_config = config.get('reward_predictors', {}).get('encoder_decoder_reward_mlp', {})
    jepa_mlp_config = config.get('reward_predictors', {}).get('jepa_reward_mlp', {})

    num_epochs = config.get('num_epochs', 10)
    log_interval = config.get('log_interval', 50)

    # Define custom x-axes for wandb logging
    if wandb_run:
        for model_prefix in ["StdEncDec", "JEPA"]:
            # Metrics for the main model (StdEncDec or JEPA)
            wandb_run.define_metric(f"{model_prefix}/train/step")
            wandb_run.define_metric(f"{model_prefix}/epoch")
            wandb_run.define_metric(f"{model_prefix}/train/*", step_metric=f"{model_prefix}/train/step")
            wandb_run.define_metric(f"{model_prefix}/val/*", step_metric=f"{model_prefix}/epoch")
            wandb_run.define_metric(f"{model_prefix}/train_epoch_avg/*", step_metric=f"{model_prefix}/epoch")

            # Metrics for the reward MLP associated with the main model
            wandb_run.define_metric(f"reward_mlp/{model_prefix}/train/step")
            wandb_run.define_metric(f"reward_mlp/{model_prefix}/epoch")
            wandb_run.define_metric(f"reward_mlp/{model_prefix}/train/*", step_metric=f"reward_mlp/{model_prefix}/train/step")
            wandb_run.define_metric(f"reward_mlp/{model_prefix}/val/*", step_metric=f"reward_mlp/{model_prefix}/epoch")
            wandb_run.define_metric(f"reward_mlp/{model_prefix}/train_epoch_avg/*", step_metric=f"reward_mlp/{model_prefix}/epoch")

        # Define metrics for JEPA_Decoder
        wandb_run.define_metric("JEPA_Decoder/train/step")      # Step for batch-wise train logs
        wandb_run.define_metric("JEPA_Decoder/epoch")           # Step for epoch-wise logs
        wandb_run.define_metric("JEPA_Decoder/train/*", step_metric="JEPA_Decoder/train/step")
        wandb_run.define_metric("JEPA_Decoder/val/*", step_metric="JEPA_Decoder/epoch")
        wandb_run.define_metric("JEPA_Decoder/train_epoch_avg/*", step_metric="JEPA_Decoder/epoch")

    print(f"Starting training, main models for up to {num_epochs} epochs...")

    # Initialize early_stopping_state dictionaries
    early_stopping_state_enc_dec = {
        'best_val_loss': float('inf'), 'epochs_no_improve': 0,
        'early_stop_flag': not (std_enc_dec and optimizer_std_enc_dec),
        'patience': early_stopping_config.get('patience_enc_dec', patience),
        'delta': early_stopping_config.get('delta_enc_dec', delta),
        'checkpoint_path': checkpoint_path_enc_dec
    }
    early_stopping_state_jepa = {
        'best_val_loss': float('inf'), 'epochs_no_improve': 0,
        'early_stop_flag': not (jepa_model and optimizer_jepa),
        'patience': early_stopping_config.get('patience_jepa', patience),
        'delta': early_stopping_config.get('delta_jepa', delta),
        'checkpoint_path': checkpoint_path_jepa
    }

    # Handle Skip Training Flags
    training_options = config.get('training_options', {})
    skip_std_enc_dec_opt = training_options.get('skip_std_enc_dec_training_if_loaded', False)
    skip_jepa_opt = training_options.get('skip_jepa_training_if_loaded', False)

    if std_enc_dec_loaded_successfully and skip_std_enc_dec_opt and not early_stopping_state_enc_dec['early_stop_flag']:
        early_stopping_state_enc_dec['early_stop_flag'] = True
        print("Standard Encoder/Decoder training will be skipped as a pre-trained model was loaded and skip option is enabled.")
    if jepa_loaded_successfully and skip_jepa_opt and not early_stopping_state_jepa['early_stop_flag']:
        early_stopping_state_jepa['early_stop_flag'] = True
        print("JEPA model training will be skipped as a pre-trained model was loaded and skip option is enabled.")

    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")

        # --- Standard Encoder/Decoder Training ---
        if std_enc_dec and not early_stopping_state_enc_dec['early_stop_flag']:
            print(f"Standard Encoder/Decoder: Running training and validation for (Epoch {epoch+1})...")
            early_stopping_state_enc_dec, _, _ = _train_validate_model_epoch(
                model=std_enc_dec, optimizer=optimizer_std_enc_dec, train_dataloader=train_dataloader,
                val_dataloader=val_dataloader, loss_fn=mse_loss_fn, aux_loss_fn=None, aux_loss_name="N/A",
                aux_loss_weight=0, device=device, epoch_num=epoch + 1, log_interval=log_interval,
                action_dim=action_dim, action_type=action_type, early_stopping_state=early_stopping_state_enc_dec,
                checkpoint_path=early_stopping_state_enc_dec['checkpoint_path'], model_name_log_prefix="StdEncDec",
                wandb_run=wandb_run
            )
        elif not std_enc_dec:
             if not early_stopping_state_enc_dec['early_stop_flag']: print(f"StdEncDec model not provided, skipping epoch {epoch+1}.")
             early_stopping_state_enc_dec['early_stop_flag'] = True
        elif early_stopping_state_enc_dec['early_stop_flag']:
            print(f"StdEncDec training already early stopped or skipped. Skipping epoch {epoch+1}.")
        print('')

        # --- JEPA Model Training ---
        if jepa_model and not early_stopping_state_jepa['early_stop_flag']:
            print(f"JEPA: Running training and validation for (Epoch {epoch+1})...")
            update_fn = getattr(jepa_model, 'perform_ema_update', None)
            early_stopping_state_jepa, _, _ = _train_validate_model_epoch(
                model=jepa_model, optimizer=optimizer_jepa, train_dataloader=train_dataloader,
                val_dataloader=val_dataloader, loss_fn=mse_loss_fn, aux_loss_fn=aux_loss_fn,
                aux_loss_name=aux_loss_name, aux_loss_weight=aux_loss_weight, device=device,
                epoch_num=epoch + 1, log_interval=log_interval, action_dim=action_dim, action_type=action_type,
                early_stopping_state=early_stopping_state_jepa, checkpoint_path=early_stopping_state_jepa['checkpoint_path'],
                model_name_log_prefix="JEPA", wandb_run=wandb_run,
                update_target_fn=update_fn
            )
        elif not jepa_model:
            if not early_stopping_state_jepa['early_stop_flag']: print(f"JEPA model not provided, skipping epoch {epoch+1}.")
            early_stopping_state_jepa['early_stop_flag'] = True
        elif early_stopping_state_jepa['early_stop_flag']:
            print(f"JEPA training already early stopped or skipped. Skipping epoch {epoch+1}.")

        if early_stopping_state_enc_dec['early_stop_flag'] and early_stopping_state_jepa['early_stop_flag']:
            print("Both main models have triggered early stopping or were skipped. Halting training.")
            break

    print("\nMain models training loop finished.")

    # --- Reward MLP Training Loop ---
    # Load best models if they were saved via early stopping, for subsequent Reward MLP and JEPA Decoder training
    if std_enc_dec and early_stopping_state_enc_dec['checkpoint_path'] and os.path.exists(early_stopping_state_enc_dec['checkpoint_path']):
        print(f"Loading best Standard Encoder/Decoder model from {early_stopping_state_enc_dec['checkpoint_path']} for subsequent tasks.")
        std_enc_dec.load_state_dict(torch.load(early_stopping_state_enc_dec['checkpoint_path'], map_location=device))
    if jepa_model and early_stopping_state_jepa['checkpoint_path'] and os.path.exists(early_stopping_state_jepa['checkpoint_path']):
        print(f"Loading best JEPA model from {early_stopping_state_jepa['checkpoint_path']} for subsequent tasks.")
        jepa_model.load_state_dict(torch.load(early_stopping_state_jepa['checkpoint_path'], map_location=device))


    # (Original Reward MLP and JEPA State Decoder training loops follow)
    # Ensure these loops use the potentially reloaded best models.
    # The rest of the code for Reward MLP and JEPA State Decoder training will remain here.
    # --- Reward MLP Training using _train_reward_mlp_epoch ---
    if reward_mlp_enc_dec and enc_dec_mlp_config.get('enabled', False) and optimizer_reward_mlp_enc_dec and std_enc_dec and train_dataloader:
        _train_reward_mlp_epoch(
            reward_mlp_model=reward_mlp_enc_dec,
            base_model=std_enc_dec, # This is the best loaded std_enc_dec
            optimizer_reward_mlp=optimizer_reward_mlp_enc_dec,
            train_dataloader=train_dataloader,
            loss_fn=mse_loss_fn,
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            model_name_log_prefix="Reward MLP (Enc-Dec)",
            num_epochs_reward_mlp=enc_dec_mlp_config.get('num_epochs', 1), # Default to 1 epoch if not specified
            log_interval_reward_mlp=enc_dec_mlp_config.get('log_interval', log_interval), # Use specific or general log_interval
            is_jepa_base_model=False,
            wandb_run=wandb_run
        )
    elif enc_dec_mlp_config.get('enabled', False):
        print("Reward MLP (Enc-Dec) training skipped due to missing components (model, optimizer, base_model, or dataloader).")

    if reward_mlp_jepa and jepa_mlp_config.get('enabled', False) and optimizer_reward_mlp_jepa and jepa_model and train_dataloader:
        _train_reward_mlp_epoch(
            reward_mlp_model=reward_mlp_jepa,
            base_model=jepa_model, # This is the best loaded jepa_model
            optimizer_reward_mlp=optimizer_reward_mlp_jepa,
            train_dataloader=train_dataloader,
            loss_fn=mse_loss_fn,
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            model_name_log_prefix="Reward MLP (JEPA)",
            num_epochs_reward_mlp=jepa_mlp_config.get('num_epochs', 1), # Default to 1 epoch
            log_interval_reward_mlp=jepa_mlp_config.get('log_interval', log_interval),
            is_jepa_base_model=True,
            wandb_run=wandb_run
        )
    elif jepa_mlp_config.get('enabled', False):
        print("Reward MLP (JEPA) training skipped due to missing components (model, optimizer, base_model, or dataloader).")

    # --- JEPA State Decoder Training Loop ---
    # The JEPA State Decoder training loop remains as is, but it will use the
    # potentially best JEPA model loaded above.
    # --- JEPA State Decoder Training ---
    final_checkpoint_path_jepa_decoder = None
    jepa_decoder_training_config = config.get('jepa_decoder_training', {})

    if (jepa_decoder and optimizer_jepa_decoder and jepa_model and
        jepa_decoder_training_config.get('enabled', False) and train_dataloader):
        # Note: jepa_model here is the one loaded from its best checkpoint (if saved)
        final_checkpoint_path_jepa_decoder = _train_jepa_state_decoder(
            jepa_decoder_model=jepa_decoder,
            jepa_model=jepa_model,
            optimizer_jepa_decoder=optimizer_jepa_decoder,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=mse_loss_fn,
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            decoder_training_config=jepa_decoder_training_config,
            main_model_dir=model_dir, # Pass the main model directory for path construction
            general_log_interval=log_interval, # Pass general log interval as fallback
            wandb_run=wandb_run
        )
    elif jepa_decoder_training_config.get('enabled', False):
        print("JEPA State Decoder training skipped due to missing components (decoder model, its optimizer, main JEPA model, or dataloader).")
    else:
        if not jepa_decoder: print("JEPA State Decoder model not provided.")
        elif not optimizer_jepa_decoder: print("Optimizer for JEPA State Decoder not provided.")
        elif not jepa_model: print("Main JEPA model (for embeddings) not available for JEPA State Decoder training.")
        elif not jepa_decoder_training_config.get('enabled', False): print("JEPA State Decoder training is disabled in config.")
        elif not train_dataloader: print("Train dataloader not available for JEPA State Decoder training.")


    print("\nAll training processes finished from training_engine.")

    final_checkpoint_enc_dec = early_stopping_state_enc_dec.get('checkpoint_path') \
        if std_enc_dec and early_stopping_state_enc_dec.get('checkpoint_path') and os.path.exists(early_stopping_state_enc_dec['checkpoint_path']) \
        else None
    final_checkpoint_jepa = early_stopping_state_jepa.get('checkpoint_path') \
        if jepa_model and early_stopping_state_jepa.get('checkpoint_path') and os.path.exists(early_stopping_state_jepa['checkpoint_path']) \
        else None
    # final_checkpoint_path_jepa_decoder is already assigned from the helper function's return

    return {
        "best_checkpoint_enc_dec": final_checkpoint_enc_dec,
        "best_checkpoint_jepa": final_checkpoint_jepa,
        "best_checkpoint_jepa_decoder": final_checkpoint_path_jepa_decoder
    }