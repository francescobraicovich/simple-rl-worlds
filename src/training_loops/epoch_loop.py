"""Handles the main training and validation loop for a single epoch of a model."""
import torch
import torch.nn.functional as F # For F.one_hot
import os # For os.path.exists in early stopping save/load
# import matplotlib.pyplot as plt # Not used in this specific function directly
# import numpy as np # Not used in this specific function directly
# import time # Not used in this specific function directly

# Note: Loss functions (mse_loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight)
# will be passed in via the 'losses_map' dictionary.
# Models (std_enc_dec, jepa_model, reward_mlp_enc_dec, reward_mlp_jepa, jepa_decoder) via 'models_map'.
# Optimizers (optimizer_std_enc_dec, etc.) via 'optimizers_map'.
# Dataloaders (train_dataloader, val_dataloader) via 'dataloaders_map'.
# Device, action_dim, action_type also passed as arguments.

def train_validate_model_epoch(
    model, optimizer, train_dataloader, val_dataloader,
    loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight,
    use_aux_for_jepa, use_aux_for_enc_dec,  # use_aux_for_enc_dec is now ignored
    device, epoch_num, log_interval, action_dim, action_type,
    early_stopping_state, checkpoint_path, model_name_log_prefix,
    wandb_run,
    update_target_fn=None,
    max_grad_norm: float = 1.0,
    validation_plotter=None  # New parameter for validation plotting
):
    """
    Handles the training and validation process for a single epoch of a given model.

    This function iterates over the training dataloader, computes losses, performs
    backpropagation, and updates optimizer steps. For JEPA models, it uses VICReg
    loss with projector head for all three loss components (invariance, variance, 
    covariance). For non-JEPA models, auxiliary loss is disabled. Metrics are logged
    periodically during training and at the end of the epoch for both training
    and validation phases using Weights & Biases (wandb). Early stopping logic
    is also implemented based on validation loss.

    Args:
        model (torch.nn.Module): The model to be trained and validated.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation set.
        loss_fn (callable): The primary loss function (MSE for fallback in JEPA).
        aux_loss_fn (callable, optional): VICReg loss function with optional projector.
        aux_loss_name (str): Name of the auxiliary loss, used for logging.
        aux_loss_weight (float): Weighting factor for the VICReg loss.
        use_aux_for_jepa (bool): Whether to apply VICReg loss for JEPA models.
        use_aux_for_enc_dec (bool): Ignored - auxiliary loss disabled for non-JEPA models.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        epoch_num (int): The current epoch number.
        log_interval (int): Interval (in batches) for logging training progress.
        action_dim (int): Dimension of the action space.
        action_type (str): Type of action space ('discrete' or 'continuous').
        early_stopping_state (dict): A dictionary tracking early stopping parameters like
                                     'best_val_loss', 'epochs_no_improve', 'early_stop_flag',
                                     'patience', and 'delta'.
        checkpoint_path (str): Path to save the model checkpoint if validation loss improves.
        model_name_log_prefix (str): Prefix for logging metrics in wandb (e.g., "StdEncDec", "JEPA").
        wandb_run (wandb.sdk.wandb_run.Run, optional): The active Weights & Biases run object.
        update_target_fn (callable, optional): Function to update a target network (e.g., for EMA in JEPA).
        max_grad_norm (float, optional): The maximum norm for gradient clipping. Defaults to 1.0.
        validation_plotter (ValidationPlotter, optional): Shared validation plotter for 
                                                          consistent plotting across models.

    Returns:
        tuple: A tuple containing:
            - dict: The updated early_stopping_state.
            - float: The average primary training loss for the epoch.
            - float: The average VICReg training loss for the epoch (0 if not applicable).
    """
    # === Training Phase ===
    model.train()
    if aux_loss_fn and hasattr(aux_loss_fn, 'train'):
        aux_loss_fn.train()
    model.to(device)

    epoch_train_loss_primary, epoch_train_loss_aux = 0, 0
    epoch_train_loss_std, epoch_train_loss_cov = 0, 0
    num_train_batches = len(train_dataloader) if train_dataloader else 0

    if num_train_batches == 0:
        print(f"{model_name_log_prefix} Epoch {epoch_num}: No training data. Skipping.")
        return early_stopping_state, 0, 0

    for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
        s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
        if action_type == 'discrete':
            if a_t.ndim == 1:
                a_t = a_t.unsqueeze(1)
            a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
        else:
            a_t_processed = a_t.float().to(device)

        optimizer.zero_grad()

        current_loss_primary_item, current_loss_aux_item = 0, 0
        diff_metric = 0.0
        # Initialize VICReg loss components for consistent logging
        sim_loss = torch.tensor(0.0, device=device)
        std_loss = torch.tensor(0.0, device=device)
        cov_loss = torch.tensor(0.0, device=device)
        current_loss_aux = torch.tensor(0.0, device=device)

        if model_name_log_prefix == "JEPA":
            pred_emb, target_emb_detached, online_s_t_emb, target_s_t_emb = model(s_t, a_t_processed, s_t_plus_1)
            
            if aux_loss_fn is not None and use_aux_for_jepa:
                # Use VICReg forward method with projector for all three losses
                total_vicreg_loss, sim_loss, std_loss, cov_loss = aux_loss_fn(online_s_t_emb, target_s_t_emb)
                current_loss_aux = total_vicreg_loss
                total_loss = current_loss_aux
                loss_primary = sim_loss  # For JEPA with VICReg, primary loss is the similarity (invariance) loss
            else:
                # Fallback to MSE loss between predicted and target embeddings
                loss_primary = loss_fn(pred_emb, target_emb_detached)
                total_loss = loss_primary

        else:  # Standard Encoder-Decoder or EncDecJEPAStyle - No auxiliary loss
            output = model(s_t, a_t_processed)
            predicted_s_t_plus_1 = output[0] if isinstance(output, tuple) else output
            
            loss_primary = loss_fn(predicted_s_t_plus_1, s_t_plus_1)  # For Encoder-Decoder, primary loss is reconstruction loss
            total_loss = loss_primary

        current_loss_primary_item = loss_primary.item()
        current_loss_aux_item = current_loss_aux.item()
        total_loss_item = total_loss.item()
        
        # Check for numerical instability before backprop
        if not torch.isfinite(total_loss):
            print(f"WARNING: Non-finite loss detected! Primary: {current_loss_primary_item}, Aux: {current_loss_aux_item}")
            print(f"Skipping batch {batch_idx} due to numerical instability")
            continue
            
        # Check for extremely large losses that could cause instability
        if total_loss_item > 1e6:
            print(f"WARNING: Extremely large loss detected: {total_loss_item:.2e}")
            print(f"Primary loss: {current_loss_primary_item:.2e}, Aux loss: {current_loss_aux_item:.2e}")
            print("Scaling loss down by factor of 1000 to prevent explosion")
            total_loss = total_loss / 1000.0
            total_loss_item = total_loss.item()
        
        total_loss.backward()
        
        # Enhanced gradient clipping with NaN/Inf checks
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not torch.isfinite(grad_norm):
            print("WARNING: Non-finite gradient norm detected! Zeroing gradients.")
            optimizer.zero_grad()
            continue
            
        if grad_norm > 10 * max_grad_norm:
            print(f"WARNING: Very large gradient norm: {grad_norm:.2f}, clipped to {max_grad_norm}")
            
        optimizer.step()
        
        if update_target_fn:
            update_target_fn()
            if model_name_log_prefix == "JEPA" and hasattr(model, 'online_encoder'):
                num_params = 0
                for p_online, p_target in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
                    diff_metric += F.mse_loss(p_online, p_target, reduction='sum').item()
                    num_params += p_online.numel()
                if num_params > 0:
                    diff_metric /= num_params

        epoch_train_loss_primary += current_loss_primary_item
        epoch_train_loss_aux += current_loss_aux_item
        epoch_train_loss_std += std_loss.item()
        epoch_train_loss_cov += cov_loss.item()

        if (batch_idx + 1) % log_interval == 0:
            # Periodic stability check
            if (batch_idx + 1) % (log_interval * 5) == 0:  # Check every 5 log intervals
                param_norms = []
                grad_norms = []
                for name, param in model.named_parameters():
                    if param is not None:
                        param_norm = param.data.norm().item()
                        param_norms.append(param_norm)
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm().item()
                            grad_norms.append(grad_norm)
                
                max_param_norm = max(param_norms) if param_norms else 0
                max_grad_norm = max(grad_norms) if grad_norms else 0
                
                if max_param_norm > 100 or max_grad_norm > 100:
                    print(f"WARNING: Large norms detected - Max param: {max_param_norm:.2f}, Max grad: {max_grad_norm:.2f}")
            
            log_data = {
                f"{model_name_log_prefix}/train/Prediction_Loss": current_loss_primary_item,
                # VICReg loss components
                f"{model_name_log_prefix}/train/VICReg_Sim_Loss": sim_loss.item(),
                f"{model_name_log_prefix}/train/VICReg_Std_Loss": std_loss.item(),
                f"{model_name_log_prefix}/train/VICReg_Cov_Loss": cov_loss.item(),
                f"{model_name_log_prefix}/train/VICReg_Total_Loss": current_loss_aux_item,
                f"{model_name_log_prefix}/train/VICReg_Weighted_Loss": current_loss_aux_item * aux_loss_weight,
                f"{model_name_log_prefix}/train/Total_Loss": total_loss_item,
                f"{model_name_log_prefix}/train/Learning_Rate": optimizer.param_groups[0]['lr']
            }

            if model_name_log_prefix == "JEPA" and hasattr(model, 'online_encoder'):
                log_data[f"{model_name_log_prefix}/train/Encoder_Weight_Diff_MSE"] = diff_metric

            if wandb_run:
                current_global_step = (epoch_num - 1) * num_train_batches + batch_idx
                log_data[f"{model_name_log_prefix}/train/step"] = current_global_step
                wandb_run.log(log_data)

    avg_epoch_train_loss_primary = epoch_train_loss_primary / num_train_batches
    avg_epoch_train_loss_aux = epoch_train_loss_aux / num_train_batches if num_train_batches > 0 else 0
    avg_epoch_train_loss_std = epoch_train_loss_std / num_train_batches if num_train_batches > 0 else 0
    avg_epoch_train_loss_cov = epoch_train_loss_cov / num_train_batches if num_train_batches > 0 else 0

    # === Validation and Epoch Summary Phase ===
    if val_dataloader:
        model.eval()
        if aux_loss_fn and hasattr(aux_loss_fn, 'eval'):
            aux_loss_fn.eval()

        # Set random state for consistent validation plotting across models  
        if validation_plotter and model_name_log_prefix == "StdEncDec":
            validation_plotter.set_epoch_random_state(epoch_num - 1)  # epoch_num is 1-indexed
            # Select random batch and samples for plotting
            plot_batch_data, plot_sample_indices = validation_plotter.select_random_batch_and_samples(
                val_dataloader, device
            )
        else:
            plot_batch_data, plot_sample_indices = None, None

        epoch_val_loss_primary, epoch_val_loss_aux = 0, 0
        epoch_val_loss_std, epoch_val_loss_cov = 0, 0
        num_val_batches = len(val_dataloader)
        
        # Store predictions for plotting if needed
        plot_predictions = None

        with torch.no_grad():
            for s_t_val, a_t_val, r_t_val, s_t_plus_1_val in val_dataloader:
                s_t_val, s_t_plus_1_val = s_t_val.to(device), s_t_plus_1_val.to(device)
                if action_type == 'discrete':
                    if a_t_val.ndim == 1:
                        a_t_val = a_t_val.unsqueeze(1)
                    a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_val_processed = a_t_val.float().to(device)

                val_loss_aux_item = 0
                sim_loss_val = torch.tensor(0.0, device=device)
                std_loss_val = torch.tensor(0.0, device=device)
                cov_loss_val = torch.tensor(0.0, device=device)

                if model_name_log_prefix == "JEPA":
                    pred_emb_val, target_emb_detached_val, online_s_t_emb_val, target_s_t_emb_val = model(s_t_val, a_t_val_processed, s_t_plus_1_val)
                    
                    if aux_loss_fn is not None and use_aux_for_jepa:
                        # Use VICReg forward method with projector for all three losses
                        total_vicreg_loss_val, sim_loss_val, std_loss_val, cov_loss_val = aux_loss_fn(online_s_t_emb_val, target_s_t_emb_val)
                        val_loss_aux_item = total_vicreg_loss_val.item()
                        val_loss_primary = sim_loss_val  # For JEPA with VICReg, primary loss is the similarity (invariance) loss
                    else:
                        # Fallback to MSE loss between predicted and target embeddings
                        val_loss_primary = loss_fn(pred_emb_val, target_emb_detached_val)
                        
                else:  # Standard Encoder-Decoder or EncDecJEPAStyle - No auxiliary loss
                    output_val = model(s_t_val, a_t_val_processed)
                    predicted_s_t_plus_1_val = output_val[0] if isinstance(output_val, tuple) else output_val
                    val_loss_primary = loss_fn(predicted_s_t_plus_1_val, s_t_plus_1_val)  # For Encoder-Decoder, primary loss is reconstruction loss
                    
                    # Check if this is the batch selected for plotting
                    if (plot_batch_data is not None and 
                        torch.equal(s_t_val, plot_batch_data[0]) and
                        torch.equal(s_t_plus_1_val, plot_batch_data[3])):
                        # Store predictions for the selected samples
                        plot_predictions = predicted_s_t_plus_1_val[plot_sample_indices]

                epoch_val_loss_primary += val_loss_primary.item()
                epoch_val_loss_aux += val_loss_aux_item
                epoch_val_loss_std += std_loss_val.item()
                epoch_val_loss_cov += cov_loss_val.item()

        # Handle plotting after validation loop for encoder-decoder
        if (validation_plotter and model_name_log_prefix == "StdEncDec" and 
            plot_batch_data is not None and plot_predictions is not None):
            validation_plotter.plot_validation_samples(
                batch_data=plot_batch_data,
                selected_indices=plot_sample_indices,
                predictions=plot_predictions,
                epoch=epoch_num,
                model_name="Encoder-Decoder"
            )

        avg_val_loss_primary = epoch_val_loss_primary / num_val_batches if num_val_batches > 0 else float('inf')
        avg_val_loss_aux_raw = epoch_val_loss_aux / num_val_batches if num_val_batches > 0 else 0
        avg_val_std_loss = epoch_val_loss_std / num_val_batches if num_val_batches > 0 else 0
        avg_val_cov_loss = epoch_val_loss_cov / num_val_batches if num_val_batches > 0 else 0

        # Consolidated Epoch Logging
        log_epoch_summary = {}
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Prediction_Loss"] = avg_epoch_train_loss_primary
        log_epoch_summary[f"{model_name_log_prefix}/val/Prediction_Loss"] = avg_val_loss_primary
        
        # VICReg loss components logging
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/VICReg_Sim_Loss"] = epoch_train_loss_aux / num_train_batches if num_train_batches > 0 else 0
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/VICReg_Std_Loss"] = avg_epoch_train_loss_std
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/VICReg_Cov_Loss"] = avg_epoch_train_loss_cov
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/VICReg_Total_Loss"] = avg_epoch_train_loss_aux
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/VICReg_Weighted_Loss"] = avg_epoch_train_loss_aux * aux_loss_weight
        
        log_epoch_summary[f"{model_name_log_prefix}/val/VICReg_Sim_Loss"] = avg_val_loss_aux_raw  # This contains total VICReg loss in validation
        log_epoch_summary[f"{model_name_log_prefix}/val/VICReg_Std_Loss"] = avg_val_std_loss
        log_epoch_summary[f"{model_name_log_prefix}/val/VICReg_Cov_Loss"] = avg_val_cov_loss
        log_epoch_summary[f"{model_name_log_prefix}/val/VICReg_Total_Loss"] = avg_val_loss_aux_raw
        log_epoch_summary[f"{model_name_log_prefix}/val/VICReg_Weighted_Loss"] = avg_val_loss_aux_raw * aux_loss_weight

        # Calculate total losses for reporting and early stopping
        avg_total_train_loss = avg_epoch_train_loss_primary
        current_total_val_loss = avg_val_loss_primary
        
        # For JEPA models using VICReg, the aux loss IS the primary loss
        if model_name_log_prefix == "JEPA" and aux_loss_fn and use_aux_for_jepa:
            avg_total_train_loss = avg_epoch_train_loss_aux * aux_loss_weight
            current_total_val_loss = avg_val_loss_aux_raw * aux_loss_weight

        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Total_Loss"] = avg_total_train_loss
        log_epoch_summary[f"{model_name_log_prefix}/val/Total_Loss"] = current_total_val_loss
        
        print(f"  {'Avg Train Loss':<22}: {avg_epoch_train_loss_primary:>8.4f} | {'Avg Val Loss':<22}: {avg_val_loss_primary:>8.4f}")
        if model_name_log_prefix == "JEPA" and aux_loss_fn is not None:
            print(f"  {'Avg Train VICReg Loss':<22}: {avg_epoch_train_loss_aux:>8.4f} | {'Avg Val VICReg Loss':<22}: {avg_val_loss_aux_raw:>8.4f}")
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
                print("  Early stopping triggered.")

    return early_stopping_state, avg_epoch_train_loss_primary, avg_epoch_train_loss_aux