"""Handles the main training and validation loop for a single epoch of a model."""
import torch
import torch.nn.functional as F # For F.one_hot
import os # For os.path.exists in early stopping save/load
# import matplotlib.pyplot as plt # Not used in this specific function directly
# import numpy as np # Not used in this specific function directly
# import time # Not used in this specific function directly
import wandb # For wandb.Image

# Note: Loss functions (mse_loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight)
# will be passed in via the 'losses_map' dictionary.
# Models (std_enc_dec, jepa_model, reward_mlp_enc_dec, reward_mlp_jepa, jepa_decoder) via 'models_map'.
# Optimizers (optimizer_std_enc_dec, etc.) via 'optimizers_map'.
# Dataloaders (train_dataloader, val_dataloader) via 'dataloaders_map'.
# Device, action_dim, action_type also passed as arguments.

def train_validate_model_epoch(
    model, optimizer, train_dataloader, val_dataloader,
    loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight,
    use_aux_for_jepa, use_aux_for_enc_dec,  # New parameters
    device, epoch_num, log_interval, action_dim, action_type,
    early_stopping_state, checkpoint_path, model_name_log_prefix,
    wandb_run,
    update_target_fn=None
):
    """
    Handles the training and validation process for a single epoch of a given model.

    This function iterates over the training dataloader, computes losses, performs
    backpropagation, and updates optimizer steps. It then iterates over the
    validation dataloader to compute validation losses. Metrics are logged
    periodically during training and at the end of the epoch for both training
    and validation phases using Weights & Biases (wandb). Early stopping logic
    is also implemented based on validation loss.

    Args:
        model (torch.nn.Module): The model to be trained and validated.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation set.
        loss_fn (callable): The primary loss function.
        aux_loss_fn (callable, optional): An auxiliary loss function (e.g., for regularization).
        aux_loss_name (str): Name of the auxiliary loss, used for logging.
        aux_loss_weight (float): Weighting factor for the auxiliary loss.
        use_aux_for_jepa (bool): Whether to apply auxiliary loss for JEPA models.
        use_aux_for_enc_dec (bool): Whether to apply auxiliary loss for Encoder-Decoder models.
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

    Returns:
        tuple: A tuple containing:
            - dict: The updated early_stopping_state.
            - float: The average primary training loss for the epoch.
            - float: The average auxiliary training loss for the epoch (0 if not applicable).
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
            if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
            a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
        else:
            a_t_processed = a_t.float().to(device)

        optimizer.zero_grad()

        current_loss_primary_item, current_loss_aux_item, total_loss_item = 0, 0, 0
        diff_metric = 0.0
        # Initialize std and cov loss variables for consistent logging
        std_loss_s_t = torch.tensor(0.0, device=device)
        cov_loss_s_t = torch.tensor(0.0, device=device)

        if model_name_log_prefix == "JEPA":
            # Dummy logic for JEPA model forward pass
            # JEPA model forward pass and primary loss
            pred_emb, target_emb_detached, online_s_t_emb, _ = model(s_t, a_t_processed, s_t_plus_1)
            loss_primary = loss_fn(pred_emb, target_emb_detached)
            current_loss_primary_item = loss_primary.item()
            
            current_loss_aux = torch.tensor(0.0, device=device)
            # Always calculate auxiliary loss components for JEPA if aux_loss_fn exists, regardless of use_aux_for_jepa
            if aux_loss_fn is not None:
                aux_term_s_t, std_loss_s_t, cov_loss_s_t = aux_loss_fn.calculate_reg_terms(online_s_t_emb)
                current_loss_aux = aux_term_s_t
                current_loss_aux_item = current_loss_aux.item()
            
            # Only apply auxiliary loss to total loss if configured for JEPA
            if use_aux_for_jepa:
                total_loss = loss_primary + current_loss_aux * aux_loss_weight
            else:
                total_loss = loss_primary

        else:  # Standard Encoder-Decoder or EncDecJEPAStyle
            # For Encoder-Decoder, the model's forward pass might just return the prediction.
            # We need the encoder's representation of s_t for the auxiliary loss.
            # Assuming model has an 'encoder' attribute and it can process s_t.
            # If model's forward needs to be changed to return embeddings, that's a separate step.
            
            s_t_emb = None
            if hasattr(model, 'encoder') and callable(model.encoder):
                 s_t_emb = model.encoder(s_t)
            elif hasattr(model, 'encode') and callable(model.encode): # Alternative common pattern
                 s_t_emb = model.encode(s_t)
            # If EncDecJEPAStyle model, it might already return embeddings.
            # Let's assume 'model(s_t, a_t_processed)' returns (prediction, s_t_embedding)
            # or we can get s_t_embedding from model.encoder(s_t)
            
            # Standard forward pass for prediction
            output = model(s_t, a_t_processed)

            # Adapt based on what the model returns.
            # Common patterns:
            # 1. model returns only prediction: predicted_s_t_plus_1 = output
            # 2. model returns (prediction, embedding): predicted_s_t_plus_1, s_t_emb_from_fwd = output
            # For now, assume model.encoder(s_t) is the way for aux loss, and output is prediction.
            
            predicted_s_t_plus_1 = output
            if isinstance(output, tuple): # Handle cases like EncDecJEPAStyle that might return more
                predicted_s_t_plus_1 = output[0] # Assuming prediction is the first element
                if s_t_emb is None and len(output) > 1 and torch.is_tensor(output[1]): # Check if second output could be s_t_emb
                    # This is a guess; ideally, the model's API is clear.
                    # For EncDecJEPAStyle, s_t_emb might be output[2] if it returns (pred, context_emb, s_t_emb_for_predictor)
                    # For simplicity, we rely on model.encoder or model.encode first.
                    # If specific models (like EncDecJEPAStyle) expose embeddings differently,
                    # this part might need refinement or specific handling for that model_name_log_prefix.
                    pass


            loss_primary = loss_fn(predicted_s_t_plus_1, s_t_plus_1)
            current_loss_primary_item = loss_primary.item()
            total_loss = loss_primary
            
            current_loss_aux = torch.tensor(0.0, device=device)
            # Initialize std and cov loss variables for consistent logging  
            std_loss_s_t = torch.tensor(0.0, device=device)
            cov_loss_s_t = torch.tensor(0.0, device=device)
            
            # Always calculate auxiliary loss components for Encoder-Decoder if aux_loss_fn exists, regardless of use_aux_for_enc_dec
            if aux_loss_fn is not None:
                if s_t_emb is not None:
                    aux_term_s_t, std_loss_s_t, cov_loss_s_t = aux_loss_fn.calculate_reg_terms(s_t_emb)
                    current_loss_aux = aux_term_s_t
                    current_loss_aux_item = current_loss_aux.item()
                    # Only apply auxiliary loss to total loss if configured for Encoder-Decoder
                    if use_aux_for_enc_dec:
                        total_loss = total_loss + current_loss_aux * aux_loss_weight
                else:
                    if batch_idx == 0 and epoch_num == 1: # Log warning only once per run
                         print(f"Warning: Could not get s_t_emb for {model_name_log_prefix} to apply auxiliary loss. Model might need 'encoder' attribute or 'encode' method.")


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
        epoch_train_loss_std += std_loss_s_t.item() if std_loss_s_t is not None else 0
        epoch_train_loss_cov += cov_loss_s_t.item() if cov_loss_s_t is not None else 0

        if (batch_idx + 1) % log_interval == 0:
            log_data = {
                f"{model_name_log_prefix}/train/Prediction_Loss": current_loss_primary_item,
                f"{model_name_log_prefix}/train/Total_Loss": total_loss_item,
                f"{model_name_log_prefix}/train/Learning_Rate": optimizer.param_groups[0]['lr']
            }
            
            # Always log auxiliary loss components if aux_loss_fn exists, regardless of weight or usage
            if aux_loss_fn is not None:
                log_data[f"{model_name_log_prefix}/train/Aux_Raw_Loss"] = current_loss_aux_item
                log_data[f"{model_name_log_prefix}/train/Aux_Std_Loss"] = std_loss_s_t.item() if std_loss_s_t is not None else 0
                log_data[f"{model_name_log_prefix}/train/Aux_Cov_Loss"] = cov_loss_s_t.item() if cov_loss_s_t is not None else 0
                log_data[f"{model_name_log_prefix}/train/Aux_Weighted_Loss"] = current_loss_aux_item * aux_loss_weight

            if model_name_log_prefix == "JEPA" and hasattr(model, 'online_encoder'):
                log_data[f"{model_name_log_prefix}/train/Encoder_Weight_Diff_MSE"] = diff_metric

            if wandb_run:
                current_global_step = (epoch_num - 1) * num_train_batches + batch_idx
                log_data[f"{model_name_log_prefix}/train/step"] = current_global_step
                wandb_run.log(log_data)

    avg_epoch_train_loss_primary = epoch_train_loss_primary / num_train_batches
    # Calculate avg_epoch_train_loss_aux always if aux_loss_fn exists
    avg_epoch_train_loss_aux = epoch_train_loss_aux / num_train_batches if aux_loss_fn and num_train_batches > 0 else 0

    # Calculate training averages for std and cov losses
    avg_epoch_train_std_loss = epoch_train_loss_std / num_train_batches if num_train_batches > 0 else 0
    avg_epoch_train_cov_loss = epoch_train_loss_cov / num_train_batches if num_train_batches > 0 else 0


    # === Validation and Epoch Summary Phase ===
    if val_dataloader:
        model.eval()
        if aux_loss_fn and hasattr(aux_loss_fn, 'eval'):
            aux_loss_fn.eval()

        epoch_val_loss_primary, epoch_val_loss_aux = 0, 0
        epoch_std_loss_s_t, epoch_cov_loss_s_t = 0, 0
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
                # Initialize std and cov loss variables for consistent logging
                std_loss_s_t_val = torch.tensor(0.0, device=device)
                cov_loss_s_t_val = torch.tensor(0.0, device=device)

                if model_name_log_prefix == "JEPA":
                    pred_emb_val, target_emb_detached_val, online_s_t_emb_val, _ = model(s_t_val, a_t_val_processed, s_t_plus_1_val)
                    val_loss_primary = loss_fn(pred_emb_val, target_emb_detached_val)
                    val_loss_primary_item = val_loss_primary.item()
                    # Always calculate auxiliary loss components for JEPA if aux_loss_fn exists
                    if aux_loss_fn is not None:
                        aux_term_s_t_val, std_loss_s_t_val, cov_loss_s_t_val = aux_loss_fn.calculate_reg_terms(online_s_t_emb_val)
                        val_loss_aux_item = aux_term_s_t_val.item()
                else: # Standard Encoder-Decoder or EncDecJEPAStyle
                    s_t_emb_val = None
                    if hasattr(model, 'encoder') and callable(model.encoder):
                        s_t_emb_val = model.encoder(s_t_val)
                    elif hasattr(model, 'encode') and callable(model.encode):
                        s_t_emb_val = model.encode(s_t_val)
                    
                    output_val = model(s_t_val, a_t_val_processed)
                    predicted_s_t_plus_1_val = output_val
                    if isinstance(output_val, tuple):
                        predicted_s_t_plus_1_val = output_val[0]
                        # s_t_emb_val could potentially be output_val[1] or output_val[2] here too
                        # but we prioritize explicit .encoder() or .encode() for aux loss input

                    val_loss_primary = loss_fn(predicted_s_t_plus_1_val, s_t_plus_1_val)
                    val_loss_primary_item = val_loss_primary.item()

                    # Always calculate auxiliary loss components for Encoder-Decoder if aux_loss_fn exists
                    if aux_loss_fn is not None:
                        if s_t_emb_val is not None:
                            aux_term_s_t_val, std_loss_s_t_val, cov_loss_s_t_val = aux_loss_fn.calculate_reg_terms(s_t_emb_val)
                            val_loss_aux_item = aux_term_s_t_val.item()
                        # else: warning already printed in training loop if s_t_emb is not found

                epoch_val_loss_primary += val_loss_primary_item
                epoch_val_loss_aux += val_loss_aux_item
                epoch_std_loss_s_t += std_loss_s_t_val.item() if std_loss_s_t_val is not None else 0
                epoch_cov_loss_s_t += cov_loss_s_t_val.item() if cov_loss_s_t_val is not None else 0

        avg_val_loss_primary = epoch_val_loss_primary / num_val_batches if num_val_batches > 0 else float('inf')
        avg_val_std_loss_s_t = epoch_std_loss_s_t / num_val_batches if num_val_batches > 0 else 0
        avg_val_cov_loss_s_t = epoch_cov_loss_s_t / num_val_batches if num_val_batches > 0 else 0

        # Calculate avg_val_loss_aux_raw always if aux_loss_fn exists
        avg_val_loss_aux_raw = epoch_val_loss_aux / num_val_batches if aux_loss_fn and num_val_batches > 0 else 0


        # Consolidated Epoch Logging
        log_epoch_summary = {}
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Prediction_Loss"] = avg_epoch_train_loss_primary
        log_epoch_summary[f"{model_name_log_prefix}/val/Prediction_Loss"] = avg_val_loss_primary
        
        # Always log auxiliary loss components if aux_loss_fn exists, regardless of weight or usage
        if aux_loss_fn is not None:
            log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Aux_Raw_Loss"] = avg_epoch_train_loss_aux
            log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Aux_Std_Loss"] = avg_epoch_train_std_loss
            log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Aux_Cov_Loss"] = avg_epoch_train_cov_loss
            log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Aux_Weighted_Loss"] = avg_epoch_train_loss_aux * aux_loss_weight
            
            log_epoch_summary[f"{model_name_log_prefix}/val/Aux_Raw_Loss"] = avg_val_loss_aux_raw
            log_epoch_summary[f"{model_name_log_prefix}/val/Aux_Std_Loss"] = avg_val_std_loss_s_t
            log_epoch_summary[f"{model_name_log_prefix}/val/Aux_Cov_Loss"] = avg_val_cov_loss_s_t
            log_epoch_summary[f"{model_name_log_prefix}/val/Aux_Weighted_Loss"] = avg_val_loss_aux_raw * aux_loss_weight

        # Calculate total losses
        avg_total_train_loss = avg_epoch_train_loss_primary
        current_total_val_loss = avg_val_loss_primary
        
        # Add auxiliary loss to total only if configured for this model type
        is_aux_applied = aux_loss_fn and \
                        ((model_name_log_prefix == "JEPA" and use_aux_for_jepa) or \
                         (model_name_log_prefix != "JEPA" and use_aux_for_enc_dec))
        
        if is_aux_applied:
            avg_total_train_loss += avg_epoch_train_loss_aux * aux_loss_weight
            current_total_val_loss += avg_val_loss_aux_raw * aux_loss_weight

        # Always log total losses for consistent reporting structure
        log_epoch_summary[f"{model_name_log_prefix}/train_epoch_avg/Total_Loss"] = avg_total_train_loss
        log_epoch_summary[f"{model_name_log_prefix}/val/Total_Loss"] = current_total_val_loss
        
        print(f"  {'Avg Train Loss':<22}: {avg_epoch_train_loss_primary:>8.4f} | {'Avg Val Loss':<22}: {avg_val_loss_primary:>8.4f}")
        if aux_loss_fn is not None: # Print aux losses if aux_loss_fn exists
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
                print("  Early stopping triggered.")

    return early_stopping_state, avg_epoch_train_loss_primary, avg_epoch_train_loss_aux
