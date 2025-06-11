# Contents for src/training_engine.py
import torch
import torch.nn.functional as F # For F.one_hot
import os # For os.path.exists in early stopping save/load

# Note: Loss functions (mse_loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight)
# will be passed in via the 'losses_map' dictionary.
# Models (std_enc_dec, jepa_model, reward_mlp_enc_dec, reward_mlp_jepa) via 'models_map'.
# Optimizers (optimizer_std_enc_dec, etc.) via 'optimizers_map'.
# Dataloaders (train_dataloader, val_dataloader) via 'dataloaders_map'.
# Configs (early_stopping_config, enc_dec_mlp_config, jepa_mlp_config, main_config for num_epochs, log_interval) via 'config'.
# Device, action_dim, action_type also passed as arguments.

def run_training_epochs(
    models_map, optimizers_map, losses_map, dataloaders_map,
    device, config, action_dim, action_type,
    image_h_w, input_channels # For reward MLP input calculation if needed
):
    # Unpack from maps/config for convenience, matching original variable names closely
    std_enc_dec = models_map.get('std_enc_dec')
    jepa_model = models_map.get('jepa')
    reward_mlp_enc_dec = models_map.get('reward_mlp_enc_dec')
    reward_mlp_jepa = models_map.get('reward_mlp_jepa')

    optimizer_std_enc_dec = optimizers_map.get('std_enc_dec')
    optimizer_jepa = optimizers_map.get('jepa')
    optimizer_reward_mlp_enc_dec = optimizers_map.get('reward_mlp_enc_dec')
    optimizer_reward_mlp_jepa = optimizers_map.get('reward_mlp_jepa')

    mse_loss_fn = losses_map['mse']
    aux_loss_fn = losses_map.get('aux_fn')
    aux_loss_name = losses_map.get('aux_name', "None")
    aux_loss_weight = losses_map.get('aux_weight', 0.0)

    train_dataloader = dataloaders_map['train']
    val_dataloader = dataloaders_map.get('val') # val_dataloader can be None

    # Configs
    early_stopping_config = config.get('early_stopping', {})
    model_dir = config.get('model_dir', 'trained_models/') # Read model_dir
    patience = early_stopping_config.get('patience', 10)
    delta = early_stopping_config.get('delta', 0.001)
    # Prepend model_dir to checkpoint paths
    checkpoint_path_enc_dec = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_enc_dec', 'best_encoder_decoder.pth'))
    checkpoint_path_jepa = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_jepa', 'best_jepa.pth'))
    # validation_split is used to decide if validation runs, passed implicitly by val_dataloader's existence

    enc_dec_mlp_config = config.get('reward_predictors', {}).get('encoder_decoder_reward_mlp', {})
    jepa_mlp_config = config.get('reward_predictors', {}).get('jepa_reward_mlp', {})

    num_epochs = config.get('num_epochs', 10)
    log_interval = config.get('log_interval', 50)

    print(f"Starting training for {num_epochs} epochs...")

    # Early Stopping Trackers
    best_val_loss_enc_dec = float('inf')
    epochs_no_improve_enc_dec = 0
    best_val_loss_jepa = float('inf')
    epochs_no_improve_jepa = 0
    early_stop_enc_dec = False if std_enc_dec and optimizer_std_enc_dec else True # If model or optimizer is None, it's "stopped"
    early_stop_jepa = False if jepa_model and optimizer_jepa else True # If model or optimizer is None, it's "stopped"


    for epoch in range(num_epochs):
        epoch_loss_std = 0
        epoch_loss_jepa_pred = 0
        epoch_loss_jepa_aux = 0 # Raw aux loss
        epoch_loss_reward_enc_dec = 0
        epoch_loss_reward_jepa = 0

        num_train_batches = len(train_dataloader) if train_dataloader else 0
        if num_train_batches == 0:
            print(f"Epoch {epoch+1} has no training data. Skipping.")
            continue

        # Main model training loop
        if std_enc_dec and not early_stop_enc_dec: std_enc_dec.train()
        if jepa_model and not early_stop_jepa: jepa_model.train()
        if aux_loss_fn and hasattr(aux_loss_fn, 'train'): # For DINO Loss
            aux_loss_fn.train()

        # === Training Phase ===
        if not (early_stop_enc_dec and early_stop_jepa): # Only loop if at least one model needs training
            for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
                s_t, r_t, s_t_plus_1 = s_t.to(device), r_t.to(device).float().unsqueeze(1), s_t_plus_1.to(device)

                if action_type == 'discrete':
                    if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
                    a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_processed = a_t.float().to(device)

                # Standard Encoder-Decoder Training
                loss_std_item = 0 # for logging
                if std_enc_dec and not early_stop_enc_dec:
                    optimizer_std_enc_dec.zero_grad()
                    predicted_s_t_plus_1 = std_enc_dec(s_t, a_t_processed)
                    loss_std = mse_loss_fn(predicted_s_t_plus_1, s_t_plus_1)
                    loss_std.backward()
                    optimizer_std_enc_dec.step()
                    loss_std_item = loss_std.item()
                    epoch_loss_std += loss_std_item

                # JEPA Training
                current_loss_jepa_pred_item = 0 # for logging
                current_loss_jepa_aux_item = 0 # for logging
                current_total_loss_jepa_item = 0 # for logging
                if jepa_model and not early_stop_jepa:
                    optimizer_jepa.zero_grad()
                    pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = jepa_model(s_t, a_t_processed, s_t_plus_1)
                    loss_jepa_pred = mse_loss_fn(pred_emb, target_emb_detached)
                    current_loss_jepa_pred_item = loss_jepa_pred.item()

                    current_loss_jepa_aux = torch.tensor(0.0, device=device)
                    if aux_loss_fn is not None and aux_loss_weight > 0:
                        aux_term_s_t, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_emb)
                        aux_term_s_t_plus_1, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_plus_1_emb)
                        current_loss_jepa_aux = (aux_term_s_t + aux_term_s_t_plus_1) * 0.5

                    current_loss_jepa_aux_item = current_loss_jepa_aux.item() # raw aux loss
                    total_loss_jepa = loss_jepa_pred + current_loss_jepa_aux * aux_loss_weight
                    current_total_loss_jepa_item = total_loss_jepa.item()
                    total_loss_jepa.backward()
                    optimizer_jepa.step()
                    jepa_model.update_target_network() # EMA update for target encoder
                    epoch_loss_jepa_pred += current_loss_jepa_pred_item
                    epoch_loss_jepa_aux += current_loss_jepa_aux_item


                if (batch_idx + 1) % log_interval == 0:
                    log_msg = f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_train_batches}:"
                    if std_enc_dec and not early_stop_enc_dec:
                        log_msg += f" StdEncDec L: {loss_std_item:.4f} |"
                    if jepa_model and not early_stop_jepa:
                        weighted_aux_loss_str = f"{(current_loss_jepa_aux_item * aux_loss_weight):.4f}" if aux_loss_fn and aux_loss_weight > 0 else "N/A"
                        log_msg += (f" JEPA Pred L: {current_loss_jepa_pred_item:.4f},"
                                    f" {aux_loss_name} AuxRawL: {current_loss_jepa_aux_item:.4f} (W: {weighted_aux_loss_str}),"
                                    f" Total JEPA L: {current_total_loss_jepa_item:.4f}")
                    if log_msg[-1] != ':': print(log_msg) # Avoid printing if no models trained in batch

        # === Validation Phase ===
        # Run validation if val_dataloader exists and at least one model is not early_stopped
        if val_dataloader and ( (std_enc_dec and not early_stop_enc_dec) or (jepa_model and not early_stop_jepa) ):
            if std_enc_dec: std_enc_dec.eval()
            if jepa_model: jepa_model.eval()
            if aux_loss_fn and hasattr(aux_loss_fn, 'eval'): # For DINO Loss
                aux_loss_fn.eval()

            epoch_val_loss_std = 0
            epoch_val_loss_jepa_pred = 0
            epoch_val_loss_jepa_aux = 0 # Raw

            with torch.no_grad():
                for s_t_val, a_t_val, r_t_val, s_t_plus_1_val in val_dataloader:
                    s_t_val, s_t_plus_1_val = s_t_val.to(device), s_t_plus_1_val.to(device)
                    if action_type == 'discrete':
                        if a_t_val.ndim == 1: a_t_val = a_t_val.unsqueeze(1)
                        a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                    else:
                        a_t_val_processed = a_t_val.float().to(device)

                    if std_enc_dec and not early_stop_enc_dec:
                        predicted_s_t_plus_1_val = std_enc_dec(s_t_val, a_t_val_processed)
                        val_loss_std = mse_loss_fn(predicted_s_t_plus_1_val, s_t_plus_1_val)
                        epoch_val_loss_std += val_loss_std.item()

                    if jepa_model and not early_stop_jepa:
                        pred_emb_val, target_emb_detached_val, online_s_t_emb_val, online_s_t_plus_1_emb_val = jepa_model(s_t_val, a_t_val_processed, s_t_plus_1_val)
                        val_loss_jepa_pred = mse_loss_fn(pred_emb_val, target_emb_detached_val)

                        current_val_loss_jepa_aux = torch.tensor(0.0, device=device)
                        if aux_loss_fn is not None and aux_loss_weight > 0:
                            aux_term_s_t_val, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_emb_val)
                            aux_term_s_t_plus_1_val, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_plus_1_emb_val)
                            current_val_loss_jepa_aux = (aux_term_s_t_val + aux_term_s_t_plus_1_val) * 0.5
                        epoch_val_loss_jepa_pred += val_loss_jepa_pred.item()
                        epoch_val_loss_jepa_aux += current_val_loss_jepa_aux.item() # raw aux loss

            num_val_batches = len(val_dataloader)
            avg_val_loss_std = epoch_val_loss_std / num_val_batches if num_val_batches > 0 and std_enc_dec and not early_stop_enc_dec else float('inf')
            avg_val_loss_jepa_pred = epoch_val_loss_jepa_pred / num_val_batches if num_val_batches > 0 and jepa_model and not early_stop_jepa else float('inf')
            avg_val_loss_jepa_aux_raw = epoch_val_loss_jepa_aux / num_val_batches if num_val_batches > 0 and jepa_model and not early_stop_jepa else float('inf')
            avg_total_val_loss_jepa = avg_val_loss_jepa_pred + avg_val_loss_jepa_aux_raw * aux_loss_weight if jepa_model and not early_stop_jepa else float('inf')

            print(f"--- Epoch {epoch+1} Validation Summary ---")
            if std_enc_dec and not early_stop_enc_dec:
                print(f"  Avg Val StdEncDec L: {avg_val_loss_std:.4f}")
            if jepa_model and not early_stop_jepa:
                print(f"  Avg Val JEPA Pred L: {avg_val_loss_jepa_pred:.4f}, {aux_loss_name} AuxRawL: {avg_val_loss_jepa_aux_raw:.4f}, Total Val JEPA L: {avg_total_val_loss_jepa:.4f}")

            if std_enc_dec and not early_stop_enc_dec:
                if avg_val_loss_std < best_val_loss_enc_dec - delta:
                    best_val_loss_enc_dec = avg_val_loss_std
                    if checkpoint_path_enc_dec:
                        os.makedirs(model_dir, exist_ok=True) # Ensure model_dir exists
                        torch.save(std_enc_dec.state_dict(), checkpoint_path_enc_dec)
                    epochs_no_improve_enc_dec = 0
                    print(f"  Encoder/Decoder: Val loss improved. Saved model to {checkpoint_path_enc_dec}")
                else:
                    epochs_no_improve_enc_dec += 1
                    print(f"  Encoder/Decoder: No val improvement for {epochs_no_improve_enc_dec} epochs.")
                    if epochs_no_improve_enc_dec >= patience:
                        early_stop_enc_dec = True
                        print("  Encoder/Decoder: Early stopping triggered.")

            if jepa_model and not early_stop_jepa:
                if avg_total_val_loss_jepa < best_val_loss_jepa - delta:
                    best_val_loss_jepa = avg_total_val_loss_jepa
                    if checkpoint_path_jepa:
                        os.makedirs(model_dir, exist_ok=True) # Ensure model_dir exists
                        torch.save(jepa_model.state_dict(), checkpoint_path_jepa)
                    epochs_no_improve_jepa = 0
                    print(f"  JEPA: Val loss improved. Saved model to {checkpoint_path_jepa}")
                else:
                    epochs_no_improve_jepa += 1
                    print(f"  JEPA: No val improvement for {epochs_no_improve_jepa} epochs.")
                    if epochs_no_improve_jepa >= patience:
                        early_stop_jepa = True
                        print("  JEPA: Early stopping triggered.")

        # --- Reward MLP Training Loop ---
        num_batches_reward_train = 0 # Reset per epoch
        # Check if any reward MLP needs training
        reward_enc_dec_train_needed = reward_mlp_enc_dec and enc_dec_mlp_config.get('enabled', False) and not early_stop_enc_dec and optimizer_reward_mlp_enc_dec and std_enc_dec
        reward_jepa_train_needed = reward_mlp_jepa and jepa_mlp_config.get('enabled', False) and not early_stop_jepa and optimizer_reward_mlp_jepa and jepa_model

        if train_dataloader and (reward_enc_dec_train_needed or reward_jepa_train_needed):
            print(f"Epoch {epoch+1} - Starting Reward MLP Training...")
            if std_enc_dec: std_enc_dec.eval() # Main models providing input to MLPs should be in eval
            if jepa_model: jepa_model.eval()
            if reward_mlp_enc_dec and reward_enc_dec_train_needed: reward_mlp_enc_dec.train()
            if reward_mlp_jepa and reward_jepa_train_needed: reward_mlp_jepa.train()

            for reward_batch_idx, (s_t_reward, a_t_reward, r_t_reward, s_t_plus_1_reward) in enumerate(train_dataloader):
                s_t_reward, r_t_reward, s_t_plus_1_reward = s_t_reward.to(device), r_t_reward.to(device).float().unsqueeze(1), s_t_plus_1_reward.to(device)
                if action_type == 'discrete':
                    if a_t_reward.ndim == 1: a_t_reward = a_t_reward.unsqueeze(1)
                    a_t_reward_processed = F.one_hot(a_t_reward.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_reward_processed = a_t_reward.float().to(device)

                num_batches_reward_train +=1

                if reward_enc_dec_train_needed:
                    optimizer_reward_mlp_enc_dec.zero_grad()
                    with torch.no_grad():
                        # Input for this MLP: predicted next state from std_enc_dec
                        predicted_s_t_plus_1_for_reward = std_enc_dec(s_t_reward, a_t_reward_processed).detach()

                    if enc_dec_mlp_config.get('input_type') == "flatten": # As per original config
                        input_enc_dec_reward_mlp = predicted_s_t_plus_1_for_reward.view(predicted_s_t_plus_1_for_reward.size(0), -1)
                    else: # Fallback, or could be error
                        input_enc_dec_reward_mlp = predicted_s_t_plus_1_for_reward.view(predicted_s_t_plus_1_for_reward.size(0), -1)

                    pred_reward_enc_dec = reward_mlp_enc_dec(input_enc_dec_reward_mlp)
                    loss_reward_enc_dec = mse_loss_fn(pred_reward_enc_dec, r_t_reward)
                    loss_reward_enc_dec.backward()
                    optimizer_reward_mlp_enc_dec.step()
                    epoch_loss_reward_enc_dec += loss_reward_enc_dec.item()
                    if (reward_batch_idx + 1) % enc_dec_mlp_config.get('log_interval', log_interval) == 0:
                        print(f"  Epoch {epoch+1}, Reward MLP (Enc-Dec) Batch {reward_batch_idx+1}/{num_train_batches}: Loss {loss_reward_enc_dec.item():.4f}")

                if reward_jepa_train_needed:
                    optimizer_reward_mlp_jepa.zero_grad()
                    with torch.no_grad():
                        # Input for this MLP: JEPA's predicted embedding of next state
                        pred_emb_for_reward, _, _, _ = jepa_model(s_t_reward, a_t_reward_processed, s_t_plus_1_reward)
                        input_jepa_reward_mlp = pred_emb_for_reward.detach()

                    pred_reward_jepa = reward_mlp_jepa(input_jepa_reward_mlp)
                    loss_reward_jepa = mse_loss_fn(pred_reward_jepa, r_t_reward)
                    loss_reward_jepa.backward()
                    optimizer_reward_mlp_jepa.step()
                    epoch_loss_reward_jepa += loss_reward_jepa.item()
                    if (reward_batch_idx + 1) % jepa_mlp_config.get('log_interval', log_interval) == 0:
                         print(f"  Epoch {epoch+1}, Reward MLP (JEPA) Batch {reward_batch_idx+1}/{num_train_batches}: Loss {loss_reward_jepa.item():.4f}")

        # --- Epoch Summary ---
        avg_loss_std = epoch_loss_std / num_train_batches if std_enc_dec and not early_stop_enc_dec and num_train_batches > 0 else 0
        avg_loss_jepa_pred = epoch_loss_jepa_pred / num_train_batches if jepa_model and not early_stop_jepa and num_train_batches > 0 else 0
        avg_loss_jepa_aux_raw = epoch_loss_jepa_aux / num_train_batches if jepa_model and not early_stop_jepa and num_train_batches > 0 else 0

        avg_loss_reward_enc_dec = epoch_loss_reward_enc_dec / num_batches_reward_train if num_batches_reward_train > 0 and reward_enc_dec_train_needed else 0
        avg_loss_reward_jepa = epoch_loss_reward_jepa / num_batches_reward_train if num_batches_reward_train > 0 and reward_jepa_train_needed else 0

        print(f"--- Epoch {epoch+1}/{num_epochs} Training Summary ---")
        if std_enc_dec and not early_stop_enc_dec: print(f"  Avg Train StdEncDec L: {avg_loss_std:.4f}")
        if jepa_model and not early_stop_jepa:
            avg_total_jepa_loss = avg_loss_jepa_pred + avg_loss_jepa_aux_raw * aux_loss_weight
            print(f"  Avg Train JEPA Pred L: {avg_loss_jepa_pred:.4f}, {aux_loss_name} AuxRawL: {avg_loss_jepa_aux_raw:.4f}, Total Train JEPA L: {avg_total_jepa_loss:.4f}")
        if reward_enc_dec_train_needed: # Check if it was supposed to run
            print(f"  Avg Train Reward MLP (Enc-Dec) L: {avg_loss_reward_enc_dec:.4f}")
        if reward_jepa_train_needed: # Check if it was supposed to run
            print(f"  Avg Train Reward MLP (JEPA) L: {avg_loss_reward_jepa:.4f}")

        if early_stop_enc_dec and early_stop_jepa:
            print("Both models triggered early stopping or were not set to train. Ending training.")
            break

    print("Main training loop finished from training_engine.")
    # Return paths to best models for main script to load if needed.
    # Or could return the models themselves if they were modified in place (e.g. loaded best weights)
    # For now, just indicate completion. Main script will handle loading from fixed paths.
    return {
        "best_checkpoint_enc_dec": checkpoint_path_enc_dec if os.path.exists(checkpoint_path_enc_dec) and std_enc_dec else None,
        "best_checkpoint_jepa": checkpoint_path_jepa if os.path.exists(checkpoint_path_jepa) and jepa_model else None
    }
