# Contents for src/training_engine.py
import torch
import torch.nn.functional as F # For F.one_hot
import os # For os.path.exists in early stopping save/load
import matplotlib.pyplot as plt
import numpy as np

# Note: Loss functions (mse_loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight)
# will be passed in via the 'losses_map' dictionary.
# Models (std_enc_dec, jepa_model, reward_mlp_enc_dec, reward_mlp_jepa, jepa_decoder) via 'models_map'.
# Optimizers (optimizer_std_enc_dec, etc.) via 'optimizers_map'.
# Dataloaders (train_dataloader, val_dataloader) via 'dataloaders_map'.
# Configs (early_stopping_config, enc_dec_mlp_config, jepa_mlp_config, main_config for num_epochs, log_interval) via 'config'.
# Device, action_dim, action_type also passed as arguments.

def run_training_epochs(
    models_map, optimizers_map, losses_map, dataloaders_map,
    device, config, action_dim, action_type,
    image_h_w, input_channels, # For reward MLP input calculation if needed
    std_enc_dec_loaded_successfully=False, # New argument
    jepa_loaded_successfully=False         # New argument
):
    # Unpack from maps/config for convenience
    std_enc_dec = models_map.get('std_enc_dec')
    jepa_model = models_map.get('jepa')
    reward_mlp_enc_dec = models_map.get('reward_mlp_enc_dec')
    reward_mlp_jepa = models_map.get('reward_mlp_jepa')
    jepa_decoder = models_map.get('jepa_decoder') # New model

    optimizer_std_enc_dec = optimizers_map.get('std_enc_dec')
    optimizer_jepa = optimizers_map.get('jepa')
    optimizer_reward_mlp_enc_dec = optimizers_map.get('reward_mlp_enc_dec')
    optimizer_reward_mlp_jepa = optimizers_map.get('reward_mlp_jepa')
    optimizer_jepa_decoder = optimizers_map.get('jepa_decoder') # New optimizer

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
    checkpoint_path_enc_dec = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_enc_dec', 'best_encoder_decoder.pth'))
    checkpoint_path_jepa = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_jepa', 'best_jepa.pth'))

    enc_dec_mlp_config = config.get('reward_predictors', {}).get('encoder_decoder_reward_mlp', {})
    jepa_mlp_config = config.get('reward_predictors', {}).get('jepa_reward_mlp', {})

    num_epochs = config.get('num_epochs', 10)
    log_interval = config.get('log_interval', 50) # General log interval

    print(f"Starting training, main models for {num_epochs} epochs...")

    # Early Stopping Trackers
    best_val_loss_enc_dec = float('inf')
    epochs_no_improve_enc_dec = 0
    best_val_loss_jepa = float('inf')
    epochs_no_improve_jepa = 0

    # Initialize early_stop flags: if model or optimizer is None, it's effectively "stopped" or disabled.
    early_stop_enc_dec = not (std_enc_dec and optimizer_std_enc_dec)
    early_stop_jepa = not (jepa_model and optimizer_jepa)

    # Handle Skip Training Flags based on loaded models
    training_options = config.get('training_options', {})
    skip_std_enc_dec_opt = training_options.get('skip_std_enc_dec_training_if_loaded', False)
    skip_jepa_opt = training_options.get('skip_jepa_training_if_loaded', False)

    if std_enc_dec_loaded_successfully and skip_std_enc_dec_opt and not early_stop_enc_dec:
        early_stop_enc_dec = True
        print("Standard Encoder/Decoder training will be skipped as a pre-trained model was loaded and skip option is enabled.")

    if jepa_loaded_successfully and skip_jepa_opt and not early_stop_jepa:
        early_stop_jepa = True
        print("JEPA model training will be skipped as a pre-trained model was loaded and skip option is enabled.")


    for epoch in range(num_epochs):
        epoch_loss_std = 0
        epoch_loss_jepa_pred = 0
        epoch_loss_jepa_aux = 0
        epoch_loss_reward_enc_dec = 0
        epoch_loss_reward_jepa = 0

        num_train_batches = len(train_dataloader) if train_dataloader else 0
        if num_train_batches == 0:
            print(f"Epoch {epoch+1} has no training data for main models. Skipping.")
            # Still need to check for JEPA decoder training later if it's enabled.
            # If both main models are skipped, this loop might be short.
            if early_stop_enc_dec and early_stop_jepa:
                 print("Both main models already early stopped or skipped. Checking JEPA Decoder.")
                 break # Break from main model epoch loop, then JEPA decoder loop will run.
            # continue # Continue if one of them might still train

        # Main model training loop
        if std_enc_dec and not early_stop_enc_dec: std_enc_dec.train()
        if jepa_model and not early_stop_jepa: jepa_model.train()
        if aux_loss_fn and hasattr(aux_loss_fn, 'train'):
            aux_loss_fn.train()

        # === Training Phase ===
        if not (early_stop_enc_dec and early_stop_jepa):
            for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
                s_t, r_t, s_t_plus_1 = s_t.to(device), r_t.to(device).float().unsqueeze(1), s_t_plus_1.to(device)

                if action_type == 'discrete':
                    if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
                    a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_processed = a_t.float().to(device)

                loss_std_item = 0
                if std_enc_dec and not early_stop_enc_dec:
                    optimizer_std_enc_dec.zero_grad()
                    predicted_s_t_plus_1 = std_enc_dec(s_t, a_t_processed)
                    loss_std = mse_loss_fn(predicted_s_t_plus_1, s_t_plus_1)
                    loss_std.backward()
                    optimizer_std_enc_dec.step()
                    loss_std_item = loss_std.item()
                    epoch_loss_std += loss_std_item

                current_loss_jepa_pred_item = 0
                current_loss_jepa_aux_item = 0
                current_total_loss_jepa_item = 0
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

                    current_loss_jepa_aux_item = current_loss_jepa_aux.item()
                    total_loss_jepa = loss_jepa_pred + current_loss_jepa_aux * aux_loss_weight
                    current_total_loss_jepa_item = total_loss_jepa.item()
                    total_loss_jepa.backward()
                    optimizer_jepa.step()
                    jepa_model.update_target_network()
                    epoch_loss_jepa_pred += current_loss_jepa_pred_item
                    epoch_loss_jepa_aux += current_loss_jepa_aux_item

                if (batch_idx + 1) % log_interval == 0:
                    log_msg = f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_train_batches}:"
                    if std_enc_dec and not early_stop_enc_dec: log_msg += f" StdEncDec L: {loss_std_item:.4f} |"
                    if jepa_model and not early_stop_jepa:
                        weighted_aux_loss_str = f"{(current_loss_jepa_aux_item * aux_loss_weight):.4f}" if aux_loss_fn and aux_loss_weight > 0 else "N/A"
                        log_msg += (f" JEPA Pred L: {current_loss_jepa_pred_item:.4f},"
                                    f" {aux_loss_name} AuxRawL: {current_loss_jepa_aux_item:.4f} (W: {weighted_aux_loss_str}),"
                                    f" Total JEPA L: {current_total_loss_jepa_item:.4f}")
                    if log_msg[-1] != ':': print(log_msg)

        # === Validation Phase ===
        if val_dataloader and ((std_enc_dec and not early_stop_enc_dec) or (jepa_model and not early_stop_jepa)):
            if std_enc_dec: std_enc_dec.eval()
            if jepa_model: jepa_model.eval()
            if aux_loss_fn and hasattr(aux_loss_fn, 'eval'): aux_loss_fn.eval()

            epoch_val_loss_std = 0
            epoch_val_loss_jepa_pred = 0
            epoch_val_loss_jepa_aux = 0

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
                        epoch_val_loss_jepa_aux += current_val_loss_jepa_aux.item()

            num_val_batches = len(val_dataloader)
            avg_val_loss_std = epoch_val_loss_std / num_val_batches if num_val_batches > 0 and std_enc_dec and not early_stop_enc_dec else float('inf')
            avg_val_loss_jepa_pred = epoch_val_loss_jepa_pred / num_val_batches if num_val_batches > 0 and jepa_model and not early_stop_jepa else float('inf')
            avg_val_loss_jepa_aux_raw = epoch_val_loss_jepa_aux / num_val_batches if num_val_batches > 0 and jepa_model and not early_stop_jepa else float('inf')
            avg_total_val_loss_jepa = avg_val_loss_jepa_pred + avg_val_loss_jepa_aux_raw * aux_loss_weight if jepa_model and not early_stop_jepa else float('inf')

            print(f"--- Epoch {epoch+1} Validation Summary ---")
            if std_enc_dec and not early_stop_enc_dec:
                print(f"  Avg Val StdEncDec L: {avg_val_loss_std:.4f}")
                if avg_val_loss_std < best_val_loss_enc_dec - delta:
                    best_val_loss_enc_dec = avg_val_loss_std
                    if checkpoint_path_enc_dec: os.makedirs(model_dir, exist_ok=True); torch.save(std_enc_dec.state_dict(), checkpoint_path_enc_dec)
                    epochs_no_improve_enc_dec = 0; print(f"  Encoder/Decoder: Val loss improved. Saved model to {checkpoint_path_enc_dec}")
                else:
                    epochs_no_improve_enc_dec += 1; print(f"  Encoder/Decoder: No val improvement for {epochs_no_improve_enc_dec} epochs.")
                    if epochs_no_improve_enc_dec >= patience: early_stop_enc_dec = True; print("  Encoder/Decoder: Early stopping triggered.")

            if jepa_model and not early_stop_jepa:
                print(f"  Avg Val JEPA Pred L: {avg_val_loss_jepa_pred:.4f}, {aux_loss_name} AuxRawL: {avg_val_loss_jepa_aux_raw:.4f}, Total Val JEPA L: {avg_total_val_loss_jepa:.4f}")
                if avg_total_val_loss_jepa < best_val_loss_jepa - delta:
                    best_val_loss_jepa = avg_total_val_loss_jepa
                    if checkpoint_path_jepa: os.makedirs(model_dir, exist_ok=True); torch.save(jepa_model.state_dict(), checkpoint_path_jepa)
                    epochs_no_improve_jepa = 0; print(f"  JEPA: Val loss improved. Saved model to {checkpoint_path_jepa}")
                else:
                    epochs_no_improve_jepa += 1; print(f"  JEPA: No val improvement for {epochs_no_improve_jepa} epochs.")
                    if epochs_no_improve_jepa >= patience: early_stop_jepa = True; print("  JEPA: Early stopping triggered.")

        # --- Reward MLP Training Loop ---
        # (Assuming this part remains largely unchanged, but ensure it respects early_stop_enc_dec and early_stop_jepa for its inputs)
        num_batches_reward_train = 0
        reward_enc_dec_train_needed = reward_mlp_enc_dec and enc_dec_mlp_config.get('enabled', False) and optimizer_reward_mlp_enc_dec and std_enc_dec # No longer tied to early_stop_enc_dec directly for the MLP itself
        reward_jepa_train_needed = reward_mlp_jepa and jepa_mlp_config.get('enabled', False) and optimizer_reward_mlp_jepa and jepa_model # No longer tied to early_stop_jepa directly

        if train_dataloader and (reward_enc_dec_train_needed or reward_jepa_train_needed):
            if std_enc_dec: std_enc_dec.eval()
            if jepa_model: jepa_model.eval()
            if reward_mlp_enc_dec and reward_enc_dec_train_needed: reward_mlp_enc_dec.train()
            if reward_mlp_jepa and reward_jepa_train_needed: reward_mlp_jepa.train()

            current_epoch_loss_reward_enc_dec = 0 # Track per epoch for avg
            current_epoch_loss_reward_jepa = 0    # Track per epoch for avg

            for reward_batch_idx, (s_t_reward, a_t_reward, r_t_reward, s_t_plus_1_reward) in enumerate(train_dataloader):
                s_t_reward, r_t_reward, s_t_plus_1_reward = s_t_reward.to(device), r_t_reward.to(device).float().unsqueeze(1), s_t_plus_1_reward.to(device)
                if action_type == 'discrete':
                    a_t_reward_processed = F.one_hot(a_t_reward.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_reward_processed = a_t_reward.float().to(device)
                num_batches_reward_train +=1

                if reward_enc_dec_train_needed:
                    optimizer_reward_mlp_enc_dec.zero_grad()
                    with torch.no_grad():
                        predicted_s_t_plus_1_for_reward = std_enc_dec(s_t_reward, a_t_reward_processed).detach()
                    input_enc_dec_reward_mlp = predicted_s_t_plus_1_for_reward.view(predicted_s_t_plus_1_for_reward.size(0), -1)
                    pred_reward_enc_dec = reward_mlp_enc_dec(input_enc_dec_reward_mlp)
                    loss_reward_enc_dec = mse_loss_fn(pred_reward_enc_dec, r_t_reward)
                    loss_reward_enc_dec.backward(); optimizer_reward_mlp_enc_dec.step()
                    current_epoch_loss_reward_enc_dec += loss_reward_enc_dec.item()
                    if (reward_batch_idx + 1) % enc_dec_mlp_config.get('log_interval', log_interval) == 0:
                        print(f"  Epoch {epoch+1}, Reward MLP (Enc-Dec) Batch {reward_batch_idx+1}/{num_train_batches}: Loss {loss_reward_enc_dec.item():.4f}")

                if reward_jepa_train_needed:
                    optimizer_reward_mlp_jepa.zero_grad()
                    with torch.no_grad():
                        pred_emb_for_reward, _, _, _ = jepa_model(s_t_reward, a_t_reward_processed, s_t_plus_1_reward)
                        input_jepa_reward_mlp = pred_emb_for_reward.detach()
                    pred_reward_jepa = reward_mlp_jepa(input_jepa_reward_mlp)
                    loss_reward_jepa = mse_loss_fn(pred_reward_jepa, r_t_reward)
                    loss_reward_jepa.backward(); optimizer_reward_mlp_jepa.step()
                    current_epoch_loss_reward_jepa += loss_reward_jepa.item()
                    if (reward_batch_idx + 1) % jepa_mlp_config.get('log_interval', log_interval) == 0:
                         print(f"  Epoch {epoch+1}, Reward MLP (JEPA) Batch {reward_batch_idx+1}/{num_train_batches}: Loss {loss_reward_jepa.item():.4f}")
            epoch_loss_reward_enc_dec = current_epoch_loss_reward_enc_dec # Store total for epoch avg
            epoch_loss_reward_jepa = current_epoch_loss_reward_jepa       # Store total for epoch avg


        # --- Epoch Summary ---
        print(f"--- Epoch {epoch+1}/{num_epochs} Training Summary ---")
        if std_enc_dec and not early_stop_enc_dec: print(f"  Avg Train StdEncDec L: {(epoch_loss_std / num_train_batches if num_train_batches > 0 else 0):.4f}")
        if jepa_model and not early_stop_jepa:
            avg_total_jepa_loss = (epoch_loss_jepa_pred / num_train_batches if num_train_batches > 0 else 0) + \
                                  (epoch_loss_jepa_aux / num_train_batches if num_train_batches > 0 else 0) * aux_loss_weight
            print(f"  Avg Train JEPA Pred L: {(epoch_loss_jepa_pred / num_train_batches if num_train_batches > 0 else 0):.4f}, "
                  f"{aux_loss_name} AuxRawL: {(epoch_loss_jepa_aux / num_train_batches if num_train_batches > 0 else 0):.4f}, "
                  f"Total Train JEPA L: {avg_total_jepa_loss:.4f}")
        if reward_enc_dec_train_needed: print(f"  Avg Train Reward MLP (Enc-Dec) L: {(epoch_loss_reward_enc_dec / num_batches_reward_train if num_batches_reward_train > 0 else 0):.4f}")
        if reward_jepa_train_needed: print(f"  Avg Train Reward MLP (JEPA) L: {(epoch_loss_reward_jepa / num_batches_reward_train if num_batches_reward_train > 0 else 0):.4f}")

        if early_stop_enc_dec and early_stop_jepa:
            print("Both main models (StdEncDec, JEPA) triggered early stopping or were skipped. Checking JEPA Decoder training next.")
            break # Break from main model epoch loop

    print("Main models training loop finished.")

    # --- JEPA State Decoder Training Loop ---
    checkpoint_path_jepa_decoder = None # Initialize for return value
    jepa_decoder_training_config = config.get('jepa_decoder_training', {})

    if jepa_decoder and optimizer_jepa_decoder and jepa_decoder_training_config.get('enabled', False):
        print("\nStarting JEPA State Decoder training...")

        jepa_decoder_early_stop_config = jepa_decoder_training_config.get('early_stopping', {})
        patience_decoder = jepa_decoder_early_stop_config.get('patience', 10)
        delta_decoder = jepa_decoder_early_stop_config.get('delta', 0.001)
        # Ensure model_dir is used for decoder checkpoint path
        decoder_cp_name = jepa_decoder_training_config.get('checkpoint_path', 'best_jepa_decoder.pth')
        checkpoint_path_jepa_decoder = os.path.join(model_dir, decoder_cp_name)

        best_val_loss_jepa_decoder = float('inf')
        epochs_no_improve_jepa_decoder = 0
        early_stop_jepa_decoder = False

        num_epochs_decoder = jepa_decoder_training_config.get('num_epochs', 50)
        decoder_log_interval = jepa_decoder_training_config.get('log_interval', log_interval) # Use its own or general log_interval

        for decoder_epoch in range(num_epochs_decoder):
            if jepa_model: jepa_model.eval() # JEPA model (encoder part) should be in eval mode
            jepa_decoder.train()
            epoch_loss_jepa_decoder_train = 0
            num_batches_jepa_decoder_train = len(train_dataloader) if train_dataloader else 0

            if num_batches_jepa_decoder_train == 0:
                print(f"JEPA Decoder Epoch {decoder_epoch+1} has no training data. Skipping training phase.")
                # If no training data, validation might still run if it was the plan.
            else:
                for batch_idx, (s_t, a_t, _, s_t_plus_1) in enumerate(train_dataloader):
                    s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
                    if action_type == 'discrete':
                        if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
                        a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
                    else:
                        a_t_processed = a_t.float().to(device)

                    optimizer_jepa_decoder.zero_grad()

                    with torch.no_grad(): # JEPA model provides predictor embedding
                        if not jepa_model: # Should not happen if decoder is enabled and needs jepa
                            print("Error: JEPA model is None, cannot get predictor embedding for JEPA decoder.")
                            early_stop_jepa_decoder = True; break
                        pred_emb, _, _, _ = jepa_model(s_t, a_t_processed, s_t_plus_1)

                    jepa_predictor_output = pred_emb.detach() # Detach before feeding to decoder
                    reconstructed_s_t_plus_1 = jepa_decoder(jepa_predictor_output)

                    loss_jepa_decoder = mse_loss_fn(reconstructed_s_t_plus_1, s_t_plus_1)
                    loss_jepa_decoder.backward()
                    optimizer_jepa_decoder.step()

                    epoch_loss_jepa_decoder_train += loss_jepa_decoder.item()

                    if (batch_idx + 1) % decoder_log_interval == 0:
                        print(f"  JEPA Decoder Epoch {decoder_epoch+1}, Batch {batch_idx+1}/{num_batches_jepa_decoder_train}: Train Loss {loss_jepa_decoder.item():.4f}")
                if early_stop_jepa_decoder: break # from inner batch loop if error occurred

            avg_train_loss_jepa_decoder = epoch_loss_jepa_decoder_train / num_batches_jepa_decoder_train if num_batches_jepa_decoder_train > 0 else 0

            # JEPA Decoder Validation Phase
            if val_dataloader:
                jepa_decoder.eval()
                if jepa_model: jepa_model.eval() # Ensure JEPA (encoder) is also in eval for validation consistency

                epoch_val_loss_jepa_decoder = 0
                num_val_batches_decoder = len(val_dataloader)

                with torch.no_grad():
                    for val_batch_idx, (s_t_val, a_t_val, _, s_t_plus_1_val) in enumerate(val_dataloader): # Added enumerate for val_batch_idx
                        s_t_val, s_t_plus_1_val = s_t_val.to(device), s_t_plus_1_val.to(device)
                        if action_type == 'discrete':
                            if a_t_val.ndim == 1: a_t_val = a_t_val.unsqueeze(1)
                            a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                        else:
                            a_t_val_processed = a_t_val.float().to(device)

                        if not jepa_model:
                             print("Error: JEPA model is None during JEPA decoder validation.")
                             early_stop_jepa_decoder = True; break
                        pred_emb_val, _, _, _ = jepa_model(s_t_val, a_t_val_processed, s_t_plus_1_val)
                        jepa_predictor_output_val = pred_emb_val.detach()

                        reconstructed_s_t_plus_1_val = jepa_decoder(jepa_predictor_output_val)
                        val_loss_dec = mse_loss_fn(reconstructed_s_t_plus_1_val, s_t_plus_1_val)
                        epoch_val_loss_jepa_decoder += val_loss_dec.item()

                        # --- Plotting logic for JEPA State Decoder validation ---
                        if val_batch_idx == 0: # Plot only for the first validation batch to save time/space, or make configurable
                            validation_plot_dir = jepa_decoder_training_config.get('validation_plot_dir', "validation_plots/")
                            os.makedirs(validation_plot_dir, exist_ok=True)

                            num_plot_samples = min(4, s_t_val.shape[0]) # Plot up to 4 samples
                            for i in range(num_plot_samples):
                                true_img = s_t_plus_1_val[i].cpu().numpy()
                                pred_img = reconstructed_s_t_plus_1_val[i].cpu().numpy()

                                # Handle image tensor format (C, H, W) -> (H, W, C) or (H, W)
                                if true_img.shape[0] == 1 or true_img.shape[0] == 3: # Grayscale or RGB
                                    true_img = np.transpose(true_img, (1, 2, 0))
                                    pred_img = np.transpose(pred_img, (1, 2, 0))

                                if true_img.shape[-1] == 1: # Grayscale, squeeze channel dim
                                    true_img = true_img.squeeze(axis=2)
                                    pred_img = pred_img.squeeze(axis=2)

                                # Clip values for visualization if they are floats
                                if true_img.dtype == np.float32 or true_img.dtype == np.float64:
                                    true_img = np.clip(true_img, 0, 1)
                                    pred_img = np.clip(pred_img, 0, 1)

                                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                                axes[0].imshow(true_img)
                                axes[0].set_title("True Image")
                                axes[0].axis('off')

                                axes[1].imshow(pred_img)
                                axes[1].set_title("Predicted Image")
                                axes[1].axis('off')

                                plot_filename = os.path.join(validation_plot_dir, f"epoch_{decoder_epoch+1}_valbatch_{val_batch_idx}_sample_{i}.png")
                                plt.savefig(plot_filename)
                                plt.close(fig)
                            print(f"  JEPA Decoder: Saved {num_plot_samples} validation image samples to {validation_plot_dir} for epoch {decoder_epoch+1}, batch {val_batch_idx}")
                        # --- End plotting logic ---

                if early_stop_jepa_decoder: break # from epoch loop if error in val

                avg_val_loss_jepa_decoder = epoch_val_loss_jepa_decoder / num_val_batches_decoder if num_val_batches_decoder > 0 else float('inf')
                print(f"--- JEPA Decoder Epoch {decoder_epoch+1} Validation Summary ---")
                print(f"  Avg Val JEPA Decoder L: {avg_val_loss_jepa_decoder:.4f}")

                if avg_val_loss_jepa_decoder < best_val_loss_jepa_decoder - delta_decoder:
                    best_val_loss_jepa_decoder = avg_val_loss_jepa_decoder
                    if checkpoint_path_jepa_decoder:
                        os.makedirs(model_dir, exist_ok=True)
                        torch.save(jepa_decoder.state_dict(), checkpoint_path_jepa_decoder)
                    epochs_no_improve_jepa_decoder = 0
                    print(f"  JEPA Decoder: Val loss improved. Saved model to {checkpoint_path_jepa_decoder}")
                else:
                    epochs_no_improve_jepa_decoder += 1
                    print(f"  JEPA Decoder: No val improvement for {epochs_no_improve_jepa_decoder} epochs.")
                    if epochs_no_improve_jepa_decoder >= patience_decoder:
                        early_stop_jepa_decoder = True
                        print("  JEPA Decoder: Early stopping triggered.")
            else: # No validation dataloader
                print(f"--- JEPA Decoder Epoch {decoder_epoch+1} Training Summary (No Validation) ---")
                print(f"  Avg Train JEPA Decoder L: {avg_train_loss_jepa_decoder:.4f}")
                if checkpoint_path_jepa_decoder: # Save last epoch if no validation
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(jepa_decoder.state_dict(), checkpoint_path_jepa_decoder)
                    print(f"  JEPA Decoder: Saved model from last epoch to {checkpoint_path_jepa_decoder} (no validation set)")


            if early_stop_jepa_decoder:
                print(f"JEPA State Decoder early stopping at epoch {decoder_epoch+1}.")
                break # Break from JEPA decoder epoch loop

        print("JEPA State Decoder training finished.")
        if checkpoint_path_jepa_decoder and os.path.exists(checkpoint_path_jepa_decoder):
            print(f"Loading best JEPA State Decoder model from {checkpoint_path_jepa_decoder}")
            jepa_decoder.load_state_dict(torch.load(checkpoint_path_jepa_decoder, map_location=device))
    else:
        if not jepa_decoder: print("JEPA State Decoder model not provided.")
        elif not optimizer_jepa_decoder: print("Optimizer for JEPA State Decoder not provided.")
        elif not jepa_decoder_training_config.get('enabled', False): print("JEPA State Decoder training is disabled in config.")


    print("All training processes finished from training_engine.")
    return {
        "best_checkpoint_enc_dec": checkpoint_path_enc_dec if std_enc_dec and os.path.exists(checkpoint_path_enc_dec) else None,
        "best_checkpoint_jepa": checkpoint_path_jepa if jepa_model and os.path.exists(checkpoint_path_jepa) else None,
        "best_checkpoint_jepa_decoder": checkpoint_path_jepa_decoder if jepa_decoder and jepa_decoder_training_config.get('enabled', False) and checkpoint_path_jepa_decoder and os.path.exists(checkpoint_path_jepa_decoder) else None
    }
