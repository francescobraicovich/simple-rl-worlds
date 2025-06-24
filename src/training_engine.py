"""
Orchestrates the overall training process for various models.

This module defines the `run_training_epochs` function, which serves as the main
entry point for training different types of models, including Standard
Encoder-Decoders, JEPA models, and their associated Reward MLPs and JEPA State
Decoders.

The actual epoch-level training and validation logic for each model component
has been refactored into separate modules within the `src.training_loops` package.
This `training_engine` module is responsible for:
- Initializing models, optimizers, and loss functions.
- Managing early stopping criteria.
- Iterating through the specified number of epochs.
- Calling the appropriate training loop functions from `src.training_loops`
  for each model component (e.g., main model, reward predictor, state decoder).
- Handling model checkpointing (loading best models after main training for
  use in subsequent training phases like reward prediction or decoding).
- Defining and logging metrics to Weights & Biases (wandb).
"""
import torch
import torch.nn.functional as F # For F.one_hot
import os # For os.path.exists in early stopping save/load
import wandb # For wandb.Image

from .training_loops.epoch_loop import train_validate_model_epoch
from .training_loops.reward_predictor_loop import train_reward_mlp_epoch
from .training_loops.jepa_decoder_loop import train_jepa_state_decoder
from .training_loops.larp_training_loop import train_larp_epoch # Added for LARP

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
    # LARP Models
    larp_enc_dec_model = models_map.get('larp_enc_dec')
    larp_jepa_model = models_map.get('larp_jepa')

    optimizer_std_enc_dec = optimizers_map.get('std_enc_dec')
    optimizer_jepa = optimizers_map.get('jepa')
    optimizer_reward_mlp_enc_dec = optimizers_map.get('reward_mlp_enc_dec')
    optimizer_reward_mlp_jepa = optimizers_map.get('reward_mlp_jepa')
    optimizer_jepa_decoder = optimizers_map.get('jepa_decoder')
    # LARP Optimizers
    optimizer_larp_enc_dec = optimizers_map.get('larp_enc_dec')
    optimizer_larp_jepa = optimizers_map.get('larp_jepa')

    mse_loss_fn = losses_map['mse']
    aux_loss_fn = losses_map.get('aux_fn')
    aux_loss_name = losses_map.get('aux_name', "None")
    aux_loss_weight = losses_map.get('aux_weight', 0.0)

    # Get auxiliary loss usage flags from config
    aux_loss_config = config.get('models', {}).get('auxiliary_loss', {})
    use_aux_for_jepa = aux_loss_config.get('use_for_jepa', False) # Default to False if not specified
    use_aux_for_enc_dec = aux_loss_config.get('use_for_enc_dec', False) # Default to False

    train_dataloader = dataloaders_map['train']
    val_dataloader = dataloaders_map.get('val')

    # Configs
    training_config = config.get('training', {})
    early_stopping_config = training_config.get('early_stopping', {})
    model_loading_config = config.get('model_loading', {})
    model_dir = model_loading_config.get('dir', 'trained_models/')
    patience = early_stopping_config.get('patience', 10)
    delta = early_stopping_config.get('delta', 0.001)

    # Checkpoint paths
    checkpoint_path_enc_dec = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_enc_dec', 'best_encoder_decoder.pth'))
    checkpoint_path_jepa = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_jepa', 'best_jepa.pth'))

    models_config = config.get('models', {})
    enc_dec_mlp_config = models_config.get('reward_predictors', {}).get('encoder_decoder_reward_mlp', {})
    jepa_mlp_config = models_config.get('reward_predictors', {}).get('jepa_reward_mlp', {})
    print(f'Jepa reward MLP config: {jepa_mlp_config}')

    num_epochs = training_config.get('num_epochs', 10)
    log_interval = training_config.get('log_interval', 50)

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
            # wandb_run.define_metric(f"reward_mlp/{model_prefix}/val/*", step_metric=f"reward_mlp/{model_prefix}/epoch") # Original, possibly for batch val
            wandb_run.define_metric(f"reward_mlp/{model_prefix}/train_epoch_avg/*", step_metric=f"reward_mlp/{model_prefix}/epoch")
            wandb_run.define_metric(f"reward_mlp/{model_prefix}/val_epoch_avg/*", step_metric=f"reward_mlp/{model_prefix}/epoch") # New for epoch-wise val

        # Define metrics for JEPA_Decoder
        wandb_run.define_metric("JEPA_Decoder/train/step")      # Step for batch-wise train logs
        wandb_run.define_metric("JEPA_Decoder/epoch")           # Step for epoch-wise logs

        # Train metrics (batch-wise)
        wandb_run.define_metric("JEPA_Decoder/train/total_loss", step_metric="JEPA_Decoder/train/step")
        wandb_run.define_metric("JEPA_Decoder/train/loss_mse_reconstruction", step_metric="JEPA_Decoder/train/step")
        wandb_run.define_metric("JEPA_Decoder/train/loss_mse_diff", step_metric="JEPA_Decoder/train/step")
        wandb_run.define_metric("JEPA_Decoder/train/Learning_Rate", step_metric="JEPA_Decoder/train/step") # Explicitly defined

        # Validation metrics (epoch-wise)
        wandb_run.define_metric("JEPA_Decoder/val/total_loss", step_metric="JEPA_Decoder/epoch")
        wandb_run.define_metric("JEPA_Decoder/val/loss_mse_reconstruction", step_metric="JEPA_Decoder/epoch")
        wandb_run.define_metric("JEPA_Decoder/val/loss_mse_diff", step_metric="JEPA_Decoder/epoch")

        # Train epoch average metrics (epoch-wise)
        wandb_run.define_metric("JEPA_Decoder/train_epoch_avg/total_loss", step_metric="JEPA_Decoder/epoch")
        wandb_run.define_metric("JEPA_Decoder/train_epoch_avg/loss_mse_reconstruction", step_metric="JEPA_Decoder/epoch")
        wandb_run.define_metric("JEPA_Decoder/train_epoch_avg/loss_mse_diff", step_metric="JEPA_Decoder/epoch")

        # Add WandB metrics for LARP
        for model_prefix_larp in ["StdEncDec", "JEPA"]: # Using more descriptive internal var name
            # LARP metrics (model_name_log_prefix in train_larp_epoch will be "LARP (Enc-Dec)" or "LARP (JEPA)")
            # So wandb logs will be like "larp/LARP (Enc-Dec)/train/step"
            # The "StdEncDec" and "JEPA" here are just for grouping in define_metric, actual log path comes from model_name_log_prefix

            # The model_name_log_prefix passed to train_larp_epoch will include "LARP (...)"
            # So, the actual logged names will be like:
            # larp/LARP (Enc-Dec)/train/loss_step
            # larp/LARP (Enc-Dec)/train/step
            # larp/LARP (Enc-Dec)/train_epoch_avg/loss
            # larp/LARP (Enc-Dec)/val_epoch_avg/loss
            # larp/LARP (Enc-Dec)/epoch  (this will be current_epoch_main_training * num_epochs_larp + larp_epoch_inner)
            # larp/LARP (Enc-Dec)/epoch_inner

            # Define metrics based on how they are logged in train_larp_epoch
            # Using a generic placeholder like "LARP_MODEL_TYPE" for the define_metric part
            # The actual prefix "LARP (Enc-Dec)" or "LARP (JEPA)" will be used in wandb.log calls.
            # To make define_metric effective, it should match the logged metric names.
            # Let's use the actual prefixes that will be logged.

            # For LARP (Enc-Dec)
            wandb_run.define_metric("larp/LARP (Enc-Dec)/train/step")
            wandb_run.define_metric("larp/LARP (Enc-Dec)/epoch") # This is the global LARP epoch step
            wandb_run.define_metric("larp/LARP (Enc-Dec)/epoch_inner") # This is the inner epoch for a specific LARP training run
            wandb_run.define_metric("larp/LARP (Enc-Dec)/epoch_progress_main")

            wandb_run.define_metric("larp/LARP (Enc-Dec)/train/loss_step", step_metric="larp/LARP (Enc-Dec)/train/step")
            wandb_run.define_metric("larp/LARP (Enc-Dec)/train_epoch_avg/loss", step_metric="larp/LARP (Enc-Dec)/epoch")
            wandb_run.define_metric("larp/LARP (Enc-Dec)/val_epoch_avg/loss", step_metric="larp/LARP (Enc-Dec)/epoch")

            # For LARP (JEPA)
            wandb_run.define_metric("larp/LARP (JEPA)/train/step")
            wandb_run.define_metric("larp/LARP (JEPA)/epoch")
            wandb_run.define_metric("larp/LARP (JEPA)/epoch_inner")
            wandb_run.define_metric("larp/LARP (JEPA)/epoch_progress_main")

            wandb_run.define_metric("larp/LARP (JEPA)/train/loss_step", step_metric="larp/LARP (JEPA)/train/step")
            wandb_run.define_metric("larp/LARP (JEPA)/train_epoch_avg/loss", step_metric="larp/LARP (JEPA)/epoch")
            wandb_run.define_metric("larp/LARP (JEPA)/val_epoch_avg/loss", step_metric="larp/LARP (JEPA)/epoch")

    print(f"Starting training, main models for up to {num_epochs} epochs...")
    current_main_epoch_for_larp = 0 # Variable to track current main epoch for LARP logging

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
    training_options = training_config.get('options', {})
    skip_std_enc_dec_opt = training_options.get('skip_std_enc_dec_training_if_loaded', False)
    skip_jepa_opt = training_options.get('skip_jepa_training_if_loaded', False)

    if std_enc_dec_loaded_successfully and skip_std_enc_dec_opt and not early_stopping_state_enc_dec['early_stop_flag']:
        early_stopping_state_enc_dec['early_stop_flag'] = True
        print("Standard Encoder/Decoder training will be skipped as a pre-trained model was loaded and skip option is enabled.")
    if jepa_loaded_successfully and skip_jepa_opt and not early_stopping_state_jepa['early_stop_flag']:
        early_stopping_state_jepa['early_stop_flag'] = True
        print("JEPA model training will be skipped as a pre-trained model was loaded and skip option is enabled.")

    for epoch in range(num_epochs):
        current_main_epoch_for_larp = epoch + 1 # Update current main epoch
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")

        # --- Standard Encoder/Decoder Training ---
        if std_enc_dec and not early_stopping_state_enc_dec['early_stop_flag']:
            print(f"Standard Encoder/Decoder: Running training and validation for (Epoch {epoch+1})...")
            early_stopping_state_enc_dec, _, _ = train_validate_model_epoch(
                model=std_enc_dec, optimizer=optimizer_std_enc_dec, train_dataloader=train_dataloader,
                val_dataloader=val_dataloader, loss_fn=mse_loss_fn, 
                aux_loss_fn=aux_loss_fn, # Pass the general aux_loss_fn
                aux_loss_name=aux_loss_name, # Pass its name
                aux_loss_weight=aux_loss_weight, # Pass its weight
                use_aux_for_jepa=False, # Not a JEPA model
                use_aux_for_enc_dec=use_aux_for_enc_dec, # Use flag from config
                device=device, epoch_num=epoch + 1, log_interval=log_interval,
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
            early_stopping_state_jepa, _, _ = train_validate_model_epoch(
                model=jepa_model, optimizer=optimizer_jepa, train_dataloader=train_dataloader,
                val_dataloader=val_dataloader, loss_fn=mse_loss_fn, 
                aux_loss_fn=aux_loss_fn, # Pass the general aux_loss_fn
                aux_loss_name=aux_loss_name, # Pass its name
                aux_loss_weight=aux_loss_weight, # Pass its weight
                use_aux_for_jepa=use_aux_for_jepa, # Use flag from config
                use_aux_for_enc_dec=False, # Not an Encoder-Decoder model in this context
                device=device, epoch_num=epoch + 1, log_interval=log_interval, 
                action_dim=action_dim, action_type=action_type,
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
    # --- Reward MLP Training using train_reward_mlp_epoch ---
    if reward_mlp_enc_dec and enc_dec_mlp_config.get('enabled', False) and optimizer_reward_mlp_enc_dec and std_enc_dec and train_dataloader:
        print("\nStarting Reward MLP (Standard Encoder/Decoder) training...")
        early_stopping_patience_enc_dec_reward_mlp = enc_dec_mlp_config.get('early_stopping_patience', 15)
        train_reward_mlp_epoch(
            reward_mlp_model=reward_mlp_enc_dec,
            base_model=std_enc_dec, # This is the best loaded std_enc_dec
            optimizer_reward_mlp=optimizer_reward_mlp_enc_dec,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader, # Added
            loss_fn=mse_loss_fn,
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            model_name_log_prefix="Reward MLP (Enc-Dec)",
            num_epochs_reward_mlp=enc_dec_mlp_config.get('num_epochs', 1), # Default to 1 epoch if not specified
            log_interval_reward_mlp=enc_dec_mlp_config.get('log_interval', log_interval), # Use specific or general log_interval
            early_stopping_patience=early_stopping_patience_enc_dec_reward_mlp, # Added
            is_jepa_base_model=False,
            wandb_run=wandb_run
        )
    elif enc_dec_mlp_config.get('enabled', False):
        print("Reward MLP (Enc-Dec) training skipped due to missing components (model, optimizer, base_model, or dataloader).")

    if reward_mlp_jepa and jepa_mlp_config.get('enabled', False) and optimizer_reward_mlp_jepa and jepa_model and train_dataloader:
        print("\nStarting Reward MLP (JEPA) training...")
        early_stopping_patience_jepa_reward_mlp = jepa_mlp_config.get('early_stopping_patience', 15)
        train_reward_mlp_epoch(
            reward_mlp_model=reward_mlp_jepa,
            base_model=jepa_model, # This is the best loaded jepa_model
            optimizer_reward_mlp=optimizer_reward_mlp_jepa,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader, # Added
            loss_fn=mse_loss_fn,
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            model_name_log_prefix="Reward MLP (JEPA)",
            num_epochs_reward_mlp=jepa_mlp_config.get('num_epochs', 1), # Default to 1 epoch
            log_interval_reward_mlp=jepa_mlp_config.get('log_interval', log_interval),
            early_stopping_patience=early_stopping_patience_jepa_reward_mlp, # Added
            is_jepa_base_model=True,
            wandb_run=wandb_run
        )
    elif jepa_mlp_config.get('enabled', False):
        print("Reward MLP (JEPA) training skipped due to missing components (model, optimizer, base_model, or dataloader).")

    # --- LARP Training ---
    # This happens after all main models and reward MLPs are trained,
    # using the best available checkpoints of the base models.

    # Load best models again just to be certain, especially if other components might have altered them in-memory.
    # This ensures LARP trains on the truly "best" versions of std_enc_dec and jepa_model.
    if std_enc_dec and early_stopping_state_enc_dec['checkpoint_path'] and os.path.exists(early_stopping_state_enc_dec['checkpoint_path']):
        print(f"Reloading best Standard Encoder/Decoder from {early_stopping_state_enc_dec['checkpoint_path']} for LARP training.")
        std_enc_dec.load_state_dict(torch.load(early_stopping_state_enc_dec['checkpoint_path'], map_location=device))
        std_enc_dec.eval() # Ensure it's in eval mode
    if jepa_model and early_stopping_state_jepa['checkpoint_path'] and os.path.exists(early_stopping_state_jepa['checkpoint_path']):
        print(f"Reloading best JEPA model from {early_stopping_state_jepa['checkpoint_path']} for LARP training.")
        jepa_model.load_state_dict(torch.load(early_stopping_state_jepa['checkpoint_path'], map_location=device))
        jepa_model.eval() # Ensure it's in eval mode

    larp_main_config = models_config.get('reward_predictors', {}).get('larp', {})
    enc_dec_larp_specific_config = larp_main_config.get('encoder_decoder_larp', {})
    jepa_larp_specific_config = larp_main_config.get('jepa_larp', {})

    # LARP for Encoder-Decoder
    if (larp_enc_dec_model and enc_dec_larp_specific_config.get('enabled', False) and
        optimizer_larp_enc_dec and std_enc_dec and train_dataloader):
        print("\nStarting LARP (Standard Encoder/Decoder) training...")
        train_larp_epoch(
            larp_model=larp_enc_dec_model,
            base_model=std_enc_dec, # Best loaded std_enc_dec
            optimizer_larp=optimizer_larp_enc_dec,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=mse_loss_fn, # Assuming MSE loss for rewards
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            model_name_log_prefix="LARP (Enc-Dec)", # Matches wandb define_metric
            num_epochs_larp=enc_dec_larp_specific_config.get('num_epochs', 1),
            log_interval_larp=enc_dec_larp_specific_config.get('log_interval', log_interval),
            early_stopping_patience=enc_dec_larp_specific_config.get('early_stopping_patience', 15),
            is_jepa_base_model=False, # Base model is Encoder-Decoder type
            wandb_run=wandb_run,
            current_epoch_main_training=current_main_epoch_for_larp # Or a fixed value like num_epochs if LARP trains once after all main epochs
        )
    elif enc_dec_larp_specific_config.get('enabled', False):
        print("LARP (Enc-Dec) training skipped due to missing components (model, optimizer, base_model, or dataloader).")

    # LARP for JEPA
    if (larp_jepa_model and jepa_larp_specific_config.get('enabled', False) and
        optimizer_larp_jepa and jepa_model and train_dataloader):
        print("\nStarting LARP (JEPA) training...")
        train_larp_epoch(
            larp_model=larp_jepa_model,
            base_model=jepa_model, # Best loaded jepa_model
            optimizer_larp=optimizer_larp_jepa,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=mse_loss_fn, # Assuming MSE loss for rewards
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            model_name_log_prefix="LARP (JEPA)", # Matches wandb define_metric
            num_epochs_larp=jepa_larp_specific_config.get('num_epochs', 1),
            log_interval_larp=jepa_larp_specific_config.get('log_interval', log_interval),
            early_stopping_patience=jepa_larp_specific_config.get('early_stopping_patience', 15),
            is_jepa_base_model=True, # Base model is JEPA type
            wandb_run=wandb_run,
            current_epoch_main_training=current_main_epoch_for_larp # Or a fixed value
        )
    elif jepa_larp_specific_config.get('enabled', False):
        print("LARP (JEPA) training skipped due to missing components (model, optimizer, base_model, or dataloader).")


    # --- JEPA State Decoder Training Loop ---
    # The JEPA State Decoder training loop remains as is, but it will use the
    # potentially best JEPA model loaded above.
    # --- JEPA State Decoder Training ---
    final_checkpoint_path_jepa_decoder = None
    jepa_decoder_training_config = models_config.get('jepa', {}).get('decoder_training', {})

    if (jepa_decoder and optimizer_jepa_decoder and jepa_model and
        jepa_decoder_training_config.get('enabled', False) and train_dataloader):
        # Note: jepa_model here is the one loaded from its best checkpoint (if saved)
        final_checkpoint_path_jepa_decoder = train_jepa_state_decoder(
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