import torch
import torch.nn.functional as F
import wandb
import time

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
    model_name_log_prefix,
    num_epochs_larp,
    log_interval_larp,
    early_stopping_patience,
    is_jepa_base_model,
    wandb_run,
    current_epoch_main_training
):
    """
    Training loop for Look-Ahead Reward Predictor (LARP).
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    total_train_steps = 0

    for epoch in range(num_epochs_larp):
        larp_model.train()
        if base_model:
            base_model.eval()

        train_loss_sum = 0.0
        num_train_batches = 0

        # Training phase
        for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
            s_t, a_t, r_t, s_t_plus_1 = s_t.to(device), a_t.to(device), r_t.to(device), s_t_plus_1.to(device)

            if action_type == 'discrete':
                if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
                a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
            else:
                a_t_processed = a_t.float().to(device)

            with torch.no_grad():
                if is_jepa_base_model:
                    predicted_latent_s_t_plus_1, _, online_s_t_representation, _ = base_model(s_t, a_t_processed, s_t_plus_1)
                    action_embedding = base_model.action_embedding(a_t_processed)
                    input_to_larp = torch.cat((
                        online_s_t_representation.detach(),
                        predicted_latent_s_t_plus_1.detach(),
                        action_embedding.detach()
                    ), dim=1)
                else:
                    encoded_s_t = base_model.encoder(s_t)
                    action_embedding = base_model.action_embedding(a_t_processed)

                    if hasattr(base_model, 'predictor') and hasattr(base_model, 'decoder') and callable(getattr(base_model, 'predictor')):
                        predictor_input_jepa_style = torch.cat((encoded_s_t, action_embedding), dim=-1)
                        intermediate_predictor_features = base_model.predictor(predictor_input_jepa_style)
                        input_to_larp = torch.cat((
                            encoded_s_t.detach(),
                            action_embedding.detach(),
                            intermediate_predictor_features.detach()
                        ), dim=1)
                    else:
                        input_to_larp = torch.cat((
                            encoded_s_t.detach(),
                            action_embedding.detach()
                        ), dim=1)

            optimizer_larp.zero_grad()
            pred_reward = larp_model(input_to_larp)
            loss = loss_fn(pred_reward, r_t.unsqueeze(1).float())
            loss.backward()
            optimizer_larp.step()

            train_loss_sum += loss.item()
            num_train_batches += 1
            total_train_steps += 1

            if wandb_run and batch_idx % log_interval_larp == 0:
                wandb.log({
                    f"larp/{model_name_log_prefix}/train/loss_step": loss.item(),
                    f"larp/{model_name_log_prefix}/train/step": total_train_steps,
                    f"larp/{model_name_log_prefix}/epoch_progress_main": current_epoch_main_training
                })

        avg_train_loss = train_loss_sum / num_train_batches if num_train_batches > 0 else 0

        # Validation phase
        larp_model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for s_t, a_t, r_t, s_t_plus_1 in val_dataloader:
                s_t, a_t, r_t, s_t_plus_1 = s_t.to(device), a_t.to(device), r_t.to(device), s_t_plus_1.to(device)

                if action_type == 'discrete':
                    if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
                    a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_processed = a_t.float().to(device)

                if is_jepa_base_model:
                    predicted_latent_s_t_plus_1, _, online_s_t_representation, _ = base_model(s_t, a_t_processed, s_t_plus_1)
                    action_embedding = base_model.action_embedding(a_t_processed)
                    input_to_larp = torch.cat((online_s_t_representation, predicted_latent_s_t_plus_1, action_embedding), dim=1)
                else:
                    encoded_s_t = base_model.encoder(s_t)
                    action_embedding = base_model.action_embedding(a_t_processed)
                    if hasattr(base_model, 'predictor') and hasattr(base_model, 'decoder') and callable(getattr(base_model, 'predictor')):
                        predictor_input_jepa_style = torch.cat((encoded_s_t, action_embedding), dim=-1)
                        intermediate_predictor_features = base_model.predictor(predictor_input_jepa_style)
                        input_to_larp = torch.cat((encoded_s_t, action_embedding, intermediate_predictor_features), dim=1)
                    else:
                        input_to_larp = torch.cat((encoded_s_t, action_embedding), dim=1)

                pred_reward = larp_model(input_to_larp)
                val_loss = loss_fn(pred_reward, r_t.unsqueeze(1).float())
                val_loss_sum += val_loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss_sum / num_val_batches if num_val_batches > 0 else 0

        print(f"{model_name_log_prefix} - Epoch {epoch+1}/{num_epochs_larp}: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        if wandb_run:
            wandb.log({
                f"larp/{model_name_log_prefix}/train_epoch_avg/loss": avg_train_loss,
                f"larp/{model_name_log_prefix}/val_epoch_avg/loss": avg_val_loss,
                f"larp/{model_name_log_prefix}/epoch": current_epoch_main_training * num_epochs_larp + epoch,
                f"larp/{model_name_log_prefix}/epoch_inner": epoch + 1
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"{model_name_log_prefix} - Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Finished training {model_name_log_prefix}. Best Val Loss: {best_val_loss:.4f}")
