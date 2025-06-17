# 06 Usage Guide

This guide provides comprehensive instructions for setting up the project, configuring experiments, and running the codebase. It is intended for researchers and developers looking to use and extend this framework for investigating world models.

## 1. Prerequisites and Setup

### Cloning the Repository
First, clone the repository to your local machine:
```bash
git clone <repository-url>
cd <repository-name>
```

### Python Virtual Environment (Recommended)
It is highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts with other projects.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Installing Dependencies
Install the required Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Specific Installations for Gymnasium Environments
Depending on the `environment_name` you choose in `config.yaml`, you may need to install additional packages for specific Gymnasium environments:

-   **Atari Games (e.g., `ALE/Pong-v5`, `ALE/Breakout-v5`):**
    ```bash
    pip install gymnasium[atari] gymnasium[accept-rom-license]
    ```
    The `accept-rom-license` is crucial for downloading and using Atari ROMs.

-   **Box2D Environments (e.g., `CarRacing-v2`, `BipedalWalker-v3`):**
    ```bash
    pip install gymnasium[box2d]
    ```
    On Linux systems, Box2D might require `swig` to be installed. You can typically install it using your system's package manager (e.g., `sudo apt-get install swig`).

### Automatic Directory Creation
Upon running `main.py` for the first time, the following directories will be automatically created in the project root if they don't already exist:
-   `datasets/`: For storing collected trajectory data.
-   `trained_models/`: For saving trained model checkpoints.
-   `validation_plots/`: For saving visual outputs like image reconstructions from the JEPA State Decoder during validation, if enabled (controlled by `jepa_decoder_training.validation_plot_dir` in `config.yaml`).

## 2. Core Configuration (`config.yaml`)

The `config.yaml` file is the central hub for controlling all aspects of the experiments, from environment selection and data collection to model architecture and training parameters. Below is a breakdown of key configuration sections:

### Environment Configuration
-   `environment_name`: Specifies the Gymnasium environment to use (e.g., `"CarRacing-v2"`, `"ALE/Pong-v5"`).
-   `input_channels`: Number of channels in the input image (e.g., 3 for RGB, 1 for grayscale). This should match the environment's observation space.
-   `image_size`: Target integer size (e.g., 64) for processed images (height and width are assumed to be the same). Images will be resized to `(image_size, image_size)`.

### Data Collection
-   `num_episodes_data_collection`: Number of episodes to collect for building the dataset.
-   `max_steps_per_episode_data_collection`: Maximum number of steps to run within each episode during data collection.
-   `dataset_dir`: Directory where datasets are stored (e.g., `"datasets/"`).
-   `load_dataset_path`: Path to a dataset file (relative to `dataset_dir`) to load. If empty or the file doesn't exist, new data will be collected. Otherwise, the specified dataset is loaded, skipping new data collection.
-   `dataset_filename`: Filename used when saving newly collected data (e.g., `"collected_data.pkl"`).
-   `frame_skipping`: (integer) Number of frames to skip after each action during data collection. For each action taken (by PPO or random agent), the environment steps `frame_skipping` additional times. During these skipped frames, new actions are sampled (randomly or by PPO based on intermediate states). Rewards are accumulated. This setting is internally disabled (treated as 0) during PPO data collection if `ppo_agent.action_repetition_k > 1`. See `docs/02_data_collection.md` for more details.

### PPO Agent for Data Collection (`ppo_agent`)
This block configures the PPO agent used if guided exploration is chosen for data collection.
-   `enabled`: Boolean (`true`/`false`). Master switch to enable/disable PPO-guided data collection. If `false`, random actions are used.
-   `learning_rate`: Learning rate for training the PPO agent.
-   `total_train_timesteps`: Number of timesteps to pre-train the PPO agent before it's used for data collection.
-   `n_steps`: PPO parameter for the number of steps to run for each environment per update.
-   `batch_size`: Minibatch size for PPO training.
-   `n_epochs`: Number of epochs when optimizing the PPO surrogate loss.
-   `policy_type`: The type of policy network for PPO (e.g., `"CnnPolicy"` for image-based environments, `"MlpPolicy"` for vector-based environments).
-   `action_repetition_k`: (integer, default: 1) If PPO is enabled and this value is > 1, the PPO agent's chosen action is repeated for `k` consecutive steps in the environment. Rewards are accumulated over these steps, and the state after `k` repetitions becomes the next state observed by the agent. If `action_repetition_k > 1`, the global `frame_skipping` parameter is internally set to 0 during PPO data collection to avoid conflicting behaviors. See `docs/02_data_collection.md` for detailed interaction.

### Model Loading Configuration
-   `model_dir`: Directory where trained models are stored (e.g., `"trained_models/"`).
-   `load_model_path`: Path to a specific model checkpoint file (relative to `model_dir`) to load before starting training. If empty, models are initialized from scratch.
-   `model_type_to_load`: Specifies which type of model `load_model_path` refers to (e.g., `"std_enc_dec"`, `"jepa"`). This ensures the checkpoint is loaded into the correct model architecture.

### Training Configuration
-   `num_epochs`: Number of training epochs to run for the main world models (Encoder-Decoder and/or JEPA). This is a general setting; specific components like `jepa_decoder_training` have their own `num_epochs`.
-   `batch_size`: Batch size for training these models.
-   `learning_rate`: Global learning rate, often used for the Standard Encoder-Decoder model and as a default.
-   `learning_rate_jepa`: Specific learning rate for the JEPA model, if different from the global `learning_rate`.
-   `num_workers`: Number of worker processes for the DataLoader.
-   `log_interval`: Frequency (in batches) for logging training progress.

### Encoder Configuration
This section defines the architecture of the primary encoder used by both Standard Encoder-Decoder and JEPA models.
-   `encoder_type`: Specifies the encoder architecture. Options: `"vit"` (Vision Transformer), `"cnn"` (Convolutional Neural Network), `"mlp"` (Multi-Layer Perceptron).
-   `patch_size`: An integer defining the size of patches (e.g., 8 for an 8x8 patch). Primarily used by the ViT encoder for input tokenization and as the default `decoder_patch_size` for image reconstruction components.
-   `encoder_params`: This dictionary contains nested sub-dictionaries, one for each `encoder_type` (`vit`, `cnn`, `mlp`), holding specific parameters for that architecture:
    -   `vit`: `depth` (number of Transformer blocks), `heads` (number of attention heads), `mlp_dim` (dimension of MLP within ViT blocks), `pool` (`'cls'` or `'mean'`), `dropout`, `emb_dropout`.
    -   `cnn`: `num_conv_layers`, `base_filters` (filters in the first layer), `kernel_size`, `stride`, `padding`, `activation_fn_str` (`'relu'` or `'gelu'`), `fc_hidden_dim` (optional FC layer dim).
    -   `mlp`: `num_hidden_layers`, `hidden_dim`, `activation_fn_str`.

### Shared Latent Dimension
-   `latent_dim`: The output dimensionality of the encoder. This is a crucial shared parameter as it defines the size of the representations `z_t`.

### Standard Encoder-Decoder Model
Parameters specific to the `StandardEncoderDecoder` model architecture:
-   `action_emb_dim`: Dimensionality of the embedded action vector.
-   `decoder_dim`: Internal dimensionality of the Transformer decoder.
-   `decoder_depth`: Number of layers in the Transformer decoder.
-   `decoder_heads`: Number of attention heads in the Transformer decoder.
-   `decoder_mlp_dim`: Dimension of the MLP within the Transformer decoder blocks.
-   `decoder_dropout`: Dropout rate for the decoder.
-   `decoder_patch_size`: Patch size used by the decoder to reconstruct the output image. If not specified or `null`, it defaults to the global `patch_size`.

### JEPA Model
Parameters specific to the `JEPA` model architecture:
-   `jepa_predictor_hidden_dim`: Hidden dimension(s) for the MLP predictor network in JEPA.
-   `ema_decay`: Decay rate for the Exponential Moving Average (EMA) update of the target encoder's weights.
-   `target_encoder_mode`: (string: `"default"`, `"vjepa2"`, `"none"`) Defines the operational strategy for JEPA's online and target encoders. This critical parameter affects how target embeddings are generated for the predictor and which encoder's outputs are used. Refer to `docs/04_jepa_model.md` for a detailed explanation of each mode's behavior regarding predictor inputs, prediction targets, EMA updates, and auxiliary loss inputs.
-   `auxiliary_loss`: Configures the auxiliary loss used to prevent representation collapse in JEPA's online encoder.
    -   `type`: Specifies the type of auxiliary loss. Options: `"vicreg"`, `"barlow_twins"`, `"dino"`.
    -   `weight`: A float controlling the contribution of this auxiliary loss to JEPA's total loss.
    -   `params`: Nested dictionaries for parameters specific to each auxiliary loss type:
        -   `vicreg`: `sim_coeff` (typically 0.0 for JEPA's `calculate_reg_terms`), `std_coeff` (variance term weight), `cov_coeff` (covariance term weight), `eps`.
        -   `barlow_twins`: `lambda_param` (weights redundancy reduction), `eps`, `scale_loss`.
        -   `dino`: `center_ema_decay` (for the DINO centering mechanism), `eps`. (Note: `out_dim` for DINOLoss is set programmatically from `latent_dim`).

### Reward Predictor MLPs (`reward_predictors`)
Configures MLPs for evaluating representations by predicting rewards. Separate configurations for MLPs using Encoder-Decoder representations and JEPA representations.
-   `encoder_decoder_reward_mlp` / `jepa_reward_mlp`:
    -   `enabled`: Boolean (`true`/`false`) to enable/disable training this reward predictor.
    -   `input_type`: (Primarily for `encoder_decoder_reward_mlp`) Often `"flatten"` if it uses the raw output of the decoder; for `jepa_reward_mlp`, the input is typically the `latent_dim` from JEPA's encoder. This is usually handled by `src/model_setup.py`.
    -   `hidden_dims`: A list of integers defining the number and size of hidden layers (e.g., `[128, 64]`).
    -   `activation`: Activation function string (e.g., `"relu"`).
    -   `use_batch_norm`: Boolean to enable/disable batch normalization.
    -   `learning_rate`: Learning rate for training this specific MLP.
    -   `log_interval`: Logging frequency for reward MLP training.

### JEPA State Decoder Training (`jepa_decoder_training`)
Configures the training of the `JEPAStateDecoder` model, which attempts to reconstruct images from JEPA's predicted embeddings.
-   `enabled`: Boolean (`true`/`false`) to enable/disable training of this decoder.
-   `num_epochs`: Number of epochs for training the `JEPAStateDecoder`.
-   `learning_rate`: Learning rate for the `JEPAStateDecoder`.
-   `checkpoint_path`: Filename for saving the best `JEPAStateDecoder` checkpoint (e.g., `"best_jepa_decoder.pth"`).
-   `validation_plot_dir`: Directory to save sample image reconstructions during validation (e.g., `"validation_plots/"`).
-   `early_stopping`: Sub-block for early stopping parameters for this decoder:
    -   `patience`: Number of epochs to wait for improvement.
    -   `delta`: Minimum change in monitored metric to qualify as improvement.
    -   `metric`: Metric to monitor (e.g., `"val_loss_jepa_decoder"`).

### Main Training Early Stopping (`early_stopping`)
Configures early stopping for the main world model training loops (Encoder-Decoder and JEPA).
-   `patience`: Number of epochs to wait.
-   `delta`: Minimum improvement change.
-   `metric_enc_dec`: Metric for Encoder-Decoder (e.g., `"val_loss_enc_dec"`).
-   `metric_jepa`: Metric for JEPA (e.g., `"val_total_loss_jepa"`).
-   `checkpoint_path_enc_dec`: Filename for Encoder-Decoder best model (e.g., `"best_encoder_decoder.pth"`).
-   `checkpoint_path_jepa`: Filename for JEPA best model (e.g., `"best_jepa.pth"`).
-   `validation_split`: Proportion of data from the main dataset to use for validation during world model training (e.g., `0.2` for 20%).

### Training Options (`training_options`)
-   `skip_std_enc_dec_training_if_loaded`: Boolean. If `true` and a Standard Encoder-Decoder model is successfully loaded via `load_model_path`, its training phase will be skipped.
-   `skip_jepa_training_if_loaded`: Boolean. If `true` and a JEPA model is successfully loaded, its training phase will be skipped.

## 3. Running the Training Script

Execute the main training script from the project's root directory:
```bash
python main.py
```

### Overview of `main.py` Execution:
The `main.py` script orchestrates the entire experimental pipeline:
1.  **Load Configuration:** Reads `config.yaml`.
2.  **Setup Device:** Detects and configures PyTorch to use CUDA, MPS (Apple Silicon), or CPU.
3.  **Create Directories:** Ensures `datasets/`, `trained_models/`, and `validation_plots/` (if JEPA decoder is used) exist.
4.  **Environment Details:** Fetches action space dimensions and type from the specified Gymnasium environment using `src/utils/env_utils.py`.
5.  **Prepare Dataloaders:** Managed by `src/data_handling.py`, this step either collects new trajectory data (using random actions or a PPO agent as per config) or loads a pre-existing dataset. It then prepares PyTorch `DataLoader` instances for training and validation.
6.  **Initialize Models:** Sets up the Standard Encoder-Decoder, JEPA, Reward Predictor MLPs, and JEPA State Decoder based on the configuration. This complex initialization is handled by `src/model_setup.py`.
7.  **Load Pre-trained Models:** If `load_model_path` is specified in the config, `main.py` attempts to load the weights into the designated `model_type_to_load`.
8.  **Initialize Losses and Optimizers:** `src/loss_setup.py` and `src/optimizer_setup.py` prepare the necessary loss functions (including auxiliary losses for JEPA) and optimizers for all trainable components.
9.  **Run Training Engine:** The core training logic orchestration resides in `src/training_engine.py`. This module calls specialized functions from the `src/training_loops/` directory (e.g., `epoch_loop.py` for main world models, `reward_predictor_loop.py` for reward MLPs, `jepa_decoder_loop.py` for the JEPA state decoder), which contain the detailed epoch-level training and validation logic for each component. The `training_engine.py` module:
    *   Conducts training epochs for the Standard Encoder-Decoder model (if not skipped and configured).
    *   Conducts training epochs for the JEPA model (if not skipped and configured), including EMA updates for JEPA's target encoder.
    *   Manages early stopping for both models based on validation performance and saves the best model checkpoints.
    *   If enabled, trains the `JEPAStateDecoder` (typically after JEPA training is complete or using frozen JEPA embeddings).
    *   If enabled, trains the `RewardPredictorMLP`s on frozen representations from the primary world models.
10. **Post-Training Model Loading:** After all training phases, `main.py` attempts to load the best saved checkpoints (identified during training) for all relevant models to ensure they are in their optimal state for any subsequent analysis. Models are then set to evaluation mode (`model.eval()`).

### Headless Operation (Servers)
If running on a server without a physical display (e.g., for environments like `CarRacing-v2` that typically render to screen), you might need to use a virtual display server like Xvfb. This tricks the environment into thinking a display is present.
```bash
Xvfb :1 -screen 0 1024x768x24 &  # Start Xvfb in the background
export DISPLAY=:1                 # Tell applications to use this virtual display
python main.py                    # Run your script
```

## 4. Interpreting Outputs

-   **Console Logs:** The script provides verbose output to the console during execution:
    -   The computing device being used (CUDA, MPS, CPU).
    -   Status of data collection or loading.
    -   Details of initialized models.
    -   Progress during training epochs (batch numbers, current loss values like total loss, prediction loss, auxiliary loss components for JEPA; reconstruction loss for Encoder-Decoder).
    -   Validation metrics (e.g., validation loss) reported at the end of each epoch or validation cycle.
    -   Notifications about early stopping and paths to saved model checkpoints.
-   **Saved Models:** Trained model checkpoints are stored in the directory specified by `model_dir` (default: `trained_models/`). The filenames for the best models are determined by `checkpoint_path_enc_dec`, `checkpoint_path_jepa`, and `checkpoint_path_jepa_decoder` in `config.yaml`.
-   **Datasets:** Collected trajectory datasets are saved in `dataset_dir` (default: `datasets/`). The filename is based on `dataset_filename` (for newly collected data) or `load_dataset_path` (if data was loaded).
-   **Validation Plots:** If `jepa_decoder_training.enabled` is `true` and `jepa_decoder_training.validation_plot_dir` is specified (e.g., `"validation_plots/"`), sample image reconstructions generated by the `JEPAStateDecoder` during its validation phases may be saved in this directory. This allows for qualitative assessment of JEPA's representation quality.

## 5. Extending the Codebase (Briefly)

This framework is designed with research extensibility in mind. Here are general pointers for adding new components:

-   **Adding New Model Architectures (World Models):**
    -   Develop your new model class (inheriting from `torch.nn.Module`) in a new Python file within `src/models/`.
    -   Integrate its instantiation into `src/model_setup.py`. This involves adding logic to parse its specific parameters from `config.yaml` and initialize the model.
    -   Update `src/training_engine.py` to include a dedicated training loop for your new model, handling its specific forward pass, loss computation, optimization steps, and checkpointing logic.
-   **Adding New Encoders:**
    -   Implement your encoder module (e.g., in `src/models/encoders/my_new_encoder.py`).
    -   In `src/model_setup.py`, add your new `encoder_type` to the conditional logic that instantiates encoders. Ensure it correctly receives its parameters from the `encoder_params` block in `config.yaml`.
-   **Adding New Loss Functions (especially Auxiliary Losses for JEPA):**
    -   Create your loss function class (inheriting from `torch.nn.Module`) in `src/losses/` (e.g., `my_custom_loss.py`).
    -   For JEPA auxiliary losses, it's good practice to include a `calculate_reg_terms(z)` method if it's intended to regularize a single set of embeddings, for consistency with existing losses like VICReg or BarlowTwins.
    -   Modify `src/loss_setup.py` to recognize a new `type` string in `config.yaml`'s `auxiliary_loss` section, allowing it to initialize your custom loss function and pass its specific parameters.
-   **Configuration is Key:** For any new component or major feature, ensure you add corresponding configuration options to `config.yaml`. This maintains the flexibility and reproducibility of experiments.

By following these guidelines, users can effectively utilize and contribute to this world model research framework, tailoring it to their specific research questions and hypotheses.The content for `docs/06_usage_guide.md` has been successfully written. It provides a comprehensive guide for users, covering prerequisites, setup, detailed configuration of `config.yaml`, instructions for running experiments, interpreting outputs, and pointers for extending the codebase.
