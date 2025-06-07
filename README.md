# RL World Models: Encoder-Decoder vs. JEPA with VICReg

This project implements and compares two world model architectures for learning from observations in Gymnasium environments:
1.  A standard Vision Transformer (ViT) based Encoder-Decoder.
2.  A Joint Embedding Predictive Architecture (JEPA) using ViTs, regularized with auxiliary losses like VICReg, Barlow Twins, or DINO.

The models are trained to predict future states or state embeddings based on current states and actions. The project now features robust dataset management, allowing for collection, saving, and reloading of experience datasets.

## Project Structure

```
.
├── .gitignore
├── README.md           # This file
├── config.yaml         # Configuration file for environment, models, and training
├── main.py             # Main training and execution script
├── requirements.txt    # Python dependencies
├── datasets/           # Directory for storing collected datasets (e.g., .pkl files)
├── src/                # Source code
│   ├── __init__.py
│   ├── config_utils.py   # Configuration loading utilities
│   ├── data_handling.py  # Dataloader preparation, dataset collection orchestration
│   ├── env_utils.py      # Environment utilities (e.g. wrapper for consistent returns)
│   ├── loss_setup.py     # Loss function initialization
│   ├── losses/           # Specific loss function implementations
│   │   ├── __init__.py
│   │   ├── barlow_twins.py
│   │   ├── dino.py
│   │   └── vicreg.py
│   ├── model_setup.py    # Model initialization (choosing encoder, main model etc.)
│   ├── models/           # Model architectures
│   │   ├── __init__.py
│   │   ├── cnn.py        # CNN Encoder
│   │   ├── encoder_decoder.py # Standard Encoder-Decoder model
│   │   ├── jepa.py       # JEPA model
│   │   ├── mlp.py        # MLP Encoder / Predictor components
│   │   └── vit.py        # Vision Transformer (ViT) Encoder
│   ├── optimizer_setup.py # Optimizer initialization
│   ├── training_engine.py # Core training loop logic, evaluation, checkpointing
│   └── utils/            # General utilities
│       ├── __init__.py
│       ├── data_utils.py   # Low-level data collection (ExperienceDataset, collect_random_episodes)
│       └── losses.py       # General loss utilities / older auxiliary loss logic
└── tests/              # Unit and integration tests
    ├── __init__.py
    ├── test_data_utils.py
    ├── test_losses.py
    ├── test_models.py
    └── test_reward_prediction.py # Example test file
```

## Setup

1.  **Clone the repository (if applicable)**
    ```bash
    # git clone <repository_url>
    # cd <project_directory>
    ```

2.  **Create a Python virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    This will install `gymnasium`, `torch`, `numpy`, `einops`, `pyyaml`, and other necessary packages.
    
    For certain Gymnasium environments, you might need additional installations:
    *   **Atari games** (e.g., `ALE/Pong-v5`):
        ```bash
        pip install gymnasium[atari] gymnasium[accept-rom-license]
        ```
    *   **Box2D environments** (e.g., `CarRacing-v2`):
        ```bash
        pip install gymnasium[box2d]
        # On some systems, you might need SWIG: sudo apt-get install swig
        ```
    *   For other environments, please refer to the Gymnasium documentation.

4.  **Datasets Directory:** The script will automatically create a `datasets/` directory if it doesn't exist when saving collected data.

## Configuration (`config.yaml`)

Edit the `config.yaml` file to set up the environment, models, and training parameters. Key sections include:

*   **Environment Configuration**:
    *   `environment_name`: Specify the Gymnasium environment ID (e.g., `"CarRacing-v2"`, `"ALE/Pong-v5"`). Ensure it provides pixel-based observations.
    *   `input_channels`: (e.g., 3 for RGB, 1 for grayscale).
    *   `image_size`: Target size for processed images (e.g., 64 for 64x64).
*   **Data Collection**:
    *   `num_episodes_data_collection`: Number of episodes for dataset creation if not loading.
    *   `max_steps_per_episode_data_collection`: Max steps per episode during data collection.
*   **Dataset Management (New!)**:
    *   `load_dataset` (boolean): Set to `true` to load an existing dataset, or `false` to collect new data.
    *   `dataset_name` (string): If `load_dataset` is true, specify the filename of the dataset to load from the `datasets/` directory (e.g., `"ALE_Pong-v5_200.pkl"`). Datasets are automatically saved with the format `<environment_name>_<num_episodes_data_collection>.pkl` (environment name slashes are replaced with underscores).
*   **Training Configuration**: `num_epochs`, `batch_size`, `learning_rate`, etc.
*   **Encoder Configuration**: `encoder_type` ("vit", "cnn", "mlp"), and specific parameters for each.
*   **Model-Specific Configurations**: Parameters for Standard Encoder-Decoder and JEPA models.
*   **Auxiliary Loss**: `type` ("vicreg", "barlow_twins", "dino") and its parameters.
*   **Early Stopping**: Configuration for early stopping based on validation metrics.

Example snippet for Dataset Management in `config.yaml`:
```yaml
# ... other configurations ...

# Data Collection
num_episodes_data_collection: 200
max_steps_per_episode_data_collection: 250

# Dataset Management
load_dataset: false       # Set to true to load a dataset
dataset_name: ""          # e.g., "ALE_Pong-v5_200.pkl" if load_dataset is true

# Training Configuration
num_epochs: 100
# ... other configurations ...
```

## Running the Training

Once the setup and configuration are complete, run the main training script:

```bash
python main.py
```

The script will perform the following steps:
1.  **Load Configuration**: Reads parameters from `config.yaml`.
2.  **Prepare Dataloaders**:
    *   Checks `config.load_dataset`.
    *   If `true` and `config.dataset_name` is valid, it attempts to load the specified dataset from the `datasets/` directory. It also verifies that the loaded dataset's environment matches the current configuration.
    *   If `false`, or if loading fails (e.g., file not found, environment mismatch), it proceeds to collect new data by interacting with the specified Gymnasium environment.
    *   Newly collected data is automatically saved to the `datasets/` directory (e.g., `datasets/ALE_Pong-v5_200.pkl`).
3.  **Initialize Models**: Sets up the chosen model architectures (Encoder-Decoder, JEPA), encoders (ViT, CNN, MLP), and loss functions based on the configuration.
4.  **Initialize Optimizers**: Sets up optimizers for the models.
5.  **Training Loop**: Runs the training for the specified number of epochs. This includes:
    *   Forward and backward passes.
    *   Loss calculation (e.g., reconstruction loss for Encoder-Decoder, prediction + auxiliary loss for JEPA).
    *   Optimizer steps.
    *   Logging training and validation metrics.
    *   Checkpointing: Saving the best performing models based on validation metrics.
6.  **Post-Training**: Loads the best saved model weights for potential further use or evaluation.

**Note on Display/Rendering:**
The data collection process and `main.py` are designed to run headlessly (`render_mode='rgb_array'` is typically used for environment creation when image data is needed). If you are running on a server without a display and encounter issues with environments that seem to require one, you might need to use a virtual framebuffer like Xvfb:
```bash
# Example for Linux:
# sudo apt-get install xvfb
# Xvfb :1 -screen 0 1024x768x24 &
# export DISPLAY=:1
```

## Implemented Architectures

### 1. Standard Encoder-Decoder
*   **Encoder**: A configurable encoder (ViT, CNN, or MLP) processes the input state image `s_t` into a latent representation.
*   **Decoder**: A Transformer Decoder takes the latent state and an embedded action `a_t` to predict the next state image `s_t+1` in pixel space.
*   **Loss**: Mean Squared Error (MSE) between the predicted `s_t+1` and the actual `s_t+1`.

### 2. JEPA (Joint Embedding Predictive Architecture)
*   **Encoders**:
    *   *Online Encoder* (ViT, CNN, or MLP): Processes the current state `s_t` and next state `s_t+1` into embeddings. This network is trained via backpropagation.
    *   *Target Encoder* (same architecture as Online Encoder): A momentum-based exponential moving average (EMA) of the online encoder. It processes `s_t` to provide a stable target representation for the predictor. Its weights are not updated by backpropagation.
*   **Predictor**: An MLP (or Transformer-based, configurable) takes the target-encoded `s_t` and an embedded action `a_t` to predict the online-encoded representation of `s_t+1`.
*   **Loss**:
    1.  *Prediction Loss (MSE)*: Between the predictor's output and the actual online-encoded `s_t+1`.
    2.  *Auxiliary Loss*: Applied to the outputs of the *online encoder* to prevent representational collapse and encourage informative embeddings. Configurable options include VICReg, Barlow Twins, and DINO.

The training script (`main.py`) allows selection and training of one of these primary architectures based on the `config.yaml` settings.
