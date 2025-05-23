# World Models: Encoder-Decoder and JEPA with VICReg

This project implements two world model architectures for learning from observations in Gymnasium environments:
1.  A standard Vision Transformer (ViT) based Encoder-Decoder.
2.  A Joint Embedding Predictive Architecture (JEPA) using ViTs and regularized with VICReg.

The models are trained to predict future states or state embeddings based on current states and actions.

## Project Structure

```
.
├── config.yaml         # Configuration file for environment and hyperparameters
├── models/             # Model implementations
│   ├── __init__.py
│   ├── vit.py          # Vision Transformer encoder
│   ├── encoder_decoder.py # Standard Encoder-Decoder model
│   └── jepa.py         # JEPA model
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── data_utils.py   # Data collection from Gymnasium environments
│   └── losses.py       # VICReg loss implementation
├── requirements.txt    # Python dependencies
├── train.py            # Main training script
└── README.md           # This file
```

## Setup

1.  **Clone the repository (if applicable)**
    ```bash
    # git clone ...
    # cd project_directory
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
    This will install `gymnasium`, `torch`, `numpy`, `einops`, and `pyyaml`.
    
    For certain Gymnasium environments, you might need additional installations:
    *   **Atari games** (e.g., `ALE/Pong-v5`):
        ```bash
        pip install gymnasium[atari] gymnasium[accept-rom-license]
        ```
        You might need to accept the ROM license for Atari.
    *   **Box2D environments** (e.g., `CarRacing-v2`):
        ```bash
        pip install gymnasium[box2d]
        # On some systems, you might need SWIG: sudo apt-get install swig
        ```
    *   For other environments, please refer to the Gymnasium documentation.

## Configuration

Edit the `config.yaml` file to set up the training:

*   `environment_name`: Specify the Gymnasium environment ID (e.g., `"CarRacing-v2"`, `"ALE/Pong-v5"`).
    *   Ensure the chosen environment provides pixel-based observations.
*   `num_episodes_data_collection`: Number of episodes to run for collecting the initial dataset.
*   `max_steps_per_episode_data_collection`: Maximum steps per episode during data collection.
*   `image_size`: The size (height and width) to which observations will be resized (e.g., 64 for 64x64).
*   `patch_size`: Patch size for the Vision Transformer.
*   `input_channels`: Number of input channels for the image (e.g., 3 for RGB, 1 for grayscale).
*   Model hyperparameters: `latent_dim`, `decoder_dim`, `num_heads`, `num_encoder_layers`, `num_decoder_layers`, `mlp_dim`, `action_emb_dim`.
*   JEPA specific: `jepa_predictor_hidden_dim`, `ema_decay`.
*   VICReg hyperparameters: `vicreg_sim_coeff`, `vicreg_std_coeff`, `vicreg_cov_coeff`.
*   Training hyperparameters: `batch_size`, `learning_rate`, `num_epochs`.

Example snippet from `config.yaml`:
```yaml
environment_name: "CarRacing-v2" # Or "ALE/Pong-v5"
image_size: 64
patch_size: 8
input_channels: 3
latent_dim: 256
# ... other parameters
```

## Running the Training

Once the setup and configuration are complete, run the main training script:

```bash
python train.py
```

The script will:
1.  Load the configuration.
2.  Collect data by interacting with the specified Gymnasium environment.
3.  Initialize both the Standard Encoder-Decoder and JEPA models.
4.  Train both models, printing loss information to the console.

**Note on Display/Rendering:**
The data collection process and `train.py` are designed to run headlessly (`render_mode=None` or `render_mode='rgb_array'` for environment creation). If you are running on a server without a display, this should work fine. If you encounter issues with environments that require a display, you might need to use a virtual framebuffer like Xvfb:
```bash
# Example for Linux:
# sudo apt-get install xvfb
# Xvfb :1 -screen 0 1024x768x24 &
# export DISPLAY=:1
```

## Implemented Architectures

### 1. Standard Encoder-Decoder
*   **Encoder**: A Vision Transformer (ViT) processes the input state image `s_t` into a latent representation.
*   **Decoder**: A Transformer Decoder takes the latent state and an embedded action `a_t` to predict the next state image `s_t+1` in pixel space.
*   **Loss**: Mean Squared Error (MSE) between the predicted `s_t+1` and the actual `s_t+1`.

### 2. JEPA (Joint Embedding Predictive Architecture)
*   **Encoders**:
    *   *Online Encoder (ViT)*: Processes the current state `s_t` and next state `s_t+1` into embeddings. This network is trained.
    *   *Target Encoder (ViT)*: A momentum-based exponential moving average (EMA) of the online encoder. It processes `s_t` to provide a stable target representation for the predictor. Its weights are not updated by backpropagation.
*   **Predictor**: An MLP (or Transformer) takes the target-encoded `s_t` and an embedded action `a_t` to predict the online-encoded representation of `s_t+1`.
*   **Loss**:
    1.  *Prediction Loss (MSE)*: Between the predictor's output and the actual online-encoded `s_t+1`.
    2.  *VICReg Loss*: Applied to the outputs of the *online encoder* (for both `s_t` and `s_t+1` embeddings) to prevent representational collapse. This includes variance and covariance regularization terms.

The training script trains both models concurrently on the same dataset of experiences.
