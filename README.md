# RL World Models: Encoder-Decoder vs. JEPA

This project implements and compares two world model architectures for learning from visual observations in Gymnasium environments:
1.  A **Standard Encoder-Decoder** model.
2.  A **Joint Embedding Predictive Architecture (JEPA)**.

Both architectures are designed to learn representations of the environment and predict future states or state embeddings. The primary goal is to investigate the quality of learned representations and their utility for downstream tasks, particularly for research purposes.

## Full Documentation

For detailed information on the project, including in-depth explanations of the models, data collection, configuration, and evaluation methodologies, please refer to our comprehensive documentation:

**[View Full Documentation in `docs/`](docs/)**

We recommend starting with `docs/index.md` (if available) or browsing the individual Markdown files in the `docs/` directory for specific topics. The `docs/06_usage_guide.md` is particularly helpful for getting started.

## Project Structure

```
.
├── .gitignore
├── README.md           # This file (high-level overview)
├── config.yaml         # Central configuration file
├── main.py             # Main training and execution script
├── requirements.txt    # Python dependencies
├── docs/               # Detailed project documentation
│   ├── index.md
│   ├── 01_introduction.md
│   ├── 02_data_collection.md
│   ├── 03_encoder_decoder_model.md
│   ├── 04_jepa_model.md
│   ├── 05_representation_evaluation.md
│   └── 06_usage_guide.md
│   └── images/         # Directory for images used in documentation
├── datasets/           # Stores collected trajectory datasets
├── src/                # Source code (models, utilities, training logic)
│   ├── __init__.py
│   ├── data_handling.py
│   ├── env_utils.py
│   ├── loss_setup.py
│   ├── losses/           # VICReg, Barlow Twins, DINO implementations
│   ├── model_setup.py
│   ├── models/           # Encoder-Decoder, JEPA, ViT, CNN, MLP, etc.
│   ├── optimizer_setup.py
│   ├── training_engine.py
│   └── utils/            # General utilities
└── tests/              # Unit and integration tests
```

## Quick Start

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    For specific Gymnasium environments (e.g., Atari, Box2D), additional installations might be needed.

For detailed setup instructions, including environment-specific installations, please see **[`docs/06_usage_guide.md`](docs/06_usage_guide.md)**.

### Running the Training

1.  **Configure your experiment:** Edit `config.yaml` to define the environment, model architectures, data collection methods, and training parameters.
2.  **Run the main script:**
    ```bash
    python main.py
    ```

For a comprehensive explanation of all configuration options in `config.yaml` and details on the execution flow, refer to **[`docs/06_usage_guide.md`](docs/06_usage_guide.md)**.

## Implemented Architectures

This project provides implementations of the following world model architectures:

### 1. Standard Encoder-Decoder
*   **Encoder**: A configurable encoder (ViT, CNN, or MLP) processes the input state image `s_t` into a latent representation.
*   **Decoder**: A Transformer-based decoder takes the latent state and an embedded action `a_t` to predict the next state image `s_t+1` in pixel space.
*   **Loss**: Typically Mean Squared Error (MSE) between the predicted `s_t+1` and the actual `s_t+1`.

For a detailed description, see **[`docs/03_encoder_decoder_model.md`](docs/03_encoder_decoder_model.md)**.

### 2. JEPA (Joint Embedding Predictive Architecture)
*   **Encoders**: Utilizes an *Online Encoder* and an EMA-updated *Target Encoder* (ViT, CNN, or MLP) to process states into embeddings.
*   **Predictor**: An MLP (or Transformer) predicts the target encoder's embedding of the next state `s_{t+1}` based on the target-encoded current state `s_t` and action `a_t`.
*   **Loss**: Combines a primary *Prediction Loss* (MSE in embedding space) with an *Auxiliary Loss* (e.g., VICReg, Barlow Twins, DINO) applied to the online encoder's outputs to encourage informative representations.

For a detailed description, see **[`docs/04_jepa_model.md`](docs/04_jepa_model.md)**.

## Contributing

Contributions to this project are welcome. Please refer to the documentation and existing code structure for guidance. (Further details on contributing can be added here or in a separate `CONTRIBUTING.md` file).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (assuming a LICENSE file will be added).
