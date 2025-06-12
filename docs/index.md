# Documentation Overview

Welcome to the comprehensive documentation for the RL World Models project. This collection of documents aims to provide a thorough understanding of the project's theoretical underpinnings, implementation details, usage instructions, and evaluation methodologies.

This documentation is structured to guide you through the various aspects of the project, from a high-level introduction to specific guides for running and extending the codebase.

## Table of Contents

Below is a list of documents that detail different components and aspects of this research project. Please refer to these for in-depth information:

*   **[01 Introduction](./01_introduction.md)**
    *   Provides the research context, introduces the core architectural approaches (Encoder-Decoder vs. JEPA), and outlines the project's aims and objectives.

*   **[02 Data Collection](./02_data_collection.md)**
    *   Details the importance of datasets, the data collection pipeline, methods for data generation (PPO-guided exploration vs. random actions), and dataset management including configuration.

*   **[03 Encoder-Decoder Model](./03_encoder_decoder_model.md)**
    *   Describes the architecture of the Standard Encoder-Decoder model, including its encoder, action embedding, Transformer-based decoder, loss function, and a discussion of its advantages and disadvantages.

*   **[04 JEPA Model](./04_jepa_model.md)**
    *   Provides a comprehensive overview of the Joint Embedding Predictive Architecture (JEPA), its core components (online encoder, target encoder, predictor), prediction loss in embedding space, and the crucial role and implementation of auxiliary losses (VICReg, Barlow Twins, DINO).

*   **[05 Representation Evaluation](./05_representation_evaluation.md)**
    *   Explains the methodologies used to evaluate the quality and utility of the learned representations from both model types, focusing on reward prediction tasks and (for JEPA) next-state image reconstruction.

*   **[06 Usage Guide](./06_usage_guide.md)**
    *   Offers a practical step-by-step guide for users to set up the project, install dependencies, understand and modify the `config.yaml` file, run training experiments, and interpret the outputs. It also includes brief pointers for extending the codebase.

We recommend navigating these documents sequentially for a complete understanding, or directly accessing the relevant section based on your specific interest.
