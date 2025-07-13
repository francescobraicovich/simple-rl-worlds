# RL World Models: Encoder-Decoder vs. JEPA

This project implements and compares two world model architectures for learning from visual observations in Gymnasium environments:
1.  A **Standard Encoder-Decoder** model.
2.  A **Joint Embedding Predictive Architecture (JEPA)**.

Both architectures are designed to learn representations of the environment and predict future states or state embeddings. The primary goal is to investigate the quality of learned representations and their utility for downstream tasks, particularly for research purposes.

## Core Concepts

### World Models
A "world model" is a component of an AI system that learns a model of its environment. Instead of learning a policy directly from high-dimensional inputs like pixels, a world model first learns to understand the dynamics of the environment: how it changes over time and what the consequences of certain actions are. This internal model can then be used for planning, imagination, or to provide a compact and informative representation for a separate policy-learning agent.

### Representation Learning
At the heart of this project is **representation learning**. When an agent receives an image from the environment, it's just a grid of pixel values. These values are not directly meaningful for decision-making. Representation learning is the process of automatically transforming this raw data into a lower-dimensional, more abstract, and more useful format—a **latent representation**. A good representation should capture the essential features of the state (e.g., player position, enemy locations) while discarding irrelevant noise (e.g., background textures, lighting variations).

## Documentation Guide

This documentation is structured to provide a comprehensive understanding of the project, from the high-level concepts to the fine-grained details of the model architectures and training workflows.

*   **[01_models.md](./01_models.md):** A deep dive into the architecture of each neural network component (Encoder, Predictor, Decoder, Reward Predictor). This is the core of the documentation.
*   **[02_training_scripts.md](./02_training_scripts.md):** A guide to the different training scripts, explaining how they orchestrate the training of the models.
*   **[03_evaluation.md](./03_evaluation.md):** Information on how to evaluate the trained models and interpret the results.
*   **[04_usage_guide.md](./04_usage_guide.md):** Practical, step-by-step instructions for running the complete training workflows.
