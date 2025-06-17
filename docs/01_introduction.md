# 01 Introduction

## Research Context: World Models for Reinforcement Learning

Reinforcement Learning (RL) has emerged as a powerful paradigm for enabling agents to learn optimal behaviors through interaction with an environment. A key challenge in RL, particularly in complex and high-dimensional environments, is the development of effective environment models, often referred to as "world models." World models aim to capture the underlying dynamics and structure of the environment, allowing agents to simulate future outcomes, plan actions, and learn more efficiently. By learning a compressed representation of the environment's state and transitions, world models can significantly improve sample efficiency and enable more sophisticated planning capabilities compared to model-free RL approaches. This project investigates methods for constructing such world models, focusing on visual input streams, which are prevalent in many real-world applications.

## Core Approaches: Encoder-Decoder vs. Joint Embedding Predictive Architectures (JEPA)

This research focuses on comparing two prominent architectural paradigms for learning world models from pixel data:

1.  **Direct Pixel-Space Prediction (Encoder-Decoder Models):** This approach typically involves an encoder network that maps high-dimensional pixel inputs to a lower-dimensional latent representation. A decoder network then attempts to reconstruct the original pixel input, or predict future pixel inputs, from this latent representation. The objective function is often based on minimizing the reconstruction or prediction error in pixel space. While intuitive, this approach can be computationally intensive and may expend representational capacity on irrelevant details in the pixel space.

2.  **Predictive Coding in Embedding Space (Joint Embedding Predictive Architectures - JEPA):** JEPA models, in contrast, operate by making predictions in an abstract embedding space. An encoder maps the input to an embedding. A predictor model then takes a representation of one part of the input (e.g., a masked portion of an image or a past state) and attempts to predict the embedding of another part of the input (e.g., the unmasked portion or a future state). The key distinction is that the prediction target is the *embedding* of the target, not the raw pixels. This encourages the model to learn representations that capture abstract, predictable features rather than focusing on pixel-level details, potentially leading to more semantically meaningful and efficient representations.

## Key Architectural Features and Data Handling

To facilitate robust research and experimentation, this project incorporates a **modular training loop design**, primarily managed within the `src/training_loops/` directory. This structure allows for flexible swapping of model components, loss functions, and training procedures, streamlining the comparative analysis of different world model architectures.

Effective world model learning is also critically dependent on the quality and characteristics of the training data. This project supports multiple **data collection strategies**:
*   **Guided Data Collection**: Utilizes a Proximal Policy Optimization (PPO) agent (`ppo_agent.enabled: true` in `config.yaml`) to gather trajectories that are potentially more structured or goal-oriented.
*   **Random Data Collection**: Employs random actions to explore the environment broadly.
The project also integrates common techniques like **action repetition** and **frame skipping** (configurable in `config.yaml`) to influence the temporal characteristics and volume of the collected data. Detailed data collection procedures are further elaborated in `docs/02_data_collection.md`.

## Project Aim and Objectives

The primary aim of this project is to conduct a comparative analysis of the representations learned by Encoder-Decoder and JEPA-based world models. Specifically, we seek to:

*   Investigate the qualitative and quantitative differences in the learned latent representations.
*   Evaluate the utility of these representations for various downstream tasks relevant to reinforcement learning, such as policy learning, planning, and transfer learning.
*   Assess the computational efficiency and scalability of each approach.
*   Provide insights and recommendations regarding the suitability of these architectures for building effective world models, particularly within the context of research intended for publication at leading AI conferences.

This investigation will contribute to a deeper understanding of how different architectural choices influence the nature of learned world models and their effectiveness in supporting intelligent agent behavior.

## Getting Started: Key Configurations

To quickly get started with experiments, familiarize yourself with these crucial parameters in the `config.yaml` file:

*   `environment_name`: Specifies the Gymnasium environment to be used (e.g., `"ALE/Pong-v5"`, `"CartPole-v1"`).
*   `model_type_to_load`: Set to `"std_enc_dec"`, `"jepa"`, or `null`. If a model type is specified, the system will attempt to load a pre-trained model of that type (if found) before starting new training or evaluation.
*   `num_epochs`: Defines the total number of training epochs for the world models. This is a primary control for the overall training duration.
*   `training_options.skip_std_enc_dec_training_if_loaded`: (boolean) If `true` and a standard encoder-decoder model is loaded, its training phase will be skipped. Useful for focusing on JEPA training or evaluation.
*   `training_options.skip_jepa_training_if_loaded`: (boolean) If `true` and a JEPA model is loaded, its training phase will be skipped. Useful for focusing on encoder-decoder training or evaluation.
*   `ppo_agent.enabled`: (boolean) Set to `true` to use the PPO agent for data collection. If `false`, data will be collected using random actions.
*   `wandb.enabled`: (boolean) Set to `true` to enable logging of experiment metrics, configurations, and outputs to Weights & Biases. Set to `false` to disable.

These parameters provide a starting point for customizing data collection, model training, and experiment tracking. For a comprehensive list and detailed explanations of all configuration options, please refer to `docs/06_usage_guide.md` and the comments within `config.yaml` itself.
