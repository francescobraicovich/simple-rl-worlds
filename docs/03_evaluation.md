# Evaluation

This document outlines how to evaluate the performance of the trained models. Proper evaluation is crucial to compare the effectiveness of the Encoder-Decoder and JEPA approaches.

---

## 1. Visual Evaluation (Reconstruction Quality)

For any model that includes a `Decoder` (`train_encoder_decoder.py`, `train_jepa_decoder.py`), a primary method of evaluation is to visually inspect the quality of the reconstructed images.

*   **How to:** Modify the validation loop in the respective training scripts to save the input frames, the ground-truth next frames, and the reconstructed next frames.
*   **What to look for:**
    *   **Clarity and Sharpness:** Are the reconstructions blurry or sharp?
    *   **Object Permanence:** Are key objects (like the player, enemies, projectiles) correctly generated?
    *   **Motion:** Does the reconstructed frame accurately reflect the motion that should have occurred?
    *   **JEPA vs. Encoder-Decoder:** A key comparison is the reconstruction quality from the `train_jepa_decoder.py` script versus the `train_encoder_decoder.py` script. If the JEPA-based decoder produces high-quality reconstructions, it suggests that the self-supervised latent space captured the necessary visual information without being explicitly trained on a reconstruction task.

---

## 2. Quantitative Evaluation (Losses)

*   **Training & Validation Loss:** The primary quantitative metric during training is the validation loss, which is logged to Weights & Biases. Lower validation loss is generally better.
*   **Reward Prediction MSE:** The Mean Squared Error of the `train_reward_predictor.py` and `train_dynamics_reward_predictor.py` scripts is a direct measure of how well the learned representations can be used to predict rewards.
    *   **Key Comparison:** Compare the reward MSE when using the ground-truth next state (`train_reward_predictor.py`) versus the predicted next state (`train_dynamics_reward_predictor.py`). A small difference indicates that the `Predictor` is generating highly accurate next-state representations.

---

## 3. Representation Quality Metrics

The scripts in `src/scripts/representation_metrics/` are designed to provide more rigorous, quantitative measures of the quality of the learned latent space.

*   **`analyse_temporal_coherence.py`:** Measures how smoothly the latent representations evolve over time. A good representation of a smoothly moving object should also be smooth in the latent space.
*   **`analyse_smoothness.py`:** Measures the smoothness of the latent space with respect to small changes in the input observations.
*   **`analyse_robustness.py`:** Evaluates how much the latent representation changes when small, random noise is added to the input images. A robust representation should be resilient to such noise.
*   **`analyse_neighborhood_preservation.py`:** Checks if frames that are visually similar in pixel space are also close together in the latent space.

To use these, you would typically load a pre-trained `Encoder` and run the analysis on a dataset of trajectories.
