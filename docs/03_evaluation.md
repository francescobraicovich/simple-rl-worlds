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

The scripts in `src/scripts/representation_metrics/` are designed to provide rigorous, quantitative measures of the quality of the learned latent space. These metrics evaluate different aspects of representation quality and provide statistically sound comparisons between JEPA and Encoder-Decoder approaches.

### 3.1 Smoothness Analysis (`analyse_smoothness.py`)

**Purpose:** Evaluates how smoothly the latent representations evolve with respect to changes in the input observations. This metric assesses whether similar states in pixel space are mapped to similar representations in latent space.

**Methodology:**
- **Core Metric:** Examines the relationship between pixel-space distance `d_s = ||s_t - s_{t+1}||_2` and latent-space distance `d_z = ||φ(s_t) - φ(s_{t+1})||_2` for consecutive states.
- **Statistical Analysis:** Performs linear regression on the scatter plot of `d_z` vs `d_s`. The slope of the regression line serves as a quantitative smoothness metric—a lower, well-correlated slope indicates better smoothness.
- **Correlation Assessment:** Computes Pearson correlation coefficient (R²) to measure the strength of the linear relationship.

**Rationale:** A smooth representation space should exhibit a strong, positive correlation between pixel-space and latent-space distances. This property is crucial for:
- **Interpolation:** Enabling meaningful interpolation between representations
- **Generalization:** Ensuring that small perturbations don't cause dramatic changes in the latent space
- **Stability:** Providing robust representations for downstream tasks

**Output:** Generates `smoothness_comparison.png` containing scatter plots with regression lines and box plots of smoothness ratios, along with quantitative metrics (slope, R², correlation coefficient).

**Interpretation:** Lower regression slopes with high correlation indicate superior smoothness. This suggests the model captures the underlying data manifold structure more effectively.

---

### 3.2 Robustness Analysis (`analyse_robustness.py`)

**Purpose:** Assesses the stability of learned representations when input observations are corrupted with additive Gaussian noise. This evaluates how resilient the representations are to minor perturbations that might occur in real-world scenarios.

**Methodology:**
- **Core Metric:** Measures the L2 distance between clean and noisy representations: `E||φ(s̃) - φ(s)||_2`, where `s̃ = s + ε` and `ε ~ N(0, σ²I)`.
- **Multi-level Analysis:** Tests robustness across multiple noise levels (σ ∈ {0.01, 0.05, 0.1, 0.15, 0.2}) to understand degradation patterns.
- **Statistical Rigor:** Computes mean distances with standard errors across the validation dataset for reliable estimates.

**Rationale:** Robust representations are essential for:
- **Real-world Deployment:** Handling sensor noise, compression artifacts, and environmental variations
- **Generalization:** Maintaining performance when test conditions differ from training
- **Reliability:** Ensuring consistent behavior under uncertainty

**Output:** Produces `robustness_comparison.png` with robustness curves showing mean latent distance vs. noise level, plus box plots for distribution analysis at fixed noise levels.

**Interpretation:** Lower latent distances at all noise levels indicate superior robustness. Flatter curves suggest better tolerance to increasing noise levels.

---

### 3.3 Neighborhood Preservation Analysis (`analyse_neighborhood_preservation.py`)

**Purpose:** Evaluates how well the local neighborhood structure of the original state space is preserved in the learned latent space using established dimensionality reduction quality metrics.

**Methodology:**
- **Trustworthiness (T):** Measures the fraction of points in a latent neighborhood that are genuine neighbors in the original space. High values (near 1.0) indicate the model doesn't create false neighbors.
- **Continuity (C):** Measures the fraction of true neighbors that remain neighbors in the latent space. High values (near 1.0) indicate the model preserves local structure.
- **k-NN Analysis:** Computes metrics for multiple neighborhood sizes (k ∈ {5, 10, 20}) to assess preservation across different scales.

**Mathematical Foundation:** Based on Van der Maaten et al. (2008) formulations:
- **Trustworthiness:** `T(k) = 1 - (2/(n×k×(2n-3k-1))) × Σᵢ Σⱼ∈N̂ᵢᵏ\Nᵢᵏ (rᵢⱼ - k)`
- **Continuity:** `C(k) = 1 - (2/(n×k×(2n-3k-1))) × Σᵢ Σⱼ∈Nᵢᵏ\N̂ᵢᵏ (r̂ᵢⱼ - k)`

Where `Nᵢᵏ` and `N̂ᵢᵏ` are k-neighborhoods in original and latent spaces, respectively.

**Rationale:** Neighborhood preservation is fundamental for:
- **Manifold Learning:** Ensuring the latent space captures the underlying data manifold topology
- **Clustering Quality:** Maintaining meaningful groupings of similar states
- **Downstream Tasks:** Preserving semantic relationships for classification or control

**Output:** Generates `neighborhood_preservation.png` with bar charts comparing Trustworthiness and Continuity scores across different k-values.

**Interpretation:** Values closer to 1.0 for both metrics indicate better preservation of local structure. Consistent performance across different k-values suggests robust neighborhood preservation.

---

### 3.4 Manifold Dimension Analysis (`analyse_manifold_dimension.py`)

**Purpose:** Provides comprehensive analysis of the intrinsic dimensionality and complexity of learned representations, revealing how efficiently models utilize their representational capacity.

**Methodology:**
- **Participation Ratio (PR):** `PR = (Σᵢ λᵢ)² / Σᵢ λᵢ²`, where {λᵢ} are eigenvalues of the empirical covariance matrix. Ranges from 1 (1D manifold) to D (uniform utilization of all dimensions).
- **Two-NN Intrinsic Dimension:** Uses nearest-neighbor distance ratios (Facco et al., 2017) to estimate local intrinsic dimensionality: `ID = 1/(log(μ) - ψ(1))`, where μ is the mean distance ratio and ψ is the digamma function.
- **Eigenvalue Spectrum Analysis:** Visualizes how representational "power" is distributed across dimensions.

**Statistical Soundness:**
- **Bootstrap Confidence Intervals:** Provides error estimates for intrinsic dimension calculations
- **Covariance Analysis:** Uses established linear algebra techniques for participation ratio computation
- **Robust Sampling:** Handles large datasets through intelligent subsampling while maintaining statistical validity

**Rationale:** Understanding manifold properties reveals:
- **Representational Efficiency:** How well models compress information into lower-dimensional structures
- **Complexity Trade-offs:** Balance between expressiveness and generalization
- **Architectural Insights:** Different training objectives' effects on learned representations

**Output:** Comprehensive `manifold_dimension_analysis.png` with six panels showing participation ratios, intrinsic dimensions, dimension usage efficiency, eigenvalue spectra, and summary statistics.

**Interpretation:** 
- **Higher Participation Ratio:** More uniform dimension usage, potentially richer representations
- **Lower Intrinsic Dimension:** Better compression and potentially improved generalization
- **Efficiency Metrics:** Reveal how effectively models use their nominal dimensionality

---

### 3.5 Usage Guidelines

To execute these analyses:

1. **Prerequisites:** Ensure you have trained models saved in `weights/jepa/best_encoder.pth` and `weights/encoder_decoder/best_encoder.pth`

2. **Execution:** Run each script individually:
   ```bash
   python src/scripts/representation_metrics/analyse_smoothness.py
   python src/scripts/representation_metrics/analyse_robustness.py
   python src/scripts/representation_metrics/analyse_neighborhood_preservation.py
   python src/scripts/representation_metrics/analyse_manifold_dimension.py
   ```

3. **Output Location:** All plots are saved to `evaluation_plots/` subdirectories with publication-ready quality (300 DPI)

4. **Computational Considerations:** 
   - Smoothness and robustness analyses are computationally efficient
   - Neighborhood preservation scales O(n²) with dataset size
   - Manifold dimension analysis includes intelligent subsampling for large datasets

These metrics collectively provide a comprehensive evaluation framework for comparing representation quality across different training paradigms, enabling rigorous scientific assessment of model performance beyond simple reconstruction quality.
