# Representation Quality Evaluation

This document provides a comprehensive guide to evaluating the quality of learned representations from different training paradigms, primarily comparing the **JEPA (Joint-Embedding Predictive Architecture)** and a standard **Encoder-Decoder** model. The evaluation is performed using a suite of rigorous, quantitative metrics designed to assess different aspects of the latent space.

---

## 1. Evaluation Philosophy

The goal of these evaluations is to move beyond simple reconstruction quality and to quantitatively measure the intrinsic properties of the learned representations. A good representation should be:

- **Smooth:** Small changes in the input should lead to small changes in the representation.
- **Robust:** The representation should be resilient to noise and perturbations.
- **Well-structured:** The local neighborhood structure of the data should be preserved in the latent space.
- **Efficient:** The model should learn a low-dimensional manifold that captures the essential features of the data.

## 2. Representation Quality Metrics

The following metrics are implemented in the `src/scripts/representation_metrics/` directory. Each script is designed to be run independently and will generate a publication-quality plot comparing the JEPA and Encoder-Decoder models.

### 2.1. Smoothness Analysis (`analyse_smoothness.py`)

**Purpose:**
This metric assesses the local Lipschitz continuity of the encoder. In simpler terms, it measures how well the encoder preserves the local geometry of the data. A smooth encoder will map nearby points in the input space to nearby points in the latent space.

**Methodology:**
- **Core Idea:** We compare the Euclidean distance between consecutive states in pixel space (`d_s = ||s_t - s_{t+1}||_2`) with the distance between their corresponding representations in the latent space (`d_z = ||φ(s_t) - φ(s_{t+1})||_2`).
- **Statistical Analysis:** We perform a linear regression on the scatter plot of `d_z` vs. `d_s`. The slope of the regression line is used as the primary metric for smoothness. A lower, well-correlated slope indicates better smoothness.
- **Correlation:** The Pearson correlation coefficient (R²) is computed to measure the strength of the linear relationship between the two distances.

**Why it Matters:**
A smooth representation is crucial for:
- **Generalization:** It helps the model generalize to unseen data by ensuring that small variations in the input do not lead to large, unpredictable changes in the representation.
- **Downstream Tasks:** A smooth latent space is easier for downstream models (e.g., a policy or value function) to learn from.

**Output:**
- `smoothness_comparison.png`: A scatter plot with regression lines and a box plot of the smoothness ratios.

**Interpretation:**
- **Lower Slope:** A lower slope on the regression plot indicates that the latent space is less sensitive to small changes in the input space, which is a sign of better smoothness.
- **Higher R²:** A higher R² value indicates a stronger linear relationship between the pixel and latent distances, which means the smoothness is consistent across the data distribution.

### 2.2. Robustness Analysis (`analyse_robustness.py`)

**Purpose:**
This metric evaluates the stability of the learned representations when the input observations are corrupted with noise. A robust encoder should produce similar representations for a clean image and its noisy version.

**Methodology:**
- **Core Idea:** We measure the L2 distance between the representation of a clean state and the representation of the same state with added Gaussian noise: `E||φ(s) - φ(s + ε)||_2`, where `ε ~ N(0, σ²I)`.
- **Multi-level Analysis:** The analysis is performed across a range of noise levels (σ) to understand how the robustness degrades as the noise increases.

**Why it Matters:**
Robust representations are essential for:
- **Real-world Applications:** Real-world data is often noisy. A robust model will be more reliable in such scenarios.
- **Adversarial Robustness:** While not a direct measure of adversarial robustness, it provides an indication of how well the model can handle small, unexpected perturbations.

**Output:**
- `robustness_comparison.png`: A line plot of the mean latent distance vs. the noise level, and a box plot comparing the distance distributions at a fixed noise level.

**Interpretation:**
- **Lower is Better:** A lower latent distance at all noise levels indicates better robustness.
- **Flatter Curve:** A flatter curve on the line plot suggests that the model is more tolerant to increasing levels of noise.

### 2.3. Neighborhood Preservation (`analyse_neighborhood_preservation.py`)

**Purpose:**
This metric assesses how well the local topology of the original data is preserved in the latent space. It answers the question: do points that are close in the input space remain close in the latent space?

**Methodology:**
- **Core Metrics:** We use two standard metrics from the dimensionality reduction literature:
    - **Trustworthiness (T):** Measures the extent to which the k-nearest neighbors of a point in the latent space are also its neighbors in the original space. A high trustworthiness score means the model does not create false neighbors.
    - **Continuity (C):** Measures the extent to which the k-nearest neighbors of a point in the original space are also its neighbors in the latent space. A high continuity score means the model does not break apart the local structure.
- **k-NN Analysis:** The metrics are computed for multiple neighborhood sizes (k) to assess preservation at different scales.

**Why it Matters:**
Preserving the local structure of the data is crucial for:
- **Clustering:** If the local structure is preserved, clustering algorithms will be more effective in the latent space.
- **Manifold Learning:** It indicates that the encoder has learned the underlying manifold of the data.

**Output:**
- `neighborhood_preservation.png`: Bar charts comparing the Trustworthiness and Continuity scores for different values of k.

**Interpretation:**
- **Closer to 1.0 is Better:** Scores closer to 1.0 for both metrics indicate better preservation of the local structure.

### 2.4. Manifold Dimension Analysis (`analyse_manifold_dimension.py`)

**Purpose:**
This analysis provides insights into the intrinsic dimensionality and complexity of the learned representations. It helps us understand how efficiently the models are using their representational capacity.

**Methodology:**
- **Participation Ratio (PR):** Measures how uniformly the variance is distributed across the dimensions of the latent space. It is calculated as `PR = (Σλᵢ)² / Σλᵢ²`, where `λᵢ` are the eigenvalues of the covariance matrix of the representations. A PR of 1 means the data lies on a 1D line, while a PR equal to the embedding dimension means all dimensions are used equally.
- **Two-NN Intrinsic Dimension:** A method to estimate the local intrinsic dimensionality of the data manifold using the ratio of distances to the first and second nearest neighbors.

**Why it Matters:**
- **Compression:** A lower intrinsic dimension suggests that the model has learned to compress the data into a more efficient representation.
- **Overfitting:** A very high intrinsic dimension might indicate that the model is overfitting and capturing noise in the data.

**Output:**
- `manifold_dimension_analysis.png`: A comprehensive plot with six panels showing the participation ratios, intrinsic dimensions, dimension usage efficiency, and eigenvalue spectra.

**Interpretation:**
- **Participation Ratio:** A higher PR is generally better, as it indicates that the model is using all the dimensions of the latent space effectively.
- **Intrinsic Dimension:** A lower intrinsic dimension is often desirable, as it suggests that the model has learned a more compact and efficient representation of the data.

### 2.5. Geometry Analysis (`analyse_geometry.py`)

**Purpose:**
This analysis evaluates the geometric properties of learned representations, providing insights into their structural organization and distribution in the latent space.

**Methodology:**
- **Uniformity on Hypersphere:** Measures how uniformly distributed the normalized representations are on the unit hypersphere using the formula `uniformity = log(mean(exp(-t * D²)))` where t=2.0. Lower values indicate more uniform distribution.
- **Silhouette Score:** Evaluates clustering quality and separation in the representation space using k-means clustering (when no labels are provided) or provided labels.
- **Clustering Quality:** Assesses how well representations cluster using k-means with configurable number of clusters. Returns normalized mutual information (NMI) if labels exist, plus inertia and cluster balance metrics.
- **k-NN Label Consistency:** Measures neighborhood coherence using k-nearest neighbors classification accuracy on a train/test split (only applicable when labels are available).

**Why it Matters:**
Geometric properties reveal important characteristics of the learned representations:
- **Uniform Distribution:** Well-distributed representations avoid clustering artifacts and may generalize better.
- **Good Clustering:** Clear separation between different states or behaviors indicates structured representations.
- **Neighborhood Coherence:** Consistent local neighborhoods suggest smooth, meaningful latent spaces.
- **Balanced Clusters:** Even distribution across clusters indicates the model isn't biased toward specific patterns.

**Output:**
- `geometry_analysis.png`: A comprehensive plot with six panels showing uniformity, silhouette score, clustering quality (NMI), clustering inertia, cluster balance, and k-NN accuracy, plus a summary comparison table.

**Interpretation:**
- **Uniformity:** Lower values indicate better distribution on the hypersphere
- **Silhouette Score:** Higher values (closer to 1.0) indicate better-separated clusters
- **NMI Score:** Higher values indicate better alignment between learned and true structure
- **Clustering Inertia:** Lower values indicate tighter, more cohesive clusters
- **Cluster Balance:** Lower values indicate more evenly distributed clusters
- **k-NN Accuracy:** Higher values indicate more consistent local neighborhoods

---

## 3. How to Run the Evaluations

1.  **Train the Models:** Make sure you have trained both the JEPA and Encoder-Decoder models. The scripts expect the trained encoder weights to be saved at:
    - `weights/jepa/best_encoder.pth`
    - `weights/encoder_decoder/best_encoder.pth`

2.  **Run the Scripts:** Execute each evaluation script from the root of the project:
    ```bash
    python src/scripts/representation_metrics/analyse_smoothness.py
    python src/scripts/representation_metrics/analyse_robustness.py
    python src/scripts/representation_metrics/analyse_neighborhood_preservation.py
    python src/scripts/representation_metrics/analyse_manifold_dimension.py
    python src/scripts/representation_metrics/analyse_geometry.py
    ```

    Or run all metrics at once:
    ```bash
    python src/scripts/run_representation_metrics.py
    ```

3.  **View the Results:** The output plots will be saved in the `evaluation_plots/` directory:
    - `evaluation_plots/smoothness_analysis/`
    - `evaluation_plots/robustness_analysis/`
    - `evaluation_plots/neighborhood_preservation/`
    - `evaluation_plots/manifold_dimension/`
    - `evaluation_plots/geometry_analysis/`