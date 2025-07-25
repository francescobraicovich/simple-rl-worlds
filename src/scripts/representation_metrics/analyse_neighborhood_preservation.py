#!/usr/bin/env python3
"""
Neighborhood Preservation Analysis (JEPA vs. Encoder-Decoder)
  — uses sklearn.manifold.trustworthiness and vectorized continuity
"""

import sys
from pathlib import Path
import logging
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.utils import resample

# -------------------------------------------------------------------
# add project root to path so `src` imports work
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class ModelType:
    JEPA = "jepa"
    ENCODER_DECODER = "encoder_decoder"


def load_model(model_type: str, config_path: str, device: torch.device) -> torch.nn.Module:
    logging.info(f"Loading {model_type} encoder…")
    encoder = init_encoder(config_path).to(device)
    weights = project_root / "weights" / model_type / "best_encoder.pth"
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights at {weights}")
    encoder.load_state_dict(torch.load(weights, map_location=device))
    encoder.eval()
    return encoder


@torch.no_grad()
def get_latent_reps(encoder: torch.nn.Module, states: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    states: Tensor of shape (B, T, H, W)
    returns: (B, D) numpy array of L2‐normalized embeddings
    """
    states = states.to(device)
    out = encoder(states)           # now B×D
    reps = torch.nn.functional.normalize(out, p=2, dim=1)
    return reps.cpu().numpy()


def compute_continuity(X_true: np.ndarray, X_latent: np.ndarray, k: int) -> float:
    """
    Vectorized Continuity: fraction of true neighbors missing in latent.
    """
    n = X_true.shape[0]
    # get the k‐NN indices (0…n−1) in each space
    nbrs_t = NearestNeighbors(n_neighbors=k).fit(X_true)
    nbrs_l = NearestNeighbors(n_neighbors=k).fit(X_latent)
    idx_t = nbrs_t.kneighbors(return_distance=False)
    idx_l = nbrs_l.kneighbors(return_distance=False)

    # build boolean adjacency matrices
    A_t = np.zeros((n, n), bool)
    A_l = np.zeros((n, n), bool)
    A_t[np.arange(n)[:,None], idx_t] = True
    A_l[np.arange(n)[:,None], idx_l] = True

    # “extrusions” = in true but not in latent
    extr_mask = A_t & ~A_l

    # compute ranks in latent space
    d_lat = pairwise_distances(X_latent, metric="sqeuclidean")
    ranks_lat = np.argsort(np.argsort(d_lat, axis=1), axis=1)

    # penalty = sum(rank−k) over extrusions
    diffs = np.maximum(0, ranks_lat - k)
    penalty = diffs[extr_mask].sum()

    norm = 2.0 / (n * k * (2 * n - 3*k - 1))
    return 1.0 - norm * penalty


def run_bootstrap_analysis(
    encoder: torch.nn.Module,
    all_states: torch.Tensor,
    k_values: list[int],
    n_boot: int,
    sample_size: int,
    device: torch.device
) -> dict:
    """
    Returns {k: {"T_mean":…, "T_std":…, "C_mean":…, "C_std":…}, …}
    """
    stats = {k: {"T": [], "C": []} for k in k_values}

    for i in tqdm(range(n_boot), desc="Bootstraps"):
        # --- 1) sample raw states as a 5D tensor ---
        sample = resample(all_states, n_samples=sample_size, random_state=i)

        # --- 2) get latent reps (shape B×D) and flatten true states (B×(C·T·H·W)) ---
        X_lat = get_latent_reps(encoder, sample, device)
        X_true = sample.cpu().numpy().reshape(sample.size(0), -1)

        # --- 3) for each k compute T & C ---
        for k in k_values:
            T = trustworthiness(X_true, X_lat, n_neighbors=k)
            C = compute_continuity(X_true, X_lat, k)
            stats[k]["T"].append(T)
            stats[k]["C"].append(C)

    # aggregate
    results = {}
    for k in k_values:
        results[k] = {
            "T_mean": np.mean(stats[k]["T"]),
            "T_std":  np.std(stats[k]["T"]),
            "C_mean": np.mean(stats[k]["C"]),
            "C_std":  np.std(stats[k]["C"]),
        }
    return results


def create_plots(results: dict, k_values: list[int], out_png: Path):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6), sharey=True)
    x = np.arange(len(k_values))
    width = 0.35
    colors = {ModelType.JEPA: "royalblue", ModelType.ENCODER_DECODER: "coral"}

    for i, model in enumerate(results):
        Ts = [results[model][k]["T_mean"] for k in k_values]
        Ts_std = [results[model][k]["T_std"]  for k in k_values]
        Cs = [results[model][k]["C_mean"] for k in k_values]
        Cs_std = [results[model][k]["C_std"]  for k in k_values]

        ax1.bar(x + i*width, Ts,  width, yerr=Ts_std,  capsize=4, label=model.upper(), color=colors[model])
        ax2.bar(x + i*width, Cs,  width, yerr=Cs_std,  capsize=4, label=model.upper(), color=colors[model])

    ax1.set_title("Trustworthiness")
    ax1.set_xticks(x); ax1.set_xticklabels(k_values)
    ax1.set_ylim(0,1.05); ax1.set_xlabel("k"); ax1.set_ylabel("Score"); ax1.legend()

    ax2.set_title("Continuity")
    ax2.set_xticks(x); ax2.set_xticklabels(k_values)
    ax2.set_xlabel("k"); ax2.legend()

    plt.tight_layout()
    out_png.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_png, dpi=300)
    logging.info(f"Saved plot to {out_png}")


def save_summary_csv(results: dict, k_values: list[int], out_dir: Path) -> Path:
    rows = []
    for model in results:
        for k in k_values:
            r = results[model][k]
            rows.append({
                "Model": model.upper(),
                "k": k,
                "Trust_mean": r["T_mean"],
                "Trust_std":  r["T_std"],
                "Cont_mean":  r["C_mean"],
                "Cont_std":   r["C_std"],
            })
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "neighborhood_preservation_summary.csv"
    df.to_csv(path, index=False)
    logging.info(f"Saved CSV to {path}")
    return path


def main():
    config_path = str(project_root / "config.yaml")
    device = set_device()

    # load data
    data_pipe = DataLoadingPipeline(batch_size=128, config_path=config_path)
    _, val_loader = data_pipe.run_pipeline()
    all_states = torch.cat([b[0] for b in val_loader], dim=0)  # shape (N, C, T, H, W)

    k_values = [5,10,15]
    n_boot    = 10
    sample_sz = 200

    final = {}
    for model in (ModelType.JEPA, ModelType.ENCODER_DECODER):
        enc = load_model(model, config_path, device)
        final[model] = run_bootstrap_analysis(enc, all_states, k_values, n_boot, sample_sz, device)

    # print summary
    print("\n" + "="*60)
    print("k   MODEL               Trustworthiness (μ±σ)     Continuity (μ±σ)")
    print("-"*60)
    for k in k_values:
        for model in final:
            r = final[model][k]
            print(f"{k:<4}{model.upper():<20}"
                  f"{r['T_mean']:.4f}±{r['T_std']:.4f}        "
                  f"{r['C_mean']:.4f}±{r['C_std']:.4f}")
        print("-"*60)
    print("\n")

    # plots & CSV
    out_dir = project_root / "evaluation_plots" / "neighborhood_preservation"
    create_plots(final, k_values, out_dir / "neighborhood_preservation.png")
    save_summary_csv(final, k_values, out_dir)


if __name__ == "__main__":
    main()
