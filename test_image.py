"""
Analyze GaussianImage (2D Cholesky) checkpoint: distributions of Cholesky elements,
covariance (after LL^T), conic (Sigma^{-1}), scales (sqrt diagonal), and radii.

GaussianImage_Cholesky uses 2D Gaussians: L has 3 elements [l11, l21, l22],
covariance has 3 elements [Cxx, Cxy, Cyy], scales are [scale_x, scale_y].

Usage:
  python test_image.py --checkpoint path/to/gaussian_model.pth.tar [--H 1080 --W 1920] [--no-radii]
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Cholesky bound for GaussianImage_Cholesky (2D: l11, l21, l22)
CHOLESKY_BOUND_2D = torch.tensor([0.5, 0, 0.5], dtype=torch.float32)


def load_checkpoint(path: str, device="cpu"):
    """Load GaussianImage checkpoint (_xyz, _cholesky with 3 elements, optional _features_dc)."""
    ckpt = torch.load(path, map_location=device)
    if "_cholesky" not in ckpt or ckpt["_cholesky"].shape[1] != 3:
        raise KeyError("Expected GaussianImage (Cholesky) checkpoint with _cholesky (N, 3)")
    cholesky_raw = ckpt["_cholesky"]
    xyz_raw = ckpt["_xyz"]
    bound = CHOLESKY_BOUND_2D.to(cholesky_raw.device).view(1, 3)
    cholesky_elements = cholesky_raw + bound  # (N, 3)
    feature_dc = ckpt.get("_features_dc")  # (N, 3) or None
    return cholesky_elements, xyz_raw, feature_dc


def cholesky_2d_to_covariance(L_flat: torch.Tensor) -> torch.Tensor:
    """L_flat (N, 3) = [l11, l21, l22]. Returns cov (N, 3) = [Cxx, Cxy, Cyy]. Sigma = L L^T."""
    l11, l21, l22 = L_flat[:, 0], L_flat[:, 1], L_flat[:, 2]
    Cxx = l11 * l11
    Cxy = l11 * l21
    Cyy = l21 * l21 + l22 * l22
    return torch.stack([Cxx, Cxy, Cyy], dim=1)


def covariance_2d_to_conic_flat(cov_flat: torch.Tensor) -> torch.Tensor:
    """cov_flat (N, 3) = [Cxx, Cxy, Cyy]. Returns conic (N, 3) = inv(Sigma) upper triangle."""
    N = cov_flat.shape[0]
    device = cov_flat.device
    Cxx, Cxy, Cyy = cov_flat[:, 0], cov_flat[:, 1], cov_flat[:, 2]
    det = Cxx * Cyy - Cxy * Cxy
    det = torch.clamp(det, min=1e-12)
    invCxx = Cyy / det
    invCxy = -Cxy / det
    invCyy = Cxx / det
    return torch.stack([invCxx, invCxy, invCyy], dim=1)


def covariance_2d_diagonal_scales(cov_flat: torch.Tensor) -> torch.Tensor:
    """Return (scale_x, scale_y) = (sqrt(Cxx), sqrt(Cyy)) as (N, 2)."""
    Cxx, Cyy = cov_flat[:, 0], cov_flat[:, 2]
    return torch.stack([torch.sqrt(Cxx.clamp(min=1e-10)), torch.sqrt(Cyy.clamp(min=1e-10))], dim=1)


def compute_projected_radii_2d(xyz: torch.Tensor, cholesky_elements: torch.Tensor, H: int, W: int):
    """Run project_gaussians_2d to get radii."""
    from gsplat.project_gaussians_2d import project_gaussians_2d
    BLOCK_W, BLOCK_H = 16, 16
    tile_bounds = (
        (W + BLOCK_W - 1) // BLOCK_W,
        (H + BLOCK_H - 1) // BLOCK_H,
        1,
    )
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
        xyz.float().contiguous(),
        cholesky_elements.float().contiguous(),
        H, W, tile_bounds,
    )
    return radii


def plot_histograms_2d(
    cholesky_elements: np.ndarray,
    cov_elements: np.ndarray,
    scale_xy: np.ndarray,
    radii: Optional[np.ndarray],
    out_dir: Path,
    prefix: str = "",
    conic_elements: Optional[np.ndarray] = None,
    feature_dc: Optional[np.ndarray] = None,
):
    """Histograms for 2D GaussianImage: 3 Cholesky, 3 cov, 3 conic, 2 scales, radii, feature_dc."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chol_names = ["l11", "l21", "l22"]
    cov_names = ["Cxx", "Cxy", "Cyy"]
    conic_names = ["invCxx", "invCxy", "invCyy"]
    scale_names = ["scale_x", "scale_y"]
    feature_dc_names = ["feature_dc_R", "feature_dc_G", "feature_dc_B"]

    def _hist(data, title, xlabel, fname, bins=80):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(data.flatten(), bins=bins, density=True, alpha=0.8, color="steelblue", edgecolor="black", linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    for i, name in enumerate(chol_names):
        _hist(cholesky_elements[:, i], f"Cholesky (2D): {name}", name, f"{prefix}hist_cholesky_{name}.png")
    for i, name in enumerate(cov_names):
        _hist(cov_elements[:, i], f"Covariance (2D LL^T): {name}", name, f"{prefix}hist_cov_{name}.png")
    if conic_elements is not None:
        valid = np.isfinite(conic_elements).all(axis=1)
        if np.any(valid):
            for i, name in enumerate(conic_names):
                _hist(conic_elements[valid, i], f"Conic (2D Sigma^{{-1}}): {name}", name, f"{prefix}hist_conic_{name}.png")
    for i, name in enumerate(scale_names):
        _hist(scale_xy[:, i], f"Scale (2D sqrt diag): {name}", name, f"{prefix}hist_scale_{name}.png")

    if feature_dc is not None:
        for i, name in enumerate(feature_dc_names):
            _hist(feature_dc[:, i], f"Feature DC: {name}", name, f"{prefix}hist_feature_dc_{name}.png")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for i, name in enumerate(feature_dc_names):
            ax.hist(feature_dc[:, i], bins=80, density=True, alpha=0.5, label=name, histtype="step", linewidth=1.5)
        ax.set_title("Feature DC (RGB channels)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}hist_feature_dc_all.png", dpi=150)
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, name in enumerate(scale_names):
        ax.hist(scale_xy[:, i], bins=80, density=True, alpha=0.5, label=name, histtype="step", linewidth=1.5)
    ax.set_title("Covariance scales (2D, sqrt diagonal)")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}hist_scale_all.png", dpi=150)
    plt.close()

    if radii is not None:
        r = radii.flatten()
        r = r[r > 0]
        if len(r) > 0:
            _hist(r, "Projected radii (2D)", "Radius", f"{prefix}hist_radii.png", bins=min(80, int(r.max()) + 1))

    # Overview: Cholesky diag (l11, l22), scales, radii
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for j, lab in enumerate(["l11", "l22"]):
        axes[0].hist(cholesky_elements[:, j], bins=60, density=True, alpha=0.5, label=lab, histtype="step", linewidth=1.2)
    axes[0].set_title("Cholesky diag (l11, l22)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    for j, lab in enumerate(scale_names):
        axes[1].hist(scale_xy[:, j], bins=60, density=True, alpha=0.5, label=lab, histtype="step", linewidth=1.2)
    axes[1].set_title("Scales (sqrt diag)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    if radii is not None and radii.size > 0:
        r = radii.flatten()
        r = r[r > 0]
        if len(r) > 0:
            axes[2].hist(r, bins=min(60, int(r.max()) + 1), density=True, alpha=0.8, color="green", edgecolor="black")
    else:
        axes[2].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("Radii")
    axes[2].set_ylabel("Density")
    plt.suptitle("GaussianImage (2D) shape overview", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}overview.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze GaussianImage checkpoint distributions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to GaussianImage checkpoint (.pth.tar)")
    parser.add_argument("--out_dir", type=str, default="./analysis_out_image", help="Output directory")
    parser.add_argument("--H", type=int, default=1080)
    parser.add_argument("--W", type=int, default=1920)
    parser.add_argument("--no-radii", action="store_true", help="Skip projected radii")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    print(f"Loading checkpoint: {args.checkpoint}")
    cholesky_elements, xyz_raw, feature_dc_t = load_checkpoint(args.checkpoint, device)
    N = cholesky_elements.shape[0]
    print(f"GaussianImage (2D), num_gaussians: {N}")

    feature_dc_np = feature_dc_t.detach().cpu().numpy() if feature_dc_t is not None else None
    if feature_dc_np is not None:
        print(f"Feature DC shape: {feature_dc_np.shape}")

    cholesky_np = cholesky_elements.detach().cpu().numpy()
    cov_flat = cholesky_2d_to_covariance(cholesky_elements)
    cov_np = cov_flat.detach().cpu().numpy()
    scale_xy = covariance_2d_diagonal_scales(cov_flat)
    scale_np = scale_xy.detach().cpu().numpy()
    conic_flat = covariance_2d_to_conic_flat(cov_flat)
    conic_np = conic_flat.detach().cpu().numpy()

    radii_np = None
    if not args.no_radii:
        try:
            xyz = torch.tanh(xyz_raw).to(device)
            radii = compute_projected_radii_2d(xyz, cholesky_elements.to(device), args.H, args.W)
            radii_np = radii.detach().cpu().numpy()
            print(f"Radii: min={radii_np.min()}, max={radii_np.max()}")
        except Exception as e:
            print(f"Could not compute radii: {e}")

    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"H={args.H} W={args.W} (2D image)",
        f"Total Gaussians: {N}",
        "",
        "Cholesky (2D) [l11, l21, l22]:",
    ]
    for i, name in enumerate(chol_names := ["l11", "l21", "l22"]):
        col = cholesky_np[:, i]
        lines.append(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")
    lines.append("")
    lines.append("Covariance (2D) [Cxx, Cxy, Cyy]:")
    for i, name in enumerate(["Cxx", "Cxy", "Cyy"]):
        col = cov_np[:, i]
        lines.append(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")
    lines.append("")
    lines.append("Conic (2D) [invCxx, invCxy, invCyy]:")
    valid = np.isfinite(conic_np).all(axis=1)
    if np.any(valid):
        for i, name in enumerate(["invCxx", "invCxy", "invCyy"]):
            col = conic_np[valid, i]
            lines.append(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")
    lines.append("")
    lines.append("Scales (2D) [scale_x, scale_y]:")
    for i, name in enumerate(["scale_x", "scale_y"]):
        col = scale_np[:, i]
        lines.append(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")
    if radii_np is not None:
        r = radii_np[radii_np > 0]
        lines.append("")
        lines.append(f"Radii: min={radii_np.min()}, max={radii_np.max()}, mean(>0)={r.mean():.2f}")
    if feature_dc_np is not None:
        lines.append("")
        lines.append("Feature DC (2D) [R, G, B]:")
        for i, name in enumerate(["feature_dc_R", "feature_dc_G", "feature_dc_B"]):
            col = feature_dc_np[:, i]
            lines.append(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")

    for s in lines:
        print(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{Path(args.checkpoint).stem}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary: {summary_path}")

    prefix = Path(args.checkpoint).stem + "_"
    plot_histograms_2d(cholesky_np, cov_np, scale_np, radii_np, out_dir, prefix=prefix, conic_elements=conic_np, feature_dc=feature_dc_np)
    print(f"Histograms: {out_dir}")


if __name__ == "__main__":
    main()
