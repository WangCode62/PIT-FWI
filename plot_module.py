"""Plotting module for the PIT project.

This file groups the plotting and figure-saving functions for
models, acquisition geometry, and training results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_2d(data: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor-like data to a NumPy array for plotting."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def plot_survey_layout(vp_true, source_locations, receiver_locations):
    """Plot the velocity model together with source and receiver geometry."""
    vel_np = _to_numpy_2d(vp_true).T
    plt.figure(figsize=(10, 8))
    plt.imshow(vel_np, cmap="jet")
    source_x = source_locations[:, 0, 0].detach().cpu().numpy()
    source_z = source_locations[:, 0, 1].detach().cpu().numpy()
    plt.scatter(source_x, source_z, c="red", marker="^")
    receiver_x = receiver_locations[:, :, 0].detach().cpu().numpy()
    receiver_z = receiver_locations[:, :, 1].detach().cpu().numpy()
    plt.scatter(receiver_x.flatten(), receiver_z.flatten(), c="blue", marker="o", s=10)
    plt.xlabel("X (m)")
    plt.ylabel("Depth (m)")
    plt.title("Velocity Model with Sources and Receivers")
    plt.tight_layout()


def plot_true_and_initial(vp_true, vp_initial):
    """Plot the true model and the initial model for comparison."""
    true_np = _to_numpy_2d(vp_true).T
    init_np = _to_numpy_2d(vp_initial).T
    vmin = np.min(true_np)
    vmax = np.max(true_np)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(true_np, cmap="jet", aspect="auto")
    plt.colorbar()
    plt.title("True model")
    plt.subplot(2, 1, 2)
    plt.imshow(init_np, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Initial model")
    plt.tight_layout()


def save_training_snapshot(model, save_path: str | Path):
    """Save a snapshot of the velocity model at a training iteration."""
    model_np = _to_numpy_2d(model).T
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(model_np, cmap="jet", vmin=model_np.min(), vmax=model_np.max())
    points = ax.get_position().get_points()
    dy = points[1, 1] - points[0, 1]
    cax = fig.add_axes([0.91, points[0, 1], 0.02, dy])
    cax.yaxis.set_ticks_position("right")
    fig.colorbar(im, cax=cax, orientation="vertical", extend="neither", label="$V_P (m/s)$")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_final_comparison(vp_true, initial_model, learned_model, save_path: str | Path):
    """Save a three-panel comparison of true, initial, and learned models."""
    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.imshow(_to_numpy_2d(vp_true).T, cmap="jet", aspect="auto")
    plt.colorbar()
    plt.title("True model")
    plt.subplot(3, 1, 2)
    plt.imshow(_to_numpy_2d(initial_model).T, cmap="jet", aspect="auto")
    plt.colorbar()
    plt.title("Initial model")
    plt.subplot(3, 1, 3)
    plt.imshow(_to_numpy_2d(learned_model).T, cmap="jet", aspect="auto")
    plt.colorbar()
    plt.title("Learned model")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
