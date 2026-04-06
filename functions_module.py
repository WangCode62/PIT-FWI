"""Utility functions for the PIT project.

This file contains training helpers, data-processing utilities,
small reusable tools, and run-parameter saving logic.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from scipy.signal import butter, filtfilt


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds to improve experiment reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def downsample(img: torch.Tensor, aim_height: int, aim_width: int) -> torch.Tensor:
    """Downsample a 3D tensor to the target size with nearest-neighbor sampling."""
    channel, height, width = img.shape
    out = torch.zeros((channel, aim_height, aim_width), dtype=img.dtype, device=img.device)
    transform_h = aim_height / height
    transform_w = aim_width / width
    for i in range(aim_height):
        for j in range(aim_width):
            x = int(i / transform_h)
            y = int(j / transform_w)
            out[:, i, j] = img[:, x, y]
    return out


def add_awgn_with_snr(data: torch.Tensor, snr: float, device: str | torch.device = "cuda") -> torch.Tensor:
    """Add white Gaussian noise to seismic data at a given SNR."""
    data = data.float()
    data_power = torch.norm(data - torch.mean(data)) ** 2 / data.numel()
    noise_variance = data_power / torch.pow(torch.tensor(10.0, device=data.device), snr / 10)
    noise = torch.randn_like(data)
    noise = noise - torch.mean(noise)
    noise = (torch.sqrt(noise_variance) / torch.std(noise)) * noise
    return (data + noise).to(device)


def get_dir(directory: str | os.PathLike[str]) -> str:
    """Create a directory if needed and return it as a string."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return str(directory)


def highpass_filter(
    data: torch.Tensor,
    cutoff: float,
    fs: float,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Apply a high-pass filter to seismic records and keep tensor I/O."""
    data_np = data.squeeze().detach().cpu().numpy()
    shots, nt, nx = data_np.shape
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype="high", analog=False)
    filtered_data = np.zeros((shots, nt, nx), dtype=np.float32)
    for i in range(shots):
        filtered_data[i] = filtfilt(b, a, data_np[i], axis=0)
    return torch.from_numpy(filtered_data).to(device).unsqueeze(0).float()


def clear_dir(directory: str) -> None:
    """Clear a directory safely after basic path validation."""
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a directory")
    if directory in ["..", ".", "", "/", "./", "../", "*"]:
        raise ValueError("Refusing to clear an unsafe directory")
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def weights_init(module: nn.Module, leak_value: float) -> None:
    """Initialize convolution and linear layers with Xavier initialization."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if module.weight is not None:
            init.xavier_normal_(module.weight, leak_value)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)


def load_velocity_model(path: str | os.PathLike[str], shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """Load a velocity model from a binary file and reshape it to the target grid."""
    data = np.fromfile(path, dtype=np.float32).reshape(shape)
    return torch.from_numpy(data).to(device)


def train_engine(
    autoencoder: nn.Module,
    physics: nn.Module,
    criteria: nn.Module,
    optim_autoencoder: torch.optim.Optimizer,
    vp_initial: torch.Tensor,
    vp_scale: float,
    d_obs: torch.Tensor,
    batch: int,
    mini_batches: int,
) -> tuple[float, torch.Tensor]:
    """Run one first-stage mini-batch update.

    The network predicts a velocity-model update, the physics operator
    generates synthetic data, and the loss is backpropagated.
    """
    earth_model = autoencoder(d_obs).squeeze()
    if earth_model.shape != vp_initial.shape:
        raise ValueError(
            f"shape mismatch: vp_initial={tuple(vp_initial.shape)}, earth_model={tuple(earth_model.shape)}"
        )
    # Interpret the network output as an update relative to the initial model.
    vp = (earth_model * vp_scale + vp_initial).requires_grad_(True)
    m = vp.to(earth_model.device)
    # Run forward modeling with the current velocity model.
    taux_est = physics(m)
    d_obs_filtered = d_obs[:, batch::mini_batches]
    loss_data = criteria(taux_est, d_obs_filtered)
    loss_data.backward()
    optim_autoencoder.step()
    return loss_data.item(), m


def adjest_engine(
    physics: nn.Module,
    netD: nn.Module,
    d_model: torch.nn.Parameter,
    d_obs: torch.Tensor,
    batch: int,
    mini_batches: int,
    optim_vel: torch.optim.Optimizer,
    optim_net: torch.optim.Optimizer,
) -> float:
    """Run one second-stage joint FWI/Siamese mini-batch update."""
    optim_vel.zero_grad()
    netD.train()
    netD.zero_grad()
    d_fake = physics(d_model)
    d_real = d_obs[:, batch::mini_batches]
    _, _, o1, o2 = netD(d_fake, d_real)
    # Measure the gap between synthetic and observed data in feature space.
    loss = F.pairwise_distance(o1, o2, keepdim=True).mean()
    loss.backward()
    optim_net.step()
    # Limit gradients and model values to reduce numerical instability.
    torch.nn.utils.clip_grad_value_(d_model, 1e3)
    optim_vel.step()
    d_model.data = torch.clamp(d_model.data, min=1e-12)
    return loss.item()


def train_deepwave(
    Physics: type[nn.Module],
    autoencoder: nn.Module,
    d_obs: torch.Tensor,
    optim_autoencoder: torch.optim.Optimizer,
    vp_initial: torch.Tensor,
    vp_scale: float,
    criteria: nn.Module,
    mini_batches: int,
    src_loc: torch.Tensor,
    rec_loc: torch.Tensor,
    src: torch.Tensor,
    dx: float,
    dt: float,
    pml_freq: float,
) -> tuple[float, torch.Tensor, nn.Module]:
    """Run first-stage training in shot-based mini-batches and return the mean loss."""
    loss_data_minibatch = []
    last_model = None
    for batch in range(mini_batches):
        optim_autoencoder.zero_grad()
        # Each mini-batch uses only a subset of shots and receivers.
        physics = Physics(
            dx=dx,
            dt=dt,
            src=src[batch::mini_batches],
            pml_freq=pml_freq,
            src_loc=src_loc[batch::mini_batches],
            rec_loc=rec_loc[batch::mini_batches],
        )
        loss_data, last_model = train_engine(
            autoencoder=autoencoder,
            physics=physics,
            criteria=criteria,
            optim_autoencoder=optim_autoencoder,
            vp_initial=vp_initial,
            vp_scale=vp_scale,
            d_obs=d_obs,
            batch=batch,
            mini_batches=mini_batches,
        )
        loss_data_minibatch.append(loss_data)
    return float(np.mean(loss_data_minibatch)), last_model, autoencoder


def adjest_deepwave(
    Physics: type[nn.Module],
    netD: nn.Module,
    d_model: torch.nn.Parameter,
    d_obs: torch.Tensor,
    mini_batches: int,
    optim_vel: torch.optim.Optimizer,
    optim_net: torch.optim.Optimizer,
    src_loc: torch.Tensor,
    rec_loc: torch.Tensor,
    src: torch.Tensor,
    dx: float,
    dt: float,
    pml_freq: float,
) -> tuple[float, torch.optim.Optimizer, torch.optim.Optimizer]:
    """Run second-stage training in shot-based mini-batches and return the mean loss."""
    loss_data_minibatch = []
    for batch in range(mini_batches):
        physics = Physics(
            dx=dx,
            dt=dt,
            src=src[batch::mini_batches],
            pml_freq=pml_freq,
            src_loc=src_loc[batch::mini_batches],
            rec_loc=rec_loc[batch::mini_batches],
        )
        loss_data = adjest_engine(
            physics=physics,
            netD=netD,
            d_model=d_model,
            d_obs=d_obs,
            batch=batch,
            mini_batches=mini_batches,
            optim_vel=optim_vel,
            optim_net=optim_net,
        )
        loss_data_minibatch.append(loss_data)
    return float(np.mean(loss_data_minibatch)), optim_vel, optim_net


def save_run_parameters(path: str | os.PathLike[str], params: Dict[str, Dict[str, Any]]) -> None:
    """Write run parameters, losses, and timing information to a text file."""
    with open(path, "w", encoding="utf-8") as f:
        for category, items in params.items():
            f.write(f"{category}:\n")
            for key, value in items.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

# Keep the legacy notebook name to avoid breaking migrated code.
Downsample = downsample
