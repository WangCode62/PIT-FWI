"""Forward-modeling module for the PIT project.

This file defines the survey configuration, acquisition geometry,
and the Deepwave-based forward-modeling wrapper.
"""

from dataclasses import dataclass

import deepwave
import torch
import torch.nn as nn
from deepwave import scalar


@dataclass
class SurveyConfig:
    """Survey and time-sampling configuration."""
    ny: int = 201
    nx: int = 225
    dx: float = 10
    n_shots: int = 21
    n_sources_per_shot: int = 1
    d_source: int = 10
    first_source: int = 0
    source_depth: int = 0
    n_receivers_per_shot: int = 200
    dz_receiver: int = 1
    receiver_depth_start: int = 5
    well_x: int = 100
    freq: float = 25.0
    dt: float = 0.001
    nt: int = 2000

    @property
    def peak_time(self) -> float:
        return 1.0 / self.freq


class PhysicsDeepwave(nn.Module):
    """A light wrapper around Deepwave scalar forward modeling."""
    def __init__(self, dx, dt, pml_freq, src, src_loc, rec_loc):
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.src = src
        self.src_loc = src_loc
        self.rec_loc = rec_loc
        self.pml_freq = pml_freq

    def forward(self, vp: torch.Tensor) -> torch.Tensor:
        """Take a 2D velocity model and return formatted synthetic records."""
        out = scalar(
            vp,
            self.dx,
            self.dt,
            source_amplitudes=self.src,
            source_locations=self.src_loc,
            receiver_locations=self.rec_loc,
            accuracy=8,
            pml_freq=self.pml_freq,
            pml_width=35,
        )
        vx = out[-1]
        return vx.permute(0, 2, 1).unsqueeze(0)


def build_survey_geometry(config: SurveyConfig, device: torch.device):
    """Build source locations, receiver locations, and the source wavelet."""
    source_locations = torch.zeros(
        config.n_shots,
        config.n_sources_per_shot,
        2,
        dtype=torch.long,
        device=device,
    )
    # Place sources laterally along the surface.
    source_locations[..., 1] = config.source_depth
    source_locations[:, 0, 0] = torch.arange(config.n_shots, device=device) * config.d_source + config.first_source

    receiver_locations = torch.zeros(
        config.n_shots,
        config.n_receivers_per_shot,
        2,
        dtype=torch.long,
        device=device,
    )
    # Place receivers on a vertical line next to the well.
    receiver_locations[..., 0] = config.well_x
    receiver_locations[:, :, 1] = (
        torch.arange(config.n_receivers_per_shot, device=device) * config.dz_receiver + config.receiver_depth_start
    ).repeat(config.n_shots, 1)

    # Use a Ricker wavelet as the source signature.
    source_amplitudes = deepwave.wavelets.ricker(
        config.freq,
        config.nt,
        config.dt,
        config.peak_time,
    ).repeat(config.n_shots, config.n_sources_per_shot, 1).to(device)

    return source_locations, receiver_locations, source_amplitudes


def simulate_observed_data(
    vp_true: torch.Tensor,
    config: SurveyConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate observed data and return the matching acquisition geometry."""
    source_locations, receiver_locations, source_amplitudes = build_survey_geometry(config, device)
    physics = PhysicsDeepwave(
        dx=config.dx,
        dt=config.dt,
        pml_freq=config.freq,
        src=source_amplitudes,
        src_loc=source_locations,
        rec_loc=receiver_locations,
    )
    d_obs = physics(vp_true.to(device))
    return d_obs, source_locations, receiver_locations, source_amplitudes

# Keep the original notebook class name.
Physics_deepwave = PhysicsDeepwave
