# adaoted from diffusers v0.35.1
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.outputs import BaseOutput


__all__ = ["FlowMatchEulerSDEDiscreteScheduler", "FlowMatchEulerSDEDiscreteSchedulerOutput"]

@dataclass
class FlowMatchEulerSDEDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    prev_sample_mean: Optional[torch.Tensor] = None
    std_dev_t: Optional[torch.Tensor] = None
    dt: Optional[torch.Tensor] = None


class FlowMatchEulerSDEDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """
    FlowMatch Euler SDE Discrete Scheduler.

    This class extends the FlowMatchEulerDiscreteScheduler to support SDE (Stochastic Differential Equations) sampling.
    Following https://arxiv.org/abs/2505.05470
    """

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        noise_level: float = 0.7,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerSDEDiscreteSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        assert per_token_timesteps is None
        sigma_idx = self.step_index

        sigma = self.sigmas[sigma_idx]
        sigma_next = self.sigmas[sigma_idx + 1]

        prev_sample_mean, std_dev_t, dt = self.compute_prev_sample_mean(
            model_output, sample, sigma, sigma_next, noise_level=noise_level
        )

        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample, prev_sample_mean, std_dev_t, dt)

        return FlowMatchEulerSDEDiscreteSchedulerOutput(
            prev_sample=prev_sample,
            prev_sample_mean=prev_sample_mean,
            std_dev_t=std_dev_t,
            dt=dt,
        )
    
    def compute_prev_sample_mean(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        noise_level: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_max = self.sigmas[1]
        dt = sigma_next - sigma

        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level

        # our sde
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
        return prev_sample_mean, std_dev_t, dt

