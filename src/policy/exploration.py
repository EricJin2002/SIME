import torch
import numpy as np

def cads_linear_schedule(t, tau1, tau2):
    if t <= tau1:
            return 1.0
    if t>= tau2:
            return 0.0
    gamma = (tau2-t)/(tau2-tau1)
    return gamma

def add_noise(y, gamma, noise_scale):
    noise = torch.randn_like(y)
    return y + noise_scale * noise * (1 - gamma)

def apply_modal_level_exploration(global_cond, sampling_step, total_sampling_step, tau1, tau2, noise_scale):
    t = 1.0 - max(min(sampling_step / total_sampling_step, 1.0), 0.0) # Algorithms assumes we start at 1.0 and go to 0.0
    gamma = cads_linear_schedule(t, tau1, tau2)
    return add_noise(global_cond, gamma, noise_scale)