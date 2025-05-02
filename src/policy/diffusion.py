import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from einops import reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policy.diffusion_modules.conditional_unet1d import ConditionalUnet1D
from policy.diffusion_modules.mask_generator import LowdimMaskGenerator
from policy.exploration import apply_modal_level_exploration

class CustomDDIMScheduler(DDIMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # not changed yet        
    def _get_variance(self, timestep, prev_timestep):
        # print("Using custom variance")
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
        

class DiffusionUNetPolicy(nn.Module):
    def __init__(self,
            action_dim,
            horizon, 
            n_obs_steps,
            obs_feature_dim,
            num_inference_steps=20,
            diffusion_step_embed_dim=256,
            down_dims=(256,512),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            noise_scheduler_type="ddim",

            # parameters passed to step
            **kwargs):
        super().__init__()

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # create noise scheduler
        if noise_scheduler_type in ["ddim", "DDIM"]:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon"
            )
        elif noise_scheduler_type in ["custom_ddim", "CustomDDIM"]:
            self.noise_scheduler = CustomDDIMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon"
            )
        elif noise_scheduler_type in ["ddpm", "DDPM"]:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                variance_type="fixed_small",
                prediction_type="epsilon"
            )
        else:
            raise ValueError(f"Unsupported noise scheduler type {noise_scheduler_type}")

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        self.enable_exploration = False
        self.tau1 = 0.0
        self.tau2 = 1.0
        self.noise_scale = 0.01
        self.lowdim_size = 0
        self.noise_scale_for_lowdim = 0.01

        self.enable_action_noise = False
        self.action_noise_scale = 0.05

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            return_intermediate=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        if return_intermediate:
            action_list = []

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            if return_intermediate:
                action_list.append(trajectory.clone())

            # 2. predict model output
            if self.enable_exploration:
                global_cond_image, global_cond_lowdim = global_cond.split([self.obs_feature_dim - self.lowdim_size, self.lowdim_size], dim=-1)
                _global_cond_image = apply_modal_level_exploration(
                    global_cond_image, t, scheduler.config.num_train_timesteps, 
                    tau1=self.tau1, tau2=self.tau2, noise_scale=self.noise_scale
                )
                _global_cond_lowdim = apply_modal_level_exploration(
                    global_cond_lowdim, t, scheduler.config.num_train_timesteps, 
                    tau1=self.tau1, tau2=self.tau2, noise_scale=self.noise_scale_for_lowdim
                )
                _global_cond = torch.cat([_global_cond_image, _global_cond_lowdim], dim=-1)
            else:
                _global_cond = global_cond

            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=_global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        if self.enable_action_noise:
            trajectory += torch.randn_like(trajectory) * self.action_noise_scale

        if return_intermediate:
            action_list.append(trajectory.clone())
            return trajectory, action_list

        return trajectory

    def predict_action(self, readout) -> Dict[str, torch.Tensor]:
        B = readout.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = readout.device
        dtype = readout.dtype
        obs_features = readout
        assert obs_features.shape[0] == B * To
        
        # condition through global feature
        local_cond = None
        global_cond = None
        # reshape back to B, Do
        global_cond = obs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        sample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        action_pred = sample[...,:Da]

        return action_pred

    # ========= training  ============
    def compute_loss(self, readout, actions):
        batch_size = readout.shape[0]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = actions
        cond_data = trajectory
        assert readout.shape[0] == batch_size * self.n_obs_steps
        # reshape back to B, Do
        global_cond = readout.reshape(batch_size, -1) # (B, T*C)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def compute_weighted_loss(self, readout, actions, weights=None, weight_type="binary"):
        batch_size = readout.shape[0]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = actions
        cond_data = trajectory
        assert readout.shape[0] == batch_size * self.n_obs_steps
        # reshape back to B, Do
        global_cond = readout.reshape(batch_size, -1) # (B, T*C)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        if weights is not None:
            if weight_type=="binary":
                weights[weights<0] = 0
                weights[weights>0] = 1
            elif weight_type=="exp":
                weights = torch.exp(weights)
            elif weight_type=="sigmoid":
                weights = torch.sigmoid(weights)
            elif weight_type=="not_neg":
                weights[weights<0] = 0
            else:
                raise ValueError(f"Unsupported weight type {self.weight_type}")
            loss = loss * weights.unsqueeze(-1).type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
