import torch
import torch.nn as nn

from policy.diffusion import DiffusionUNetPolicy
from policy.img_encoder.multi_image_obs_encoder import MultiImageObsEncoder

class DP(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        obs_shape_meta = dict(
            color_image = dict(
                shape = [3, 64, 64],
                type = 'rgb'                    
            )
        ),
        action_dim = 10, 
        weight_type = None,
        use_group_norm = True,
        resnet_out_features = 64,

        # parameters passed to step
        **kwargs
    ):
        super().__init__()
        num_obs = 1
        self.img_encoder = MultiImageObsEncoder(
            shape_meta = dict(
                action = dict(shape = [action_dim]),
                obs = obs_shape_meta,
            ),
            use_group_norm = use_group_norm,
            share_rgb_model = False,
            imagenet_norm = True,
            resnet_out_features = resnet_out_features
        )
        obs_feature_dim = self.img_encoder.output_shape()[0]
        print("obs_feature_dim:", obs_feature_dim)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim, **kwargs)
        self.weight_type = weight_type

    def forward(
            self, obs_dict, actions = None, 
            return_intermediate = False, 
            weights = None
        ):
        readout = self.img_encoder(obs_dict)
        if return_intermediate:
            B = readout.shape[0]
            T = self.action_decoder.horizon
            Da = self.action_decoder.action_dim
            Do = self.action_decoder.obs_feature_dim
            To = self.action_decoder.n_obs_steps

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
            if return_intermediate:
                sample, action_list = self.action_decoder.conditional_sample(
                    cond_data, 
                    cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    return_intermediate=return_intermediate,
                    **self.action_decoder.kwargs)
                action_pred = sample[...,:Da]
                return action_pred, action_list
            
            sample = self.action_decoder.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                return_intermediate=return_intermediate,
                **self.action_decoder.kwargs)
            action_pred = sample[...,:Da]
            return action_pred

        if actions is not None:
            # loss = self.action_decoder.compute_loss(readout, actions)
            loss = self.action_decoder.compute_weighted_loss(readout, actions, weights=weights, weight_type=self.weight_type)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred