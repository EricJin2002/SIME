import json
import torch
import torch.nn as nn

from copy import deepcopy
from easydict import EasyDict

from policy.bc import get_bc

class ResBC(nn.Module):
    def __init__(
        self,
        base_policy_ckpt,
        base_policy_config,
        predict_residual = False,
        base_action_source = "base_policy",
        action_noise_std = 0.02,    # Only used when base_action_source == "noisified_actions"

        **residual_params
    ):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(base_policy_config, "r") as f:
            self.base_policy_cfg = EasyDict(json.load(f))
        if self.base_policy_cfg.policy.name == "DP":
            from policy.dp import DP
            self.base_policy = DP(**self.base_policy_cfg.policy.params).to(self.device)
        else:
            raise NotImplementedError
        self.base_policy.load_state_dict(torch.load(base_policy_ckpt, map_location = self.device), strict = False)

        self.residual_params = EasyDict(deepcopy(residual_params))
        self.residual_params.obs_shape_meta["_action"] = dict(
            shape = [self.base_policy_cfg.policy.params.action_dim],
            type = "low_dim"
        )
        self.residual_policy = get_bc(self.residual_params.algo)(**self.residual_params).to(self.device)
        self.predict_residual = predict_residual
        self.base_action_source = base_action_source
        self.action_noise_std = action_noise_std

    def forward(self, obs_dict, actions, weights=None):
        actions = actions.view(-1, actions.size(-1))
        if self.base_action_source == "base_policy":
            base_actions = self.base_policy({k: v[:,0,:] for k, v in obs_dict.items()}).detach()
            base_actions = base_actions.view(-1, base_actions.size(-1))
        else:
            raise NotImplementedError
        
        obs_dict = {k: v.view(-1, v.size(-1)) for k, v in obs_dict.items()}
        obs_dict["_action"] = base_actions

        if weights is not None:
            weights = weights.view(-1)
        if self.predict_residual:
            actions = actions - base_actions
        losses = self.residual_policy(obs_dict, actions, weights=weights)
        return losses