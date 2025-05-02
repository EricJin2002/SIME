import torch
import torch.nn as nn

from copy import deepcopy
from easydict import EasyDict
from collections import OrderedDict

import robomimic.models.policy_nets as PolicyNets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.obs_utils as ObsUtils

def get_bc(algo_config):
    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)

    rnn_enabled = algo_config.rnn.enabled
    # support legacy configs that do not have "transformer" item
    transformer_enabled = ("transformer" in algo_config) and algo_config.transformer.enabled

    if gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class = BC_Gaussian
    elif gmm_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class = BC_GMM
    elif vae_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class = BC

    return algo_class

def obs_encoder_kwargs_from_config(obs_encoder_config):
    for obs_modality, encoder_kwargs in obs_encoder_config.items():
        # First run some sanity checks and store the classes
        for cls_name, cores in zip(("core", "obs_randomizer"), (ObsUtils.OBS_ENCODER_CORES, ObsUtils.OBS_RANDOMIZERS)):
            # Make sure the requested encoder for each obs_modality exists
            cfg_cls = encoder_kwargs[f"{cls_name}_class"]
            if cfg_cls is not None:
                assert cfg_cls in cores, f"No {cls_name} class with name {cfg_cls} found, must register this class before" \
                    f"creating model!"
                # encoder_kwargs[f"{cls_name}_class"] = cores[cfg_cls]

        # Process core and randomizer kwargs
        encoder_kwargs.core_kwargs = dict() if encoder_kwargs.core_kwargs is None else \
            deepcopy(encoder_kwargs.core_kwargs)
        encoder_kwargs.obs_randomizer_kwargs = dict() if encoder_kwargs.obs_randomizer_kwargs is None else \
            deepcopy(encoder_kwargs.obs_randomizer_kwargs)
    return dict(obs_encoder_config)

class BC(nn.Module):
    """
    Normal BC training.
    """
    def __init__(
        self,
        obs_shape_meta,
        action_dim,
        obs_encoder,
        algo
    ):
        super().__init__()
        self.obs_shape_meta = obs_shape_meta
        self.obs_shapes = OrderedDict({k: v["shape"] for k, v in self.obs_shape_meta.items()})
        self.goal_shapes = OrderedDict()
        self.obs_spec = dict()
        for k, v in obs_shape_meta.items():
            if v["type"] not in self.obs_spec:
                self.obs_spec[v["type"]] = []
            self.obs_spec[v["type"]].append(k)
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": self.obs_spec, "goal": dict()})
        self.obs_config = EasyDict(encoder=obs_encoder)
        self.ac_dim = action_dim
        self.algo_config = algo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._create_networks()

    def _create_networks(self):
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)
    
    def get_action(self, obs_dict):
        assert not self.nets.training
        return self.nets["policy"](obs_dict)

    def forward(self, obs_dict, actions = None):
        '''
        obs_dict: {k: [B, Do]}
        actions: [B, Da]
        '''
        if actions is None:
            return self.get_action(obs_dict)
        
        actions_pred = self.nets["policy"](obs_dict=obs_dict)
        
        losses = OrderedDict()
        losses["l2_loss"] = nn.MSELoss()(actions_pred, actions)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions_pred, actions)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions_pred[..., :3], actions[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["loss"] = action_loss
        return losses

class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        assert self.algo_config.gaussian.enabled
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def forward(self, obs_dict, actions = None, weights = None):
        '''
        obs_dict: {k: [B, Do]}
        actions: [B, Da]
        '''
        if actions is None:
            return self.get_action(obs_dict)
        
        dists = self.nets["policy"].forward_train(obs_dict=obs_dict)

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(actions)

        # estimate entropy of the distribution`while dist.entropy() is not implemented
        sample_log_probs = dists.log_prob(dists.sample())

        losses = OrderedDict()
        if weights is None or self.algo_config.loss.weight_type is None:
            pred_loss = -log_probs.mean()
        elif self.algo_config.loss.weight_type == "-w_log_p":
            exploit_term = -(log_probs[weights>0] * weights[weights>0])
            explore_term = -(sample_log_probs[weights<0] * weights[weights<0])
            pred_loss = torch.cat([exploit_term, explore_term], dim=0).mean()
            losses["pred_loss_exploit_term"] = exploit_term.mean()
            losses["pred_loss_explore_term"] = explore_term.mean()
        elif self.algo_config.loss.weight_type == "-log_p":
            pred_loss = -log_probs[weights>0].mean()
        elif self.algo_config.loss.weight_type == "-w_p":
            pred_loss = -(torch.exp(log_probs) * weights).mean()
        elif self.algo_config.loss.weight_type == "-exp_w_log_p":
            pred_loss = -(log_probs * torch.exp(weights)).mean()
        else:
            raise NotImplementedError
        
        losses["pred_loss"] = pred_loss
        losses["regu_loss"] = dists.mean.pow(2).mean()
        losses["entropy_loss"] = sample_log_probs.mean()
        action_losses = [
            self.algo_config.loss.pred_weight * losses["pred_loss"],
            self.algo_config.loss.regu_weight * losses["regu_loss"],
            self.algo_config.loss.entropy_weight * losses["entropy_loss"],
        ]
        action_loss = sum(action_losses)
        losses["loss"] = action_loss
        return losses


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        assert self.algo_config.gmm.enabled
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)
