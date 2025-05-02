import os
import sys
import torch
import imageio
import numpy as np
from copy import deepcopy
from PIL import Image

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

from image_generation_utils import TrajectoryRenderer
from realworld_utils import apply_image_filter, plot

DP_SRC_ROOT = os.path.join(os.path.dirname(__file__), "../src")
print("diffusion policy source root:", DP_SRC_ROOT)
sys.path.insert(0, DP_SRC_ROOT)

from policy.normalizer import Normalizer

def rollout(
    policy, env, horizon, 
    render=False, video_dir=None, video_skip=5, 
    return_obs=False, camera_names=None,
    initial_state_dict=None,
    traj_renderer:TrajectoryRenderer=None,
    abs_action=False,
    rotation_transformer=None,
):
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy) or isinstance(policy, RolloutDP)
    assert not (render and (video_dir is not None))
    write_video = (video_dir is not None)
    render_traj = (traj_renderer is not None)

    # maybe create video writer
    video_writer = None
    if write_video:
        os.makedirs(video_dir, exist_ok=True)
        video_root = "/".join(video_dir.split("/")[:-1])
        demo_id = video_dir.split("/")[-1]
        os.makedirs(os.path.join(video_root, "video"), exist_ok=True)
        video_writer = imageio.get_writer(os.path.join(video_root, "video", demo_id+".mp4"), fps=20)
        # video_writer = imageio.get_writer(os.path.join(video_dir, "video.mp4"), fps=20)

    policy.start_episode()
    obs = env.reset()
    if initial_state_dict is None:
        state_dict = env.get_state()
    else:
        state_dict = initial_state_dict

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(
        actions=[], 
        rewards=[], 
        dones=[], 
        states=[], 
        # v_preds=[],
        # q_preds=[],
        initial_state_dict=state_dict,
    )
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            act = policy(ob=obs)
            if isinstance(policy, RolloutDP):
                if policy.base_ac is not None:
                    if "base_actions" not in traj:
                        print("store base actions")
                        traj["base_actions"] = []
                    traj["base_actions"].append(policy.base_ac)

                if policy.t==1 and policy.intermediate_ac is not None and abs_action:
                    if write_video and render_traj:
                        assert return_obs
                        frame = env.render(mode="rgb_array", height=480, width=480, camera_name=traj_renderer.camera_name)
                        image2 = Image.fromarray(frame)
                        # factor = get_save_mode_factor(0.5)
                        image_array = apply_image_filter(image2, 0.5)

                        def plot_points(points, img_array, **kwargs):
                            transformed_points = np.dot(points - traj_renderer.camera_position, traj_renderer.camera_rotation)
                            img_array = plot(transformed_points, img_array, **kwargs)
                            return img_array
                        
                        # os.makedirs(os.path.join(video_dir, "AAA_intermediate_points"), exist_ok=True)
                        
                        for i in range(policy.intermediate_ac.shape[0]):
                            if i != policy.intermediate_ac.shape[0]-1:
                                continue
                                # pass
                            img = image_array.copy()
                            img = plot_points(policy.intermediate_ac[i, :, :3], img, plot_mode="point")
                            # add text to img
                            # text = "x: {:.2f}, y: {:.2f}, z: {:.2f}".format(*policy.intermediate_ac[i, -1, :3])
                            # imageio.imsave(os.path.join(video_dir, "AAA_intermediate_points", "frame_{:06d}_step_{:03d}.png".format(video_count, i)), img)
                            imageio.imsave(os.path.join(video_dir, "frame_{:06d}_step_{:03d}.png".format(video_count, i)), img)

            # play action
            env_act = act
            if abs_action:
                env_act = rotation_transformer.undo_transform_action(env_act)
            next_obs, r, done, _ = env.step(env_act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]
            done = done or success
            assert done == success

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if write_video:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                    # imageio.imsave(os.path.join(video_dir, "frame_{:06d}.png".format(video_count)), video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            # traj["v_preds"].append(v_pred)
            # traj["q_preds"].append(q_pred)
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if write_video:
        video_writer.close()
        pass

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    if render_traj:
        assert return_obs
        # traj_points = traj["obs"]["robot0_eef_pos"][()]
        traj_points = traj["actions"][:, :3]
        samples = {"images": [], "labels": []}
        traj_renderer.render_trajectory_image(
            traj_points, None, samples, 
            save_mode=0.5, linewidth=1
        )
        assert len(samples["images"]) == 1
        sample = samples["images"][0]
        video_root = "/".join(video_dir.split("/")[:-1])
        demo_id = video_dir.split("/")[-1]
        os.makedirs(os.path.join(video_root, "traj"), exist_ok=True)
        imageio.imsave(os.path.join(video_root, "traj", demo_id+".png"), sample)

    return stats, traj

def vanilla_load(args):
    assert not args.abs_action, "Vanilla BC with absolute actions not supported yet"

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_dir is not None), 
        verbose=True,
    )

    return policy, env, rollout_num_episodes, rollout_horizon


class RolloutDP:
    def __init__(self, policy, args, cfg):
        self.policy = policy
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self.t = 0
        self.ac = None
        self.timestep = 0

        self.policy_name = cfg.policy.name
        if hasattr(cfg.dataset.params, "num_action"):
            self.horizon = cfg.dataset.params.num_action
        else:
            self.horizon = 1
        self.inference_horizon = args.inference_horizon
        if self.inference_horizon is None:
            self.inference_horizon = self.horizon
        print("horizon: ", self.horizon)
        print("inference_horizon: ", self.inference_horizon)
        
        if hasattr(cfg.policy.params, "predict_residual") and cfg.policy.params.predict_residual:
            self.predict_residual = True
        else:
            self.predict_residual = False

        if args.critic_agent:
            # relative path to agent
            ckpt_path = args.critic_agent
            # device
            device = TorchUtils.get_torch_device(try_to_use_cuda=True)
            # restore policy
            self.critic_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
        else:
            self.critic_policy = None

        if args.eta is not None:
            if self.policy_name == "DP":
                self.policy.action_decoder.kwargs = {"eta": args.eta}
            elif self.policy_name == "ResBC":
                self.policy.base_policy.action_decoder.kwargs = {"eta": args.eta}
        
        if args.num_inference_steps is not None:
            if self.policy_name == "DP":
                self.policy.action_decoder.num_inference_steps = args.num_inference_steps
            elif self.policy_name == "ResBC":
                self.policy.base_policy.action_decoder.num_inference_steps = args.num_inference_steps

        if args.enable_exploration:
            if self.policy_name == "DP":
                self.policy.action_decoder.enable_exploration = True
                if args.tau1 is not None:
                    self.policy.action_decoder.tau1 = args.tau1
                if args.tau2 is not None:
                    self.policy.action_decoder.tau2 = args.tau2
                if args.noise_scale is not None:
                    self.policy.action_decoder.noise_scale = args.noise_scale
                use_different_noise_scale_at_lowdim = False
                if use_different_noise_scale_at_lowdim:
                    self.policy.action_decoder.noise_scale_for_lowdim = 0.01
                    lowdim_size = 0
                    for k, v in cfg.policy.params.obs_shape_meta.items():
                        if v.type == "low_dim":
                            assert len(v.shape) == 1
                            lowdim_size += v.shape[0]
                    print("lowdim_size: ", lowdim_size)
                    self.policy.action_decoder.lowdim_size = lowdim_size
            elif self.policy_name == "ResBC":
                self.policy.base_policy.action_decoder.enable_exploration = True
                if args.tau1 is not None:
                    self.policy.base_policy.action_decoder.tau1 = args.tau1
                if args.tau2 is not None:
                    self.policy.base_policy.action_decoder.tau2 = args.tau2
                if args.noise_scale is not None:
                    self.policy.base_policy.action_decoder.noise_scale = args.noise_scale

        if args.enable_action_noise:
            if self.policy_name == "DP":
                self.policy.action_decoder.enable_action_noise = True
                if args.action_noise_scale is not None:
                    self.policy.action_decoder.action_noise_scale = args.action_noise_scale
            elif self.policy_name == "ResBC":
                self.policy.base_policy.action_decoder.enable_action_noise = True
                if args.action_noise_scale is not None:
                    self.policy.base_policy.action_decoder.action_noise_scale = args.action_noise_scale

        self.low_noise_eval = not args.high_noise_eval
        self.base_ac = None
        self.intermediate_ac = None
        if hasattr(cfg.dataset.params, "normalize_actions") and cfg.dataset.params.normalize_actions:
            self.normalizer = Normalizer(cfg.dataset.params.normalize_params)
        else:
            self.normalizer = None
        self.return_intermediate = args.return_intermediate

    def start_episode(self):
        self.policy.eval()
        # print(self.policy)
        # self.low_noise_eval = False
        if self.policy_name == "BC":
            self.policy.nets["policy"].low_noise_eval = self.low_noise_eval
        elif self.policy_name == "ResBC":
            self.policy.residual_policy.nets["policy"].low_noise_eval = self.low_noise_eval
        self.t = 0
        self.ac = None
        self.timestep = 0

    def _prepare_observation(self, ob):
        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.device)
        ob = TensorUtils.to_float(ob)
        return ob

    def get_v(self, ob):
        assert self.critic_policy is not None
        prepared_obs = self.critic_policy._prepare_observation(ob)
        v_pred = self.critic_policy.policy.nets["vf"](obs_dict=prepared_obs, goal_dict=None)
        v_pred = TensorUtils.to_numpy(v_pred).item()
        return v_pred

    def __call__(self, ob):
        if self.ac is None or self.t >= self.inference_horizon:
            ob = self._prepare_observation(ob)
            if self.policy_name == "DP":
                if self.return_intermediate:
                    ac, intermediate_ac = self.policy(ob, return_intermediate=self.return_intermediate)
                else:
                    ac = self.policy(ob)
                ac = TensorUtils.to_numpy(ac[0])
                if self.return_intermediate:
                    intermediate_ac = np.array([TensorUtils.to_numpy(ac_i[0]) for ac_i in intermediate_ac])
                if self.normalizer is not None:
                    ac = self.normalizer.unnormalize_(ac)
                    if self.return_intermediate:
                        intermediate_ac = self.normalizer.unnormalize_(intermediate_ac)
                self.ac = ac
                if self.return_intermediate:
                    self.intermediate_ac = intermediate_ac
                
                # print("ac: ", self.ac)
                # input()
            elif self.policy_name == "ResBC":
                # start_time = time.time()
                ac = TensorUtils.to_numpy(self.policy.base_policy(ob))
                if self.normalizer is not None:
                    ac = self.normalizer.unnormalize_(ac)
                self.ac = ac
                # print("base policy time: ", time.time()-start_time)
            else:
                ac = TensorUtils.to_numpy(self.policy(ob))
                if self.normalizer is not None:
                    ac = self.normalizer.unnormalize_(ac)
                self.ac = ac
            self.t = 0
        
        if self.policy_name == "ResBC":
            # start_time = time.time()
            obs_dict = self._prepare_observation(ob)
            obs_dict["_action"] = torch.tensor(self.ac[:, self.t, :]).to(self.device)
            residual_ac = TensorUtils.to_numpy(self.policy.residual_policy(obs_dict))[0]
            self.base_ac = TensorUtils.to_numpy(self.ac[0, self.t, :])
            if self.predict_residual:
                ac = self.base_ac + residual_ac
            else:
                ac = residual_ac
            # print("residual policy time: ", time.time()-start_time)
        else:
            # ac = self.ac[self.t][:7]
            ac = self.ac[self.t][:] # TODO: DP output might include other things besides action
        self.t += 1
        self.timestep += 1
        return ac

def dp_load(args, cfg):    
    obs_spec = dict()
    use_image_obs = False
    for k, v in cfg.policy.params.obs_shape_meta.items():
        if v.type not in obs_spec:
            if v.type == "rgb":
                use_image_obs = True
            obs_spec[v.type] = []
        obs_spec[v.type].append(k)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_spec, "goal": dict()})

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if cfg.policy.name == "DP":
        from policy.dp import DP
        policy = DP(**cfg.policy.params).to(device)
    elif cfg.policy.name == "BC":
        from policy.bc import get_bc
        policy = get_bc(cfg.policy.params.algo)(**cfg.policy.params).to(device)
    elif cfg.policy.name == "ResBC":
        from policy.residual_bc import ResBC
        policy = ResBC(**cfg.policy.params).to(device)
    else:
        raise NotImplementedError
    policy.load_state_dict(torch.load(args.agent, map_location = device), strict = False)
    policy = RolloutDP(policy, args, cfg)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.dataset.params.path)
    if args.abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    
    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # same as robomimic default
        if env_meta["env_name"] in ["TwoArmTransport", "ToolHang"]:
            rollout_horizon = 700
        else:
            rollout_horizon = 400 # default
    
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"], 
        render=args.render, 
        render_offscreen=(args.video_dir is not None), 
        use_image_obs=use_image_obs,
        use_depth_obs=False, # comment this if you use an early version of robomimic
    )

    return policy, env, rollout_num_episodes, rollout_horizon
