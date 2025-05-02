import os
import json
import h5py
import torch
import argparse
import numpy as np
from easydict import EasyDict

import robomimic.utils.tensor_utils as TensorUtils

from image_generation_utils import TrajectoryRenderer
from rotation_transformer import RotationTransformer
from rollout_utils import rollout, vanilla_load, dp_load

def run_trained_agent(args):
    if args.config is not None:
        # load config for diffusion policy
        with open(args.config, "r") as f:
            cfg = json.load(f)
        cfg = EasyDict(cfg)
    else:
        cfg = None

    # some arg checking
    write_video = (args.video_dir is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    if cfg is None:
        policy, env, rollout_num_episodes, rollout_horizon = vanilla_load(args)
    else:
        policy, env, rollout_num_episodes, rollout_horizon = dp_load(args, cfg)

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        if not os.path.exists(os.path.dirname(args.dataset_path)):
            os.makedirs(os.path.dirname(args.dataset_path))
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    if args.render_traj:
        traj_renderer = TrajectoryRenderer(env, "agentview")
    else:
        traj_renderer = None

    if args.abs_action:
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
    else:
        rotation_transformer = None

    rollout_stats = []
    last_initial_state_dict = None
    for i in range(rollout_num_episodes):
        print("Rollout {}".format(i))

        if i % args.try_times == 0:
            initial_state_dict = None
        else:
            initial_state_dict = last_initial_state_dict

        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_dir=os.path.join(args.video_dir, "demo_{}".format(i)) if write_video else None, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            initial_state_dict=initial_state_dict,
            traj_renderer=traj_renderer,
            abs_action=args.abs_action,
            rotation_transformer=rotation_transformer,
        )
        rollout_stats.append(stats)
        last_initial_state_dict = traj["initial_state_dict"]

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            for k in traj:
                if k in ["actions", "states", "rewards", "dones", "obs", "next_obs", "initial_state_dict"]:
                    continue
                ep_data_grp.create_dataset(k, data=np.array(traj[k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

        print("Success_Rate", np.mean(TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)["Success_Rate"]))

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # Path to trained model
    parser.add_argument(
        "--critic_agent",
        type=str,
        default=None,
        help="(optional) path to saved checkpoint pth file",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to config file, only needed by diffusion policy",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=10,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    parser.add_argument(
        "--render_traj",
        action='store_true',
        help="render trajectory",
    )

    # Dump videos of the rollouts to the specified dir
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="(optional) render rollouts to this dir",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--try_times",
        type=int,
        default=1,
        help="(optional) number of times to try to start from the same initial state",
    )

    parser.add_argument(
        "--inference_horizon",
        type=int,
        default=None,
        help="(optional) inference horizon",
    )

    parser.add_argument(
        "--high_noise_eval",
        action='store_true',
        help="(optional) high noise eval",
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="(optional) eta",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="(optional) number of inference steps",
    )

    parser.add_argument(
        "--enable_exploration",
        action='store_true',
        help="(optional) enable modal-level exploration",
    )
    parser.add_argument(
        "--tau1",
        type=float,
        default=None,
        help="(optional) exploration tau1",
    )
    parser.add_argument(
        "--tau2",
        type=float,
        default=None,
        help="(optional) exploration tau2",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=None,
        help="(optional) exploration noise scale",
    )

    parser.add_argument(
        "--enable_action_noise",
        action='store_true',
        help="(optional) enable action noise",
    )
    parser.add_argument(
        "--action_noise_scale",
        type=float,
        default=None,
        help="(optional) action noise scale",
    )

    parser.add_argument(
        "--abs_action",
        action='store_true',
        help="(optional) use absolute actions",
    )

    parser.add_argument(
        "--return_intermediate",
        action='store_true',
        help="(optional) return intermediate states",
    )

    args = parser.parse_args()
    run_trained_agent(args)

