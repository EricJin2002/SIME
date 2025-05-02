# Compared to robomimic.py, this v2 version loads the entire hdf5 file into memory to speed up the data loading process.

import os
import h5py
import torch
import numpy as np
import collections.abc as container_abcs
from torch.utils.data import Dataset

from policy.normalizer import Normalizer

def load_entire_hdf5(dct):
    if isinstance(dct, h5py.Dataset):
        return dct[()]
    ret = {}
    for k, v in dct.items():
        ret[k] = load_entire_hdf5(v)
    return ret

def get_from_loaded_hdf5(hdf5_loaded, path):
    path_list = path.split("/")
    cur = hdf5_loaded
    for key in path_list:
        cur = cur[key]
    return cur

class RoboMimicDataset(Dataset):
    """
    RoboMimic Dataset.
    """
    def __init__(
        self, 
        path, 
        train_filter_key = "train",
        num_obs = 1,
        for_residual_learning = False,
        num_action = 20, 
        success_only = False,
        traj_len_max_threshold = None,
        output_keys = ["actions"],
        output_obs_key_filter = None,
        weights_key = None,
        core_start = 500,
        not_weighted_on_core = True,
        total_num_states_limit = None,
        filter_frames_with_neg_weights = True, # Only for DP and note that the resulted trajectories may not be continuous
        normalize_actions = False,
        normalize_params = None,
        image_scale = 1.0,
        demo_num_limit = None
    ):

        self.num_obs = num_obs
        self.for_residual_learning = for_residual_learning
        assert self.num_obs == 1 or not self.for_residual_learning, "closed loop policy only conditioned on 1 previous obs currently"
        self.num_action = num_action
        self.weights_key = weights_key
        self.weights_already_used_for_filtering = False

        if isinstance(path, str):
            self.hdf5_file_path = path
            self.hdf5_file = h5py.File(self.hdf5_file_path, "r")
        else:
            self.hdf5_file = path
        self.hdf5_loaded = load_entire_hdf5(self.hdf5_file)
        print("hdf5 file loaded")
        all_demos = self.hdf5_file["mask/"+train_filter_key]
        if not success_only:
            self.all_demos = all_demos
        else:
            self.all_demos = []
            for i in range(len(all_demos)):
                dones = self.hdf5_file[os.path.join("data", str(all_demos[i], encoding="utf8"), "dones")]
                if np.any(dones):
                    self.all_demos.append(all_demos[i])
        if demo_num_limit is not None:
            print("demo_num_limit:", demo_num_limit)
            print("all_demos before applying limit:", list(self.all_demos))
            assert core_start is not None
            all_demos_except_core = []
            for i in range(len(self.all_demos)):
                if int(self.all_demos[i].decode("utf-8").split("_")[-1]) < core_start:
                    all_demos_except_core.append(self.all_demos[i])
            np.random.shuffle(all_demos_except_core)
            core_num = len(self.all_demos) - len(all_demos_except_core)
            self.all_demos = all_demos_except_core[:demo_num_limit - core_num] + list(self.all_demos)[-core_num:]
            print("all_demos after applying limit:", self.all_demos)
        self.num_demos = len(self.all_demos)
        print("num_demos:", self.num_demos)

        if total_num_states_limit is not None:
            self.collected_demos = []
            self.core_demos = []
            for i in range(len(self.all_demos)):
                if int(self.all_demos[i].decode("utf-8").split("_")[-1]) < core_start:
                    self.collected_demos.append(self.all_demos[i])
                else:
                    self.core_demos.append(self.all_demos[i])
            # randomly shuffle the demos
            np.random.shuffle(self.collected_demos)
            np.random.shuffle(self.core_demos)
            # make sure the core demos are used first
            self.all_demos = self.core_demos + self.collected_demos

        self.data_paths = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        demo_cnt = 0

        for i in range(self.num_demos):
            demo_path = os.path.join("data", str(self.all_demos[i], encoding="utf8"))
            
            frame_ids = list(range(self.hdf5_file[demo_path].attrs["num_samples"]))

            if traj_len_max_threshold is not None:
                if len(frame_ids) > traj_len_max_threshold and (
                    core_start is None or int(self.all_demos[i].decode("utf-8").split("_")[-1]) < core_start
                ):
                    # filter out trajectories with length greater than threshold
                    print("Filtered out trajectory with length greater than threshold", len(frame_ids))
                    continue
            
            if not self.for_residual_learning and self.weights_key is not None and filter_frames_with_neg_weights:
                # filter out timesteps with negative weights in advance for DP
                weights = self.hdf5_file[os.path.join(demo_path, self.weights_key)]

                # NOTE: this is a hack to avoid using weights for the core dataset
                if not_weighted_on_core and int(self.all_demos[i].decode("utf-8").split("_")[-1]) >= core_start:
                    weights = np.ones(len(weights))

                frame_ids = [frame_id for frame_id in frame_ids if weights[frame_id] > 0]
                if not self.weights_already_used_for_filtering:
                    self.weights_already_used_for_filtering = True
                    print("Filtered out timesteps with negative weights for DP")
            obs_frame_ids_list = []
            action_frame_ids_list = []
            
            for cur_idx in range(len(frame_ids) - 1):
                obs_pad_before = max(0, num_obs - cur_idx - 1)
                action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - num_obs + 1)
                frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                if self.for_residual_learning:
                    obs_frame_ids = frame_ids[cur_idx: frame_end - 1] + frame_ids[-1:] * action_pad_after
                obs_frame_ids_list.append(obs_frame_ids)
                action_frame_ids_list.append(action_frame_ids)
            
            self.data_paths += [demo_path] * len(obs_frame_ids_list)
            self.obs_frame_ids += obs_frame_ids_list
            self.action_frame_ids += action_frame_ids_list
            demo_cnt += 1

            if total_num_states_limit is not None and len(self.obs_frame_ids) >= total_num_states_limit:
                print("total_num_states_limit reached")
                break
        
        print("used demos:", demo_cnt)

        self.output_keys = output_keys
        self.output_obs_key_filter = output_obs_key_filter

        self.core_start = core_start
        self.not_weighted_on_core = not_weighted_on_core
        self.normalize_actions = normalize_actions
        self.normalize_params = normalize_params
        self.image_scale = image_scale

    def __len__(self):
        return len(self.obs_frame_ids)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        obs_dict = {
            key: torch.from_numpy(np.stack([
                get_from_loaded_hdf5(self.hdf5_loaded, os.path.join(data_path, "obs", key))[frame_id]
                # self.hdf5_file[os.path.join(data_path, "obs", key)][frame_id]
                for frame_id in obs_frame_ids
            ])).float() 
            if np.ndim(get_from_loaded_hdf5(self.hdf5_loaded, os.path.join(data_path, "obs", key))[0]) < 3 else
            # if np.ndim(self.hdf5_file[os.path.join(data_path, "obs", key)][0]) < 3 else
            torch.from_numpy(np.stack([
                get_from_loaded_hdf5(self.hdf5_loaded, os.path.join(data_path, "obs", key))[frame_id]
                # self.hdf5_file[os.path.join(data_path, "obs", key)][frame_id]
                for frame_id in obs_frame_ids
            ])[...,:3].transpose(0,3,1,2)).float()/self.image_scale    # rgba, NHWC -> rgb, NCHW
            for key in self.hdf5_file[os.path.join(data_path, "obs")].keys()
        }

        actions = np.stack([
            np.concatenate([
                (lambda x: x if x.ndim > 0 else np.expand_dims(x, axis=0))(
                    get_from_loaded_hdf5(self.hdf5_loaded, os.path.join(data_path, output_key))[frame_id]
                    # np.array(self.hdf5_file[os.path.join(data_path, output_key)][frame_id])
                    if output_key != "obs" and output_key != "next_obs" else
                    np.concatenate([
                        v[frame_id] 
                        if k in self.output_obs_key_filter
                        else []
                        for k, v in get_from_loaded_hdf5(self.hdf5_loaded, os.path.join(data_path, output_key)).items()
                        # for k, v in self.hdf5_file[os.path.join(data_path, output_key)].items()
                    ])
                ) for output_key in self.output_keys
            ])
            for frame_id in action_frame_ids
        ])

        # normalize actions
        if self.normalize_actions:
            actions = Normalizer.normalize(actions, self.normalize_params)

        actions = torch.from_numpy(actions).float()

        ret_dict = {
            "obs": obs_dict,
            "action": actions,
        }

        if self.weights_key is not None and not self.weights_already_used_for_filtering:
            weights = torch.from_numpy(np.stack([
                get_from_loaded_hdf5(self.hdf5_loaded, os.path.join(data_path, self.weights_key))[frame_id]
                # self.hdf5_file[os.path.join(data_path, self.weights_key)][frame_id]
                for frame_id in action_frame_ids
            ])).float()

            # NOTE: this is a hack to avoid using weights for the core dataset
            if self.not_weighted_on_core and int(data_path.split("_")[-1]) >= self.core_start:
                weights = torch.ones_like(weights)
            
            ret_dict["weights"] = weights

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            ret_dict[key] = collate_fn([d[key] for d in batch])
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

def pre_process_data(data, device, cfg):
    if cfg.policy.name == "BC":
        action_data = data['action'].squeeze(1).to(device)
    else:
        action_data = data['action'].to(device)

    # NOTE: squeeze(1) since num_obs is asserted to be 1
    if cfg.policy.name == "ResBC":
        obs_dict = {key: data['obs'][key].to(device) for key in cfg.policy.params.obs_shape_meta.keys()}
    else:
        obs_dict = {key: data['obs'][key].to(device).squeeze(1) for key in cfg.policy.params.obs_shape_meta.keys()}

    batch = dict(
        obs_dict=obs_dict,
        actions=action_data
    )

    if "weights" in data:
        weights = data["weights"].to(device)
        batch["weights"] = weights

    return batch