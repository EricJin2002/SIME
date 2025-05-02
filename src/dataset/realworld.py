import os
import json
import glob
import torch
import numpy as np
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image
from torch.utils.data import Dataset

from utils.constants import *
from utils.transformation import xyz_rot_transform


class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        cam_ids=['135122075425', '242322072982'],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        binarize_gripper = False
    ):
        assert split in ['train', 'val', 'all', '']

        if not isinstance(path, list):
            path = [path]
        self.data_path_list = path
        self.split = split
        self.num_obs = num_obs
        self.num_action = num_action
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.binarize_gripper = binarize_gripper
        
        self.data_paths = []
        self.cam_ids = cam_ids
        self.obs_frame_ids = []
        self.action_frame_ids = []
        
        for data_path in self.data_path_list:
            all_demos = glob.glob(os.path.join(data_path, '*'))
            num_demos = len(all_demos)
            for i in range(num_demos):
                demo_path = all_demos[i]
                cam_id = cam_ids[0]

                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # metadata
                if os.path.exists(os.path.join(demo_path, "metadata.json")):
                    with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                        meta = json.load(f)
                    # get frame ids
                    frame_ids = [
                        int(os.path.splitext(x)[0]) 
                        for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                        if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                    ]
                else:
                    # get frame ids
                    frame_ids = [
                        int(os.path.splitext(x)[0]) 
                        for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    ]
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
        
    def __len__(self):
        return len(self.obs_frame_ids)

    @staticmethod
    def _normalize_tcp(tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - WORLD_TRANS_MIN) / (WORLD_TRANS_MAX - WORLD_TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list
    
    def __getitem__(self, index):
        data_path = self.data_paths[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dirs = [os.path.join(data_path, "cam_{}".format(cam_id), 'color') for cam_id in self.cam_ids]
        depth_dirs = [os.path.join(data_path, "cam_{}".format(cam_id), 'depth') for cam_id in self.cam_ids]
        tcp_dir = os.path.join(data_path, 'tcp')
        gripper_dir = os.path.join(data_path, 'gripper_command')

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # load colors and depths
        colors_dict = {cam_id: [] for cam_id in self.cam_ids}
        depths_dict = {cam_id: [] for cam_id in self.cam_ids}
        obs_tcps = []
        obs_grippers = []
        for frame_id in obs_frame_ids:
            for color_dir, depth_dir, cam_id in zip(color_dirs, depth_dirs, self.cam_ids):
                colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
                if self.split == 'train' and self.aug_jitter:
                    colors = jitter(colors)
                colors = np.array(colors).transpose(2, 0, 1) # HWC to CHW
                colors_dict[cam_id].append(colors)
                depths_dict[cam_id].append(
                    np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
                )
            # observed tcps and grippers
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            projected_tcp = tcp # world frame tcp
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            if self.binarize_gripper:
                gripper_width = MAX_GRIPPER_WIDTH if gripper_width > MAX_GRIPPER_WIDTH / 3 else 0.0
            obs_tcps.append(projected_tcp)
            obs_grippers.append(gripper_width)
        colors_dict = {cam_id: np.stack(colors, axis = 0) for cam_id, colors in colors_dict.items()}
        depths_dict = {cam_id: np.stack(depths, axis = 0) for cam_id, depths in depths_dict.items()}

        # observation conversion
        obs_tcps = np.stack(obs_tcps)
        obs_grippers = np.stack(obs_grippers)
        obs_tcps = xyz_rot_transform(obs_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        obs_low_dim = np.concatenate((obs_tcps, obs_grippers[..., np.newaxis]), axis = -1)
        obs_low_dim_normalized = self._normalize_tcp(obs_low_dim.copy())
        obs_low_dim = torch.from_numpy(obs_low_dim).float()
        obs_low_dim_normalized = torch.from_numpy(obs_low_dim_normalized).float()

        # actions
        action_tcps = []
        action_grippers = []
        for frame_id in action_frame_ids:
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            projected_tcp = tcp # world frame tcp
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            if self.binarize_gripper:
                gripper_width = MAX_GRIPPER_WIDTH if gripper_width > MAX_GRIPPER_WIDTH / 3 else 0.0
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)
        
        # rotation transformation (to 6d)
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis = -1)

        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'colors_dict': colors_dict,
            'depths_dict': depths_dict,
            'obs_low_dim': obs_low_dim,  
            'obs_low_dim_normalized': obs_low_dim_normalized,
            'action': actions,
            'action_normalized': actions_normalized
        }
        
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


def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095


def pre_process_data(data, device, cfg):
    action_data = data['action_normalized'].to(device)
    colors = {key: (value.squeeze(1).to(device)).squeeze(1)/255 for key, value in data['colors_dict'].items()}
    obs_dict = colors
    if cfg.policy.predict_delta_pos:
        obs_low_dim = data['obs_low_dim_normalized'].to(device)
        action_data[..., :3] = action_data[..., :3] - obs_low_dim[..., :3]
    if cfg.policy.name == "BC":
        action_data = action_data.squeeze(1)
    if cfg.policy.include_low_dim:
        obs_low_dim = data['obs_low_dim_normalized'].to(device)
        obs_low_dim = obs_low_dim.squeeze(1)
        obs_dict["obs_low_dim"] = obs_low_dim
        # if include_low_dim, remember to add the followings to the config
        # obs_shape_meta["obs_low_dim"] = dict(
        #     shape = dataset[0]['obs_low_dim_normalized'].shape[1:],
        #     type = 'low_dim'
        # )

    return dict(obs_dict=obs_dict, actions=action_data)

def pre_process_inputs(colors_dict, depths_dict, device, cfg, **kwargs):
    obs_dict = {
        key: torch.tensor(
            np.array([colors_dict[key]]).transpose(0,3,1,2)
        ).to(device).float() / 255
        for key in colors_dict
    }

    if cfg.policy.include_low_dim:
        obs_dict["obs_low_dim"] = kwargs["obs_low_dim_normalized"].to(device)

    return obs_dict
