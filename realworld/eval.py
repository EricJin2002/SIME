import os
import cv2
import time
import json
import torch
import shutil
import argparse
import numpy as np
import torch.nn as nn
import multiprocessing

from PIL import Image
from easydict import EasyDict as edict

from eval_agent import Agent
from device.keyboard import Keyboard

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from utils.constants import *
from utils.training import set_seed
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform, xyz_rot_transform
from dataset.realworld import RealWorldDataset

def unnormalize_action(action, cfg):
    if cfg.policy.name == "DP":
        action[..., :3] = (action[..., :3] + 1) / 2.0 * (WORLD_TRANS_MAX - WORLD_TRANS_MIN) + WORLD_TRANS_MIN
    elif cfg.policy.name == "RISE":
        action[..., :3] = (action[..., :3] + 1) / 2.0 * (CAM_TRANS_MAX - CAM_TRANS_MIN) + CAM_TRANS_MIN
    else:
        raise NotImplementedError
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps

def save_data(
        color_image, 
        depth_image, 
        color_dir,
        depth_dir,):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(color_image).save(color_dir)
    Image.fromarray(depth_image).save(depth_dir)
    # print("saving data to", color_dir)
                    
class Recorder:
    def __init__(
        self,      
        agent: Agent, 
        pool: multiprocessing.Pool, 
        cam_ids, 
        record_path
    ):
        self.agent = agent
        self.pool = pool
        self.cam_ids = cam_ids
        self.record_path = record_path

    def new(self):
        self.start_time = int(time.time() * 1000)
        self.demo_path = os.path.join(self.record_path, f'{self.start_time}')
        os.makedirs(self.demo_path)

        self.cam_path = [os.path.join(self.demo_path, "cam_{}".format(s)) for s in self.cam_ids]
        self.color_dir = [os.path.join(path, 'color') for path in self.cam_path]
        self.depth_dir = [os.path.join(path, 'depth') for path in self.cam_path]
        for path in self.cam_path:
            os.mkdir(path)
        for path in self.color_dir:
            os.mkdir(path)
        for path in self.depth_dir:
            os.mkdir(path)
        
        self.tcp_dir = os.path.join(self.demo_path, 'tcp')
        self.joint_dir = os.path.join(self.demo_path, 'joint')
        self.action_dir = os.path.join(self.demo_path, 'action')
        self.gripper_dir = os.path.join(self.demo_path, 'gripper_command')
        os.mkdir(self.tcp_dir)
        os.mkdir(self.joint_dir)
        os.mkdir(self.gripper_dir)
        os.mkdir(self.action_dir)
        
        with open(os.path.join(self.demo_path, "timestamp.txt"), "w") as f:
            f.write('1')

    def save(self):
        curr_time = int(time.time() * 1000)
        cam_data = []
        for camera in self.agent.cameras:
            color_image, depth_image = camera.get_data()
            cam_data.append((color_image, depth_image))
        for (color_image, depth_image), color_path, depth_path in zip(cam_data, self.color_dir, self.depth_dir):
            self.pool.apply_async(save_data, args=(
                color_image.copy(), 
                depth_image.copy(), 
                os.path.join(color_path, f'{curr_time}.png'), 
                os.path.join(depth_path, f'{curr_time}.png'))
            )
        tcpPose, jointPose, _, _ = self.agent.robot.get_robot_state()
        width = self.agent.gripper.get_gripper_state() * 1000 / self.agent.gripper.max_width
        np.save(os.path.join(self.tcp_dir, f'{curr_time}.npy'), tcpPose)
        np.save(os.path.join(self.joint_dir, f'{curr_time}.npy'), jointPose)
        np.save(os.path.join(self.gripper_dir, f'{curr_time}.npy'), [width])

    def good(self):
        with open(os.path.join(self.demo_path, "label.txt"), "w") as f:
            f.write('good')
    
    def bad(self):
        with open(os.path.join(self.demo_path, "label.txt"), "w") as f:
            f.write('bad')
    
    def success(self):
        with open(os.path.join(self.demo_path, "preference.txt"), "w") as f:
            f.write('success')
    
    def fail(self):
        with open(os.path.join(self.demo_path, "preference.txt"), "w") as f:
            f.write('fail')

    def discard(self):
        print('WARNING: discard the demo!')
        time.sleep(5)
        shutil.rmtree(self.demo_path)
    

def evaluate_single(
        cfg,
        args, 
        agent: Agent, 
        policy: nn.Module, 
        keyboard: Keyboard, 
        pool: multiprocessing.Pool, 
        device: torch.device, 
    ):
    keyboard.discard = False
    # keyboard.finish = False
    keyboard.quit = False
    keyboard.success = False    # j
    keyboard.fail = False       # k
    keyboard.good = False       # g
    keyboard.bad = False        # b
    keyboard.drop = False       # h

    ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)

    if args.record:
        traj_recorder = Recorder(agent, pool, agent.cam_ids, args.record_path)
        traj_recorder.new()
        if args.step_record:
            step_recorder_path_suffix = "_steps"
            step_recorder_path = args.record_path + step_recorder_path_suffix
            step_recorder = Recorder(agent, pool, agent.cam_ids, step_recorder_path)
            step_recorder.new()

    if args.vis:
        pass

    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype = np.float32)
    with torch.inference_mode():
        policy.eval()
        prev_width = None
        for t in range(args.max_steps):
            if keyboard.quit or keyboard.discard or keyboard.success or keyboard.fail:
                break

            if args.record:
                traj_recorder.save()
                if args.step_record:
                    step_recorder.save()

            if t % args.num_inference_step == 0:
                if args.record:
                    while True and t != 0 and args.step_record:
                        cv2.waitKey(0)
                        time.sleep(0.1)
                        if keyboard.quit or keyboard.discard or keyboard.success or keyboard.fail:
                            break
                        if keyboard.good:
                            step_recorder.good()
                            step_recorder.new()
                            break
                        if keyboard.bad:
                            step_recorder.bad()
                            step_recorder.new()
                            break
                        if keyboard.drop:
                            step_recorder.discard()
                            step_recorder.new()
                            break
                    keyboard.good = False       # g
                    keyboard.bad = False        # b
                    keyboard.drop = False       # h
                    if keyboard.quit or keyboard.discard or keyboard.success or keyboard.fail:
                        break

                keyboard.ctn = True
                start_from_current = False
                while keyboard.ctn or keyboard.switch:
                    time.sleep(0.1)

                    keyboard.ctn = False
                    if keyboard.switch:
                        start_from_current = not start_from_current
                        keyboard.switch = False

                    # tcp
                    tcp_pose = agent.robot.get_tcp_pose()
                    gripper_width = agent.gripper.get_gripper_state() * 0.095 / agent.gripper.max_width
                    obs_tcps = np.stack([tcp_pose])
                    obs_grippers = np.stack([gripper_width])
                    obs_tcps = xyz_rot_transform(obs_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
                    obs_low_dim = np.concatenate((obs_tcps, obs_grippers[..., np.newaxis]), axis = -1)
                    obs_low_dim_normalized = RealWorldDataset._normalize_tcp(obs_low_dim.copy())
                    obs_low_dim = torch.from_numpy(obs_low_dim).float().to(device)
                    obs_low_dim_normalized = torch.from_numpy(obs_low_dim_normalized).float().to(device)
                    print("obs_low_dim_normalized:", obs_low_dim_normalized)

                    # rgbd
                    colors_dict, depths_dict = agent.get_observation()
                    
                    if cfg.policy.name == "DP":
                        from dataset.realworld import pre_process_inputs
                    else:
                        raise NotImplementedError
                    obs_data = pre_process_inputs(colors_dict, depths_dict, device, cfg, obs_low_dim_normalized=obs_low_dim_normalized)

                    # predict
                    pred_raw_action = policy(obs_data, actions = None).squeeze(0).cpu().numpy()
                    
                    if start_from_current:
                        pred_raw_action[...,:2] = pred_raw_action[...,:2] - pred_raw_action[0, :2] + obs_low_dim_normalized[0, :2].cpu().numpy()
                    
                    if "predict_delta_pos" in cfg.policy and cfg.policy.predict_delta_pos:
                        assert cfg.policy.name == "DP"
                        pred_raw_action[...,:3] = pred_raw_action[...,:3] + obs_low_dim_normalized[0, :3].cpu().numpy()
                    # unnormalize predicted actions
                    action = unnormalize_action(pred_raw_action, cfg)
                    action_tcp = action[..., :-1]
                    action_width = action[..., -1]
                    # safety insurance
                    action_tcp[..., :3] = np.clip(action_tcp[..., :3], SAFE_WORKSPACE_MIN + SAFE_EPS, SAFE_WORKSPACE_MAX - SAFE_EPS)
                    # full actions
                    action = np.concatenate([action_tcp, action_width[..., np.newaxis]], axis = -1)
                    
                    print("start tcp:", tcp_pose[:3])
                    print("action:", action)
                    diff = tcp_pose[:3] - action[0,:3]
                    print("diff:", diff)
                    
                    # visualization
                    if args.vis:
                        # open a window to visualize the color image
                        cam_id = agent.cam_ids[0]
                        for cam_id in colors_dict:
                            cv2.imshow(cam_id, cv2.cvtColor(colors_dict[cam_id], cv2.COLOR_RGB2BGR))

                        # plot tcp and action on a top-down 2D map
                        action_map_width = int((SAFE_WORKSPACE_MAX[0] - SAFE_WORKSPACE_MIN[0])*1000)
                        action_map_height = int((SAFE_WORKSPACE_MAX[1] - SAFE_WORKSPACE_MIN[1])*1000)
                        action_map = np.zeros((action_map_height, action_map_width, 3), dtype = np.uint8)
                        
                        begin_color = np.array([0, 255, 255]) # BGR
                        end_color = np.array([255, 0, 255])
                        # plot tcp
                        tcp_x = (tcp_pose[0] - SAFE_WORKSPACE_MIN[0])*1000
                        tcp_y = (tcp_pose[1] - SAFE_WORKSPACE_MIN[1])*1000
                        cv2.circle(action_map, (int(tcp_x), int(tcp_y)), 5, 
                                (0, 0, 255), -1)
                        # plot action
                        action_x = (action[:, 0] - SAFE_WORKSPACE_MIN[0])*1000
                        action_y = (action[:, 1] - SAFE_WORKSPACE_MIN[1])*1000
                        for i in range(len(action)):
                            tmp_color = begin_color * (1 - i/len(action)) + end_color * i/len(action)
                            cv2.circle(action_map, (int(action_x[i]), int(action_y[i])), 3, 
                                    (int(tmp_color[0]), int(tmp_color[1]), int(tmp_color[2])), -1)
                        cv2.imshow('action_map', action_map[:,:,:].transpose(1,0,2))

                        cv2.waitKey(0)
                    
                    time.sleep(0.1)

                # add to ensemble buffer
                ensemble_buffer.add_action(action, t)

            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            if step_action is None:   # no action in the buffer => no movement.
                print("step_action is None")
                continue

            step_tcp = step_action[:-1]
            step_width = step_action[-1]
            print(step_width)

            # send tcp pose to robot
            if args.discretize_rotation:
                rot_steps = discretize_rotation(last_rot, step_tcp[3:], np.pi / 16)
                last_rot = step_tcp[3:]
                for rot in rot_steps:
                    step_tcp[3:] = rot
                    agent.set_tcp_pose(
                        step_tcp, 
                        rotation_rep = "rotation_6d",
                        blocking = True
                    )
            else:
                agent.set_tcp_pose(
                    step_tcp,
                    rotation_rep = "rotation_6d",
                    blocking = True
                )

            # send gripper width to gripper (thresholding to avoid repeating sending signals to gripper)
            if prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD:
                agent.set_gripper_width(
                    step_width, 
                    blocking = prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD * 2 # blocking if the difference is large
                )
                # prev_width = step_width
                prev_width = agent.gripper.get_gripper_state()
                print("set_gripper_width", prev_width)


        if args.vis:
            pass

        if args.record:
            if keyboard.success:
                traj_recorder.success()
                if args.step_record:
                    step_recorder.success()
                    step_recorder.good()
            elif keyboard.discard or keyboard.quit:
                traj_recorder.discard()
                if args.step_record:
                    step_recorder.discard()
            else:
                traj_recorder.fail()
                if args.step_record:
                    step_recorder.fail()
                    step_recorder.bad()

def evaluate(cfg, args):
    pool = multiprocessing.Pool(16)
    keyboard = Keyboard()

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # policy
    print("Loading policy ...")
    if cfg.policy.name == "DP":
        from policy.dp import DP
        policy = DP(**cfg.policy.params).to(device)
    else:
        raise NotImplementedError
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    if args.enable_exploration and cfg.policy.name == "DP":
        policy.action_decoder.enable_exploration = True
        if args.tau1 is not None:
            policy.action_decoder.tau1 = args.tau1
        if args.tau2 is not None:
            policy.action_decoder.tau2 = args.tau2
        if args.noise_scale is not None:
            policy.action_decoder.noise_scale = args.noise_scale
        use_different_noise_scale_at_lowdim = False
        if use_different_noise_scale_at_lowdim:
            policy.action_decoder.noise_scale_for_lowdim = 0.01
            lowdim_size = 0
            for k, v in cfg.policy.params.obs_shape_meta.items():
                if v.type == "low_dim":
                    assert len(v.shape) == 1
                    lowdim_size += v.shape[0]
            print("lowdim_size: ", lowdim_size)
            policy.action_decoder.lowdim_size = lowdim_size

    # evaluation
    agent = Agent(cam_ids=cfg.dataset.params.cam_ids)

    cnt = 0
    while not keyboard.quit:
        keyboard.start = False
        print("============")
        print("Evaluation", cnt)
        print("Press 's' to start")
        print("Press 'q' to quit")
        print("============")
        agent.reset()
        while not keyboard.start and not keyboard.quit:
            time.sleep(0.1)
        if keyboard.quit:
            break
        evaluate_single(cfg, args, agent, policy, keyboard, pool, device)
        if not keyboard.discard:
            cnt += 1

    agent.stop()

    pool.close()
    pool.join()

    # count success rate
    if args.record:
        demos = os.listdir(args.record_path)
        success_cnt = 0
        fail_cnt = 0
        for demo in demos:
            if os.path.exists(os.path.join(args.record_path, demo, "preference.txt")):
                with open(os.path.join(args.record_path, demo, "preference.txt"), "r") as f:
                    pref = f.read().strip()
                    if pref == 'success':
                        success_cnt += 1
                    elif pref == 'fail':
                        fail_cnt += 1

        print("Success: {}, Fail: {}, Total: {}".format(success_cnt, fail_cnt, success_cnt + fail_cnt))
        print("Success rate: {:.2f}%".format(success_cnt / (success_cnt + fail_cnt) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action = 'store', type = str, help = 'config path', required = True)

    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--num_inference_step', action = 'store', type = int, help = 'number of inference query steps', required = False, default = 20)
    parser.add_argument('--max_steps', action = 'store', type = int, help = 'max steps for evaluation', required = False, default = 300)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis', action = 'store_true', help = 'add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')
    parser.add_argument('--record', action = 'store_true')
    parser.add_argument('--record_path', action = 'store', type = str, required = False)
    parser.add_argument('--step_record', action = 'store_true')
    
    parser.add_argument(
        "--enable_exploration",
        action='store_true',
        help="(optional) enable modal-level exploration",
    )
    parser.add_argument(
        "--tau1",
        type=float,
        default=0.0,
        help="(optional) exploration tau1",
    )
    parser.add_argument(
        "--tau2",
        type=float,
        default=1.0,
        help="(optional) exploration tau2",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.5,
        help="(optional) exploration noise scale",
    )

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg = edict(cfg)
    for key in cfg.policy.params:
        if key in vars(args):
            cfg.policy.params[key] = vars(args)[key]
    print(cfg)
    evaluate(cfg, args)