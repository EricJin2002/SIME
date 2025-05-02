# modified from https://github.com/Junxix/S2I/blob/main/utils/realworld_utils.py

import os
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image_files(sub_path):
    file_paths = []
    if os.path.exists(sub_path):
        npy_files = sorted([f for f in os.listdir(sub_path) if f.endswith('.png')])
        for file_name in npy_files:
            file_paths.append(file_name)
    else:
        print(f"The directory {sub_path} does not exist.")
    return file_paths

def load_trajectory_points(sub_path, file_paths):
    trajectory_points = []
    if os.path.exists(sub_path):
        for file_name in file_paths:
            base_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(sub_path, base_name + '.npy')
            data = np.load(file_path)
            first_three_numbers = data[:3]  
            trajectory_points.append(first_three_numbers)
    else:
        print(f"The directory {sub_path} does not exist.")
    return np.array(trajectory_points)

def load_gripper_command(sub_path, file_paths):
    gripper_command = []
    if os.path.exists(sub_path):
        for file_name in file_paths:
            base_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(sub_path, base_name + '.npy')
            data = np.load(file_path)[0]  
            gripper_command.append(data)
    else:
        print(f"The directory {sub_path} does not exist.")
    return gripper_command

def realworld_change_indices(gripper_command):
    differences = np.diff(gripper_command)
    change_indices = [0]
    for i, diff in enumerate(differences):
        if diff != 0:
            if not change_indices:
                change_indices.append(i + 1)
            else:
                current_diff = i + 1 - change_indices[-1]
                if current_diff > 5:
                    change_indices.append(i + 1)
    return change_indices

def realworld_slice(trajectory_points, file_paths, change_indices):
    trajectory_slices, state_slices = [], []
    start_idx = 0
    for idx in change_indices:
        if idx - start_idx >= 10:
            trajectory_slices.append(trajectory_points[start_idx:idx])
            state_slices.append(file_paths[start_idx:idx])
        start_idx = idx
    if len(trajectory_points) - start_idx > 15:
        trajectory_slices.append(trajectory_points[start_idx:])
        state_slices.append(file_paths[start_idx:])
    return trajectory_slices, state_slices

def get_save_mode_factor(save_mode):
    if save_mode == 'lowdim':
        return 0
    elif save_mode == 'image':
        return 0.1
    elif save_mode == 'realworld':
        return 0.1
    elif isinstance(save_mode, (int, float)):
        return save_mode
    else:
        raise ValueError(f"Unknown save_mode: {save_mode}")

def apply_image_filter(image, factor):
    image_array = np.array(image)
    white_image = np.ones_like(image_array) * 255
    new_image_array = (image_array * factor + white_image * (1 - factor)).astype(np.uint8)
    return Image.fromarray(new_image_array)


def load_calibration_data(calib_root_dir):
    tcp_file = os.path.join(calib_root_dir, 'tcp.npy')
    extrinsics_file = os.path.join(calib_root_dir, 'extrinsics.npy')
    intrinsics_file = os.path.join(calib_root_dir, 'intrinsics.npy')

    tcp = np.load(tcp_file)
    extrinsics = np.load(extrinsics_file, allow_pickle=True).item()
    intrinsics = np.load(intrinsics_file, allow_pickle=True).item()

    return tcp, extrinsics, intrinsics

def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = quaternion
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def create_transformation_matrix(position, quaternion):
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T

def compute_extrinsic_matrix(extrinsics, M_cam0433_to_end, M_end_to_base):
    M_cam0433_to_A = extrinsics['043322070878'][0]
    M_cam7506_to_A = extrinsics['750612070851'][0]

    M_cam7506_to_base = M_cam7506_to_A @ np.linalg.inv(M_cam0433_to_A) @ M_cam0433_to_end @ M_end_to_base
    return M_cam7506_to_base

def convert_to_pixel_coordinates(trajectory_points, extrinsic_matrix, camera_matrix):
    translated_points = []
    for point in trajectory_points:
        object_point_world = np.append(point, 1).reshape(-1, 1)
        object_point_camera = extrinsic_matrix @ object_point_world
        object_point_pixel = camera_matrix @ object_point_camera
        object_point_pixel /= object_point_pixel[2]
        pixel_point = np.array([int(object_point_pixel[0]), int(object_point_pixel[1])])
        translated_points.append(pixel_point)
    return np.array(translated_points)    

def translate_points(calib_root_dir, trajectory_points):
    tcp, extrinsics, intrinsics = load_calibration_data(calib_root_dir)

    position = tcp[:3]
    quaternion = tcp[3:]
    M_end_to_base = create_transformation_matrix(position, quaternion)

    M_cam0433_to_end = np.array([[0, -1, 0, 0],
                                 [1, 0, 0, 0.077],
                                 [0, 0, 1, 0.2665],
                                 [0, 0, 0, 1]])

    extrinsic_matrix = compute_extrinsic_matrix(extrinsics, M_cam0433_to_end, M_end_to_base)
    camera_matrix = intrinsics['750612070851']

    return convert_to_pixel_coordinates(trajectory_points, extrinsic_matrix, camera_matrix)

def plot(transformed_points, image, linewidth=5, weights=None, weight_mode="binary", plot_mode="line"):
    plt.clf()
    projected_points = transformed_points[:, :2]
    if plot_mode == "line":
        if weights is None:
            plt.plot(projected_points[:, 0], -projected_points[:, 1], color='red', linewidth=linewidth)
        elif weight_mode == "visualize":
            for i in range(projected_points.shape[0]-1):
                weight = np.clip(weights[i], 0, 1)
                color = (weight, 1 - weight, 1)
                plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color=color, linewidth=linewidth)
        elif weight_mode == "binary":
            for i in range(projected_points.shape[0]-1):
                if weights[i] > 0:
                    # reserved
                    plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color='green', linewidth=linewidth)
                else:
                    # discarded
                    plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color='red', linewidth=linewidth)
        elif weight_mode == "continuous":
            for i in range(projected_points.shape[0]-1):
                weights_sigmoid = 1 / (1 + np.exp(-(weights[i])/0.01))
                color = (1 - weights_sigmoid, weights_sigmoid, 0)
                plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color=color, linewidth=linewidth)
        elif weight_mode == "continuous_v2":
            for i in range(projected_points.shape[0]-1):
                # weights_sigmoid = 1 / (1 + np.exp(-(weights[i]-0.5)/0.5))
                weights_sigmoid = np.clip(weights[i], 0, 1)
                color = (1 - weights_sigmoid, weights_sigmoid, 0)
                plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color=color, linewidth=linewidth)
        elif weight_mode == "continuous_v3":
            for i in range(projected_points.shape[0]-1):
                weights_sigmoid = 1 / (1 + np.exp(-(weights[i]-1)/0.01))
                color = (1 - weights_sigmoid, weights_sigmoid, 0)
                plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color=color, linewidth=linewidth)
        elif weight_mode == "abs":
            weights = np.abs(weights)
            weights = weights / np.max(weights)
            for i in range(projected_points.shape[0]-1):
                color = (1 - weights[i], weights[i], 0)
                plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color=color, linewidth=linewidth)
        elif weight_mode == "customized":
            for i in range(projected_points.shape[0]-1):
                color = 'green'
                for j in range(i, projected_points.shape[0]):
                    if weights[j] < weights[i] - 0.1:
                        color = 'red'
                        break
                plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color=color, linewidth=linewidth)
        elif weight_mode == "segmentation":
            weights = np.clip(weights, 0, 1)
            delta_weights = weights[20:] - weights[:-20]
            seg_start = np.argmax(delta_weights)
            seg_end = seg_start + 20
            print("delta_weight:", delta_weights[seg_start])
            if delta_weights[seg_start] < 0.4:
                return None
            # plot info
            for i in range(projected_points.shape[0]-1):
                color = (
                    1 - weights[i], weights[i], 0
                ) if seg_start <= i < seg_end else 'gray'
                plt.plot(projected_points[i:i+2, 0], -projected_points[i:i+2, 1], color=color, linewidth=linewidth)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")
    elif plot_mode == "point":
        # magma
        # not using weights
        plt.scatter(
            projected_points[:, 0], -projected_points[:, 1], 
            c=np.arange(projected_points.shape[0]), 
            cmap='magma',
            s=linewidth
        )
    else:
        raise ValueError(f"Unknown plot_mode: {plot_mode}")
    plt.axis('off')
    plt.xlim(-0.45, 0.45)
    plt.ylim(-0.5, 0.5)
    plt.gcf().set_size_inches(480/96, 480/96)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=96)
    buf.seek(0)

    image1 = Image.open(buf)
    image.paste(image1, (0, 0), image1)
    buf.close()
    return image

def check_directory_exists(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        sub_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if sub_dirs:
            return os.path.join(directory, sub_dirs[0])
        else:
            print(f"No sub-directories found in {directory}.")
    else:
        print(f"The directory {directory} does not exist.")
    return None

def load_demo_files(root_dir, subdirs, ind):
    CAMERA_NAME = 'cam_750612070851'
    color_path = os.path.join(root_dir, subdirs[ind], CAMERA_NAME, 'color')
    tcp_path = os.path.join(root_dir, subdirs[ind], CAMERA_NAME, 'tcp')
    gripper_command_path = os.path.join(root_dir, subdirs[ind], CAMERA_NAME, 'gripper_command')
    
    file_paths = load_image_files(color_path)
    trajectory_points = load_trajectory_points(tcp_path, file_paths)
    gripper_command = load_gripper_command(gripper_command_path, file_paths)
    
    return file_paths, trajectory_points, gripper_command