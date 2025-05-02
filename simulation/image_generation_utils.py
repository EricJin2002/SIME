# modified from https://github.com/Junxix/S2I/blob/main/utils/image_generation.py

import os
import cv2
import numpy as np
from PIL import Image

from realworld_utils import *
import robosuite.utils.transform_utils as T

def get_camera_extrinsic_matrix(sim, camera_name):
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = T.make_pose(camera_pos, camera_rot)

    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    R = R @ camera_axis_correction
    return R

class TrajectoryRenderer:
    def __init__(self, env, camera_name, save_mode = False, calib_dir=None, root_dir=None):
        self.save_mode = save_mode
        if self.save_mode == 'realworld':
            self.calib_dir = calib_dir
            self.root_dir = root_dir
        else:
            self.env = env
            self.camera_name = camera_name

            # self.extrinsic_matrix = env.get_camera_extrinsic_matrix(camera_name)
            self.extrinsic_matrix = get_camera_extrinsic_matrix(env.env.sim, camera_name)

            print(self.extrinsic_matrix)

            self.camera_position = self.extrinsic_matrix[:3, 3]
            self.camera_rotation = self.extrinsic_matrix[:3, :3]


    def render_trajectory_image(self, trajectory_points, ind, samples, save_mode, rotate=False, state_slices=None, **kwargs):
        if self.save_mode == 'realworld':
            if rotate:
                trajectory_points = self.rotate_trajectory(trajectory_points, ind)
            transformed_points = translate_points(self.calib_dir, trajectory_points)
            self.realworld_save_and_append_images(transformed_points, samples, ind, state_slices[0], rotate)
        else:
            state_slices is not None and self.env.reset_to(state_slices[0])
            frame = self.env.render(mode="rgb_array", height=480, width=480, camera_name=self.camera_name)
            image2 = Image.fromarray(frame)
            factor = get_save_mode_factor(save_mode)
            image_array = apply_image_filter(image2, factor)

            if rotate:
                trajectory_points = self.rotate_trajectory(trajectory_points, ind)
            transformed_points = np.dot(trajectory_points - self.camera_position, self.camera_rotation)

            self.save_and_append_images(transformed_points, samples, ind, image_array, rotate, **kwargs)

    def rotate_trajectory(self, trajectory_points, i):
        start_point, end_point, middle_points = trajectory_points[0], trajectory_points[-1], trajectory_points[1:-1]
        axis = (end_point - start_point) / np.linalg.norm(end_point - start_point)
        angle = np.deg2rad(i * 360 / 30) 
        rotation_matrix = self.get_rotation_matrix(axis, angle)

        rotated_middle_points = np.dot(middle_points - start_point, rotation_matrix.T) + start_point
        return np.vstack([start_point, rotated_middle_points, end_point])

    def get_rotation_matrix(self, axis, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        return np.array([
            [cos_angle + axis[0]**2 * (1 - cos_angle), axis[0]*axis[1]*(1 - cos_angle) - axis[2]*sin_angle, axis[0]*axis[2]*(1 - cos_angle) + axis[1]*sin_angle],
            [axis[1]*axis[0]*(1 - cos_angle) + axis[2]*sin_angle, cos_angle + axis[1]**2 * (1 - cos_angle), axis[1]*axis[2]*(1 - cos_angle) - axis[0]*sin_angle],
            [axis[2]*axis[0]*(1 - cos_angle) - axis[1]*sin_angle, axis[2]*axis[1]*(1 - cos_angle) + axis[0]*sin_angle, cos_angle + axis[2]**2 * (1 - cos_angle)]
        ])

    def realworld_save_and_append_images(self, transformed_points, samples, ind, image_path, rotate):
        img_path = os.path.join(self.root_dir, image_path)            
        image = cv2.imread(img_path)
        factor = get_save_mode_factor(self.save_mode)
        image = apply_image_filter(image, factor)
        image = np.array(image)

        prev_point = None
        for point in transformed_points[:]:
            cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
            if prev_point is not None:
                cv2.line(image, prev_point, point, (0, 0, 255), thickness=4)
            prev_point = point
        # save_path = './tmp_realworld/marked_image{}.jpg'.format(ind)  
        # cv2.imwrite(save_path, image)
        samples['images'].append(np.array(image))
        samples['labels'].append(1 if rotate else 0)

    def save_and_append_images(self, transformed_points, samples, ind, image_array, rotate, **kwargs):
        image_array = plot(transformed_points, image_array, **kwargs)
        # image_array.save(f'./tmp/image_com_{ind}.png')

        samples['images'].append(np.array(image_array))
        samples['labels'].append(1 if rotate else 0)


class TrajectoryNoiseGenerator:
    def __init__(self, trajectory_points):
        self.trajectory_points = trajectory_points

    def render_one_point(self):
        start_index = np.random.randint(0, len(self.trajectory_points) - 10)
        end_index = start_index + 1

        sub_trajectory = self.trajectory_points[start_index:end_index]
        noisy_sub_trajectory = self.add_noise(sub_trajectory, scale_range=(0.02, 0.05))

        noise_trajectory = self.trajectory_points.copy()
        noise_trajectory[start_index:end_index] = noisy_sub_trajectory
        self.remove_adjacent_points(noise_trajectory, start_index)
        return noise_trajectory

    def add_noise(self, sub_trajectory, scale_range):
        noise_scale = np.random.uniform(*scale_range)
        noise = noise_scale * np.random.randn(sub_trajectory.shape[0], sub_trajectory.shape[1])
        return sub_trajectory + noise

    def remove_adjacent_points(self, noise_trajectory, start_index):
        if start_index > 1:
            for i in range(1, 4):
                noise_trajectory = np.delete(noise_trajectory, start_index - i, axis=0)

    def render_one_point_circle(self):
        start_index = np.random.randint(0, min(len(self.trajectory_points) - 10, len(self.trajectory_points)))
        end_index = start_index + np.random.randint(5, 10)

        sub_trajectory = self.trajectory_points[start_index:end_index]
        noisy_sub_trajectory = self.add_noise(sub_trajectory, scale_range=(0.02, 0.04))

        inserted_indices = self.get_inserted_indices(start_index, end_index, num_points_range=(10, 20))
        for i, index in enumerate(inserted_indices):
            self.trajectory_points = np.insert(self.trajectory_points, index + i, sub_trajectory[0] + np.random.uniform(0.04, 0.06)*np.random.randn(), axis=0)
        return self.trajectory_points

    def get_inserted_indices(self, start_index, end_index, num_points_range):
        num_inserted_points = np.random.randint(*num_points_range)
        inserted_indices = np.random.randint(start_index, end_index, size=num_inserted_points)
        inserted_indices.sort()
        return inserted_indices

    def render_series(self):
        start_index = np.random.randint(0, max(1, len(self.trajectory_points) - 20))
        end_index = start_index + np.random.randint(5, 10)
        sub_trajectory = self.trajectory_points[start_index:end_index]

        noisy_sub_trajectory = self.add_noise(sub_trajectory, scale_range=(0.03, 0.06))
        noise_trajectory = self.trajectory_points.copy()
        noise_trajectory[start_index:end_index] = noisy_sub_trajectory
        return noise_trajectory


class TrajectoryGenerator:
    def __init__(self, trajectory_points):
        self.trajectory_points = trajectory_points
        self.noise_generator = TrajectoryNoiseGenerator(trajectory_points)

    def generate_negative_trajectory_points(self):
        num_one_point = np.random.randint(0, 8)
        noise_trajectory = self.trajectory_points.copy()

        for _ in range(num_one_point):
            noise_trajectory = self.noise_generator.render_one_point()

        if num_one_point == 0:
            one_point_circle_flag = 1
        else:
            one_point_circle_flag = np.random.randint(2)

        if one_point_circle_flag:
            noise_trajectory = self.noise_generator.render_one_point_circle()

        if not one_point_circle_flag and np.random.randint(2):
            noise_trajectory = self.noise_generator.render_series()

        return noise_trajectory


class PictureGenerator:
    def __init__(self, renderer, samples, save_mode):
        self.renderer = renderer
        self.samples = samples
        self.save_mode = save_mode

    def generate_positive_picture(self, trajectory_points, state_slices, ind, num_images=30):
        for i in range(num_images):
            self.renderer.render_trajectory_image(trajectory_points, i, self.samples, self.save_mode, rotate=True, state_slices=state_slices)
            # This part can be expanded if needed

    def generate_negative_picture(self, trajectory_points, state_slices, ind, num_images):
        for i in range(num_images):
            noise_trajectory = TrajectoryGenerator(trajectory_points).generate_negative_trajectory_points()
            self.renderer.render_trajectory_image(noise_trajectory, i, self.samples, self.save_mode, rotate=False, state_slices=state_slices)

