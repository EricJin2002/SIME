import glob
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def clean_data(dataset, cam_ids):
    demos = glob.glob(f'{dataset}/*')
    print(demos)
    rm_cnt = 0
    not_rm_cnt = 0
    for demo in demos:
        # print(demo)
        for cam in cam_ids:
            colors = os.path.join(demo, f'cam_{cam}','color')
            depths = os.path.join(demo, f'cam_{cam}','depth')
            frame_ids = [
                int(os.path.splitext(x)[0]) 
                for x in sorted(os.listdir(colors))
            ]
            frame_ids = sorted(frame_ids)
            last_tcp = None
            last_width = None
            for i, id in enumerate(frame_ids):
                tcp_path = os.path.join(demo, 'tcp', f'{id}.npy')
                width_path = os.path.join(demo, 'gripper_command', f'{id}.npy')

                curr_tcp = np.load(tcp_path)
                curr_width = np.load(width_path)
                # print(curr_tcp)
                if last_tcp is not None:
                    xyz_diff = np.linalg.norm(curr_tcp[:3] - last_tcp[:3])
                    # print(np.sum(np.power(curr_tcp[3:], 2))) # quaternion
                    # diff between two quaternions
                    curr_quat = R.from_quat([*curr_tcp[4:], curr_tcp[3]])
                    last_quat = R.from_quat([*last_tcp[4:], last_tcp[3]])
                    quat_diff = curr_quat.inv() * last_quat
                    rot_diff = np.linalg.norm(quat_diff.as_rotvec())
                    width_diff = np.abs(curr_width - last_width)
                    # print(width_diff)
                    # print(xyz_diff, rot_diff)

                    if xyz_diff < 0.005 and rot_diff < np.pi / 12.0 and width_diff < 100 and i < len(frame_ids) - 1:
                        print('Remove', os.path.join(colors, f'{id}.png'))    
                        print('Remove', os.path.join(depths, f'{id}.png')) 
                        os.remove(os.path.join(colors, f'{id}.png'))
                        os.remove(os.path.join(depths, f'{id}.png'))
                        rm_cnt += 1
                        continue
                last_tcp = curr_tcp
                last_width = curr_width
                not_rm_cnt += 1
    print(rm_cnt, not_rm_cnt)
    print('Done!')

if __name__ == '__main__':
    cam_ids = ['038522063145', '104422070044']

    for dataset in glob.glob('/data/jinyang/realworld/0222_two_cups/normal/baseline/*'):
        clean_data(dataset, cam_ids)
    for dataset in glob.glob('/data/jinyang/realworld/0222_two_cups/normal/sime/*'):
        clean_data(dataset, cam_ids)