#!/bin/bash

# This script holds the commands that were used to go from raw robosuite demo.hdf5 files
# to our processed low-dim and image hdf5 files.

BASE_DATASET_DIR="datasets"
echo "Using base dataset directory: $BASE_DATASET_DIR"


### NOTE: we use done-mode 0 (dones on task success) ###

### ph ###


# lift - ph
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/lift/ph/low_dim_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# can - ph
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/can/ph/low_dim_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# square - ph
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/square/ph/low_dim_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# transport - ph
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/transport/ph/low_dim_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand --camera_height 84 --camera_width 84

# tool hang - ph
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/tool_hang/ph/low_dim_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names sideview robot0_eye_in_hand --camera_height 240 --camera_width 240



# # lift - ph
# python dataset_states_to_obs.py --done_mode 0 \
# --dataset $BASE_DATASET_DIR/lift/ph/low_dim_v141_abs_6drot.hdf5 \
# --output_name image_v141_abs_6drot.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# # can - ph
# python dataset_states_to_obs.py --done_mode 0 \
# --dataset $BASE_DATASET_DIR/can/ph/low_dim_v141_abs_6drot.hdf5 \
# --output_name image_v141_abs_6drot.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# # square - ph
# python dataset_states_to_obs.py --done_mode 0 \
# --dataset $BASE_DATASET_DIR/square/ph/low_dim_v141_abs_6drot.hdf5 \
# --output_name image_v141_abs_6drot.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# # transport - ph
# python dataset_states_to_obs.py --done_mode 0 \
# --dataset $BASE_DATASET_DIR/transport/ph/low_dim_v141_abs_6drot.hdf5 \
# --output_name image_v141_abs_6drot.hdf5 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand --camera_height 84 --camera_width 84

# # tool hang - ph
# python dataset_states_to_obs.py --done_mode 0 \
# --dataset $BASE_DATASET_DIR/tool_hang/ph/low_dim_v141.hdf5 \
# --output_name image_v141_abs_6drot.hdf5 --camera_names sideview robot0_eye_in_hand --camera_height 240 --camera_width 240
