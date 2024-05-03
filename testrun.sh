#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=5gb:ngpus=1
#PBS -lwalltime=00:40:00
#24 10gb

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate py311
  
## Verify install:
python -c "import torch;print(torch.cuda.is_available())"

root_path='/rds/general/user/sl220/home/data/'
data_mode='testrun'

# python generate_motion_config.py \
--generator_config 'motion_configs/fast_motions.json' \
--output_path ${root_path}${data_mode}/configs_testrun \
--num_sequences 5 \
--seq_start_id 0 \
--image_width 240 \
--image_height 180 \
--scene_duration 2 \
# --is_bg_only \

# Generate frames at fixed frame rate
# python generate_HFR_frames.py \
--path_to_config_files  ${root_path}${data_mode}/configs_${data_mode} \
--path_to_save_sequences  ${root_path}${data_mode}_lfr \
--image_height 180 \
--image_width 240 \
--frame_rate 30 \
--max_num_frames 100 \
# --old_root_path /home/sl220/Documents/data/EVData/ \
# --new_root_path /rds/general/user/sl220/home/data/ \
# --start_id 500 \
# --max_num_frames 750 for train

# No longer use
# Video interpolation at max mag of optical flow 1
python generate_HFR_frames.py \
--image_height 180 \
--image_width 240 \
--path_to_LFR_data ${root_path}${data_mode}_lfr \
--path_to_save_sequences ${root_path}${data_mode}_hfr \

# Generate events
#--path_to_intensity_data '/home/sl220/Documents/data/upsampled-train/' \
# python run_v2e.py \
--path_to_intensity_data ${root_path}${data_mode} \
--path_to_save_data ${root_path}${data_mode} \
--start_id 0 \
--end_id 1500 \
--cutoff_hz 200 \
--shot_noise_rate_hz 1 \
--refractory_period_s 0.001 \

# Make training/testing txt for data loader
# python generate_txt.py \
--path_to_data ${root_path}${data_mode} \
--txt_name 'load_e2v.txt' \
--path_to_config_files ${root_path}${data_mode}/configs_testrun

# python make_train_txt.py \
# --path_to_data '/home/sl220/Documents/data/COCO_test_data_flow/' \
# --txt_name 'test_e2v.txt' \



##  fuse events with limit number of events, and generate flow
# python fuse_upsamp_gen_flow.py \
--data_txt_file ${root_path}${data_mode}/load_e2v.txt \
--path_to_read_data ${root_path}${data_mode} \
--path_to_save_data ${root_path}adaptive_${data_mode}/ \
--image_dim 180 240 \
--num_events 15000 \
--len_sequence 10 \
--max_n_per_seq 5 \
--is_gt_flow \
# len_sequence = 15
# --old_root_path /home/sl220/Documents/data/EVData/ \
# --new_root_path /rds/general/user/sl220/home/data/ \
# --is_gt_flow \
# --is_gt_flow \
# --is_save_warped_events \


# write related txt file
# python generate_txt.py \
--path_to_data ${root_path}adaptive_${data_mode}/ \
--txt_name 'train_e2v.txt' \
--is_fused_data \
--is_flow \
# --is_gt_flow \
# --is_warped_events \

