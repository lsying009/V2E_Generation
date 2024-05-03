# Video-to-events Generation

This repository is used to generate data 

This Python Torch code implements video-to-events (V2E) generation using the [v2e simulator](https://github.com/SensorsINI/v2e) to create training/testing data for events-to-video (E2V) reconstruction. The repository [V2E2V](https://github.com/lsying009/V2E2V) leverages this code for data generation. This repository is related to the arxiv paper [Enhanced Event-Based Video Reconstruction with Motion Compensation](https://arxiv.org/abs/2403.11961), the TPAMI paper [Sensing Diversity and Sparsity Models for Event Generation and Video Reconstruction from Events](https://ieeexplore.ieee.org/abstract/document/10130595) and the preliminary work in the ICASSP paper [Convolutional ISTA Network with Temporal Consistency Constraints for Video Reconstruction from Event Cameras](https://ieeexplore.ieee.org/abstract/document/9746331).


This code generates high frame rate (HFR) video sequences, corresponding events and optical flow from existing low frame rate (LFR) video sequences or static frames. 

If you would like to generate data from static frame, follow these steps:

1. **Genetate config files**

Generate config files for each sequence with arbitrary affine transforms.  Random number of foregrounds move in front of the background. If ```is_bg_only``` is enabled, only the background will be included without any foreground objects. Codes are based on [esim_config_generator](https://github.com/TimoStoff/esim_config_generator)

```
        python generate_motion_config.py \
        --generator_config $path_to_motion_configs \
        --output_path $path_to_save_configs_for_each_sequence \
        --num_sequences 5 \
        --seq_start_id 0 \
        --image_width 240 \
        --image_height 180 \
        --scene_duration 2 \
        --is_bg_only \
```

2. **Generate video sequences with motions according to config files**

Run generate_HFR_frames.py to generate HFR video sequences. If ```frame_rate>0```, images are generated at the specificed frame rate; if ```frame_rate<=0```, an adaptive frame rate will be applied based on motion, where the maximum motion displacement is 1 pixel. 

        python generate_HFR_frames.py \
        --path_to_config_files  $path_to_config_files \
        --path_to_save_sequences  $path_to_save_sequences \
        --frame_rate -1 \
        --image_height $image_height \
        --image_width $image_width \

If the motion is unknown and you have LFR video seuqences, then excute:

        python generate_HFR_frames.py \
        --path_to_LFR_data  $path_to_LFR_data \
        --path_to_save_sequences  $path_to_save_sequences \
        --image_height $image_height \
        --image_width $image_width \

3. **Generate events based on HFR videos**

Events are generated using [v2e simulator](https://github.com/SensorsINI/v2e) in the folder ```v2ecore```.

        python run_v2e.py \
        --path_to_intensity_data $path_to_intensity_data \
        --path_to_save_data $path_to_save_data \
        --start_id 0 \
        --end_id $end_id \


4. **Generate .txt file for the following steps**
Create .txt file to read generated frames/events for the following steps. If flow need to be generated based on motion, ```path_to_config_files``` is required.

        python generate_txt.py \
        --path_to_data $path_to_data \
        --txt_name $txt_name \
        --path_to_config_files $path_to_config_files

5. **Combine events and generate optical flow**

Combine events with a fixed event count, and generate the corresponding flow. 
If ```is_gt_flow``` is enabled and motion config files are known (specified in ```data_txt_file```), flow is generated based on motion; otherwise, flow is generated using FlowNet in [adaptive upsampling](https://github.com/uzh-rpg/rpg_vid2e/tree/master/upsampling) based on [Super-Slomo](https://jianghz.me/projects/superslomo/). The checkpoint of Super-Slomo model is downloaded from [here](https://drive.google.com/file/d/1YL2EnX0MsrH_5_PjhDr__c6NaT_y8I7Z/view?usp=sharing). 


Each scene containing $M$ frames is divided into $N\times L$ segments, with $N$ sequences consisting of $L$ frames each. ```len_sequence``` is the number of frames for each sequence $L$, and ```max_n_per_seq``` represents the maximum value of $N$ to limit the number of sequences for each scene. If these values are negative, $L$ and $N$ are adaptive, determined by the number of events and the length of the original sequence. 

        python fuse_upsamp_gen_flow.py \
        --data_txt_file $data_txt_file \
        --path_to_read_data $path_to_read_data \
        --path_to_save_data $path_to_save_data \
        --image_dim $image_dim \
        --num_events $num_events \
        --len_sequence $len_sequence \
        --max_n_per_seq 1 \
        --is_gt_flow \

6. **Generate train.txt**

Create .txt file for training. ```is_fused_data``` should be enabled.

        python generate_txt.py \
        --path_to_data $path_to_data \
        --txt_name 'train.txt' \
        --is_fused_data \
        --is_flow \


## Acknowledgement

This code is based on [v2e simulator](https://github.com/SensorsINI/v2e) and [esim_config_generator](https://github.com/TimoStoff/esim_config_generator)

## Citation
If you use any of this code, please cite the publications as follows:
```bibtex
    @misc{liu2024enhanced,
      title={Enhanced Event-Based Video Reconstruction with Motion Compensation}, 
      author={Siying Liu and Pier Luigi Dragotti},
      year={2024},
      eprint={2403.11961},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```bibtex
    @article{liu_sensing_2023,  
    title={Sensing Diversity and Sparsity Models for Event Generation and Video Reconstruction from Events},   
    author={Liu, Siying and Dragotti, Pier Luigi},  
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
    year={2023},  
    pages={1-16},  
    publisher={IEEE}. 
    doi={10.1109/TPAMI.2023.3278940}. 
    }
```
```bibtex
    @inproceedings{liu_convolutional_2022,  
    title={Convolutional ISTA Network with Temporal Consistency Constraints for Video Reconstruction from Event Cameras},  
    author={Liu, Siying and Alexandru, Roxana and Dragotti, Pier Luigi},  
    booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
    pages={1935--1939},  
    year={2022},  
    organization={IEEE}. 
    doi={10.1109/ICASSP43922.2022.9746331}. 
    }
```

