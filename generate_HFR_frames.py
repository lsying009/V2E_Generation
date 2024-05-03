import os
from utils.video_reader import VideoRenderer, VideoInterpolator
import argparse
import torch

if __name__ == "__main__":
    '''Generate and save video sequences 
    1. From static images using motion config files 
        --path_to_config_files is required
        --frame_rate: if frame_rate<0, apply adaptive frame rate according to motion
    
    2. From existing low frame rate video sequences
        --path_to_LFR_data is requried

    '''
    parser = argparse.ArgumentParser(description='Generate intensity video sequences')
    
    # Parameters for VideoRenderer at fix frame rate (No upsampling)
    parser.add_argument('--path_to_config_files', type=str, default=None, 
        help="Required for video rendering based on static frame and motion config")
    parser.add_argument('--frame_rate', type=float, 
                        help='Frame rate of video renderer, if frame_rate<0, apply adaptive frame rate according to motion', 
                        default=None)
    
    
    # Parameters for VideoInterpolator (Save high frame rate video sequences at adaptive frame rate)
    parser.add_argument('--path_to_LFR_data', type=str, default=None,
                        help='Path to read low frame rate video sequence')
    
    # Common parameters
    parser.add_argument('--path_to_save_sequences', type=str, help='Path to save sequences dir', \
        default=None)
    parser.add_argument('--image_height', type=int, default=180)
    parser.add_argument('--image_width', type=int, default=240)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--max_num_frames', type=int, default=-1, help='Maximum number of frames to save per sequence. if =-1 save all generated frames')
    
    # Replace root path in motion configs, in case the root path changes
    parser.add_argument('--old_root_path', type=str, help='Old root dir to read frames', \
        default=None)
    parser.add_argument('--new_root_path', type=str, help='New root dir to read frames', \
        default=None)
    
    
    
    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('device: ', device)
    
    # # # ##################### save video sequences from static images based on motion configs ################
    if args.path_to_config_files is not None:
        path_to_config_files = []
        for file_name in os.listdir(args.path_to_config_files):
            ############### filter certain sequences
            if file_name.split('.')[-1]=='txt': # and int(file_name.split('.')[0].split('_')[-1])>=600:
                path_to_config_files.append(os.path.join(args.path_to_config_files, file_name))
        path_to_config_files.sort()

        frame_rate = args.frame_rate
        video_renderer = VideoRenderer(image_dim=[args.image_height,args.image_width], padding_size=[4,4], is_save_frames=True, replace_path_pair=[args.old_root_path, args.new_root_path] if args.new_root_path else None)
        
        for seq_id in range(args.start_id, len(path_to_config_files)):
            path_to_config_file = path_to_config_files[seq_id]
            video_renderer.initialize(path_to_config_file, -1, frame_rate, save_path=args.path_to_save_sequences)
            if args.max_num_frames > 0:
                max_num_frames = min(video_renderer.num_frames, args.max_num_frames) # limit the max num of frames
            else:
                max_num_frames = video_renderer.num_frames
            for i in range(max_num_frames):
                frame, t = video_renderer.update_frame()

            
    # ################### save HFR video sequences from LFR frames using adaptive upsampling  ############################
    if args.path_to_LFR_data is not None:
        path_to_org_sequences = []
        for seq_name in os.listdir(args.path_to_LFR_data):
            if os.path.isdir(os.path.join(args.path_to_LFR_data, seq_name)) and seq_name.split('_')[0]=='sequence': # and int(seq_name.split('_')[1])>=600:
                path_to_org_sequences.append(os.path.join(args.path_to_LFR_data, seq_name))
                # path_to_sequences.append(os.path.join(path_to_data_files, seq_name, 'frames'))
        path_to_org_sequences.sort()
    
        video_renderer = VideoInterpolator(image_dim=[args.image_height,args.image_width], padding_size=[4,4], device=device, is_save_frames=True)
        for seq_id, path_to_org_sequence_folder in enumerate(path_to_org_sequences):
            video_renderer.initialize(path_to_org_sequence_folder, num_load_frames=-1, save_path=args.path_to_save_sequences)
            for i in range(video_renderer.num_frames):
                frame, t = video_renderer.update_frame()    

