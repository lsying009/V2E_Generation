import os
import numpy as np
import argparse
from pathlib import Path

def generate_data_txt(data_dir: str, txt_name: str, num_frames: int, step: int):
    '''Generate txt to read HFR data, every N HFR frames
        num_frames --  number of frames for each per reconstruction, if num_frames=2, the same as generate_data_flow_txt()
        step -- stride step
    '''
    
    txt_file = os.path.join(data_dir, txt_name)
    with open(txt_file, 'w') as f:
        print('Write training TXT ... \n')
        seq_names = [ fn for fn in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir,fn)) and 'seq' in fn)] #[f for f in os.listdir(data_dir) if os.path.isdir(f)]
        seq_names.sort()
        video_idx = 0

        for seq_id, seq_name in enumerate(seq_names):
    
            path_to_seq = os.path.join(data_dir, seq_name)
            # img_dirpath = path_to_seq
            
            path_to_frames = []
            for root, dirs, files in os.walk(path_to_seq):
                for file_name in files:
                    if file_name.split('.')[-1] in ['jpg','png']:
                        # seq_00000/frames/ or seq_00000/
                        save_dir_name = seq_name if root.split('/')[-1] == seq_name else os.path.join(seq_name, root.split('/')[-1])
                        path_to_frames.append(os.path.join(save_dir_name, file_name))
                    elif file_name in ['timestamps.txt', 'images.txt', 'timestamp.txt']:#.split('.')[-1] in ['txt']:
                        timestamps_file = os.path.join(root, file_name)
            path_to_frames.sort()

            timestamps = []
            with open(timestamps_file, 'r') as ft:   
                for line in ft.readlines():
                    _, t = line.split()
                    timestamps.append(t)
                    

            events_dir = os.path.join(path_to_seq, 'events')
            eventfile_names = [f for f in os.listdir(events_dir) if Path(f).suffix.lower() in ['.npz']]
            eventfile_names.sort()
            path_to_events = [os.path.join(seq_name, 'events', f) for f in eventfile_names]
            
            if step==1:
                num_events_list = []
                for path_to_cur_events in path_to_events:
                    path_to_cur_events = os.path.join(data_dir, path_to_cur_events)
                    event_window = np.load(path_to_cur_events)
                    num_events = len(event_window["t"])
                    num_events_list.append(num_events)

            if len(path_to_events) < len(path_to_frames)-1:
                print(seq_name)
                continue
                
            for frame_idx in range(0, len(path_to_frames)-num_frames+1, step):
                path_to_all_frames = ' '.join( path_to_frames[frame_idx+i] for i in range(num_frames))
                all_ts = ' '.join(timestamps[frame_idx+i] for i in range(num_frames))
                
                path_to_all_events = ' '.join( path_to_events[frame_idx+i] for i in range(num_frames-1))
                # events = np.load(os.path.join(data_dir, path_to_all_events))
                # print(num_events_list[frame_idx], len(events["t"]), path_to_all_events)
                if step != 1:
                    f.write(str(video_idx)+' '+ all_ts +' '+ \
                        path_to_all_frames+' '+path_to_all_events+'\n')
                else:
                    all_num_events = ' '.join(str(num_events_list[frame_idx+i]) for i in range(num_frames-1))
                    f.write(str(video_idx)+' '+all_num_events+' '+ all_ts +' '+ \
                        path_to_all_frames+' '+path_to_all_events+'\n')

            video_idx += 1
    print('Finished! \n')
    f.close()
   
    
def generate_data_motion_txt(data_dir: str,  config_dir: str, txt_name: str):
    '''Generate txt to read HFR data with motion configs'''
    txt_file = os.path.join(data_dir, txt_name)
    with open(txt_file, 'w') as f:
        print('Write training TXT ... \n')
        seq_names = [ fn for fn in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir,fn)) and 'seq' in fn)] #[f for f in os.listdir(data_dir) if os.path.isdir(f)]
        seq_names.sort()
        video_idx = 0
        
        path_to_config_files = []
        for root, dirs, files in os.walk(config_dir):
                for file_name in files:
                    if file_name.split('.')[-1] in ['txt']:
                        path_to_config_files.append(os.path.join(root, file_name))
        path_to_config_files.sort()
                        
        for seq_id, seq_name in enumerate(seq_names):
    
            path_to_seq = os.path.join(data_dir, seq_name)
            # img_dirpath = path_to_seq
            
            path_to_frames = []
            for root, dirs, files in os.walk(path_to_seq):
                for file_name in files:
                    if file_name.split('.')[-1] in ['jpg','png']:
                        # seq_00000/frames/ or seq_00000/
                        save_dir_name = seq_name if root.split('/')[-1] == seq_name else os.path.join(seq_name, root.split('/')[-1])
                        path_to_frames.append(os.path.join(save_dir_name, file_name))
                    elif file_name in ['timestamps.txt', 'images.txt', 'timestamp.txt']:#.split('.')[-1] in ['txt']:
                        timestamps_file = os.path.join(root, file_name)
            path_to_frames.sort()

            timestamps = []
            with open(timestamps_file, 'r') as ft:   
                for line in ft.readlines():
                    _, t = line.split()
                    timestamps.append(t)
                    

            events_dir = os.path.join(path_to_seq, 'events')
            eventfile_names = [f for f in os.listdir(events_dir) if Path(f).suffix.lower() in ['.npz']]
            eventfile_names.sort()
            path_to_events = [os.path.join(seq_name, 'events', f) for f in eventfile_names]
            
            num_events_list = []
            for path_to_cur_events in path_to_events:
                path_to_cur_events = os.path.join(data_dir, path_to_cur_events)
                event_window = np.load(path_to_cur_events)
                num_events = len(event_window["t"])
                num_events_list.append(num_events)

            if len(path_to_events) < len(path_to_frames)-1:
                print(seq_name)
                continue
                
            for frame_idx in range(0, len(path_to_frames)-1):
                path_to_all_frames = ' '.join( path_to_frames[frame_idx+i] for i in range(2))
                all_ts = ' '.join(timestamps[frame_idx+i] for i in range(2))
                
                path_to_all_events = path_to_events[frame_idx]

                all_num_events = str(num_events_list[frame_idx])
                
                f.write(str(video_idx)+' '+all_num_events+' '+ all_ts +' '+ \
                    path_to_all_frames+' '+path_to_all_events+' '+ path_to_config_files[seq_id]+'\n')

            video_idx += 1
    print('Finished! \n')
    f.close()
   

def generate_data_flow_txt(data_dir: str, txt_name: str, is_flow:bool):
    '''Generate txt to read HFR data with flow
    is_flow: load events + frame + flow for each reconstruction
    '''
    txt_file = os.path.join(data_dir, txt_name)
    with open(txt_file, 'w') as f:
        print('Write training TXT ... \n')
        seq_names = [ fn for fn in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir,fn)) and 'seq' in fn)] #[f for f in os.listdir(data_dir) if os.path.isdir(f)]
        seq_names.sort()
        video_idx = 0

        for seq_id, seq_name in enumerate(seq_names):
    
            path_to_seq = os.path.join(data_dir, seq_name)
            img_dirpath = os.path.join(path_to_seq, 'frames')
            
            
            if not os.path.isdir(img_dirpath):
                continue

            timestamps_file = os.path.join(img_dirpath, 'timestamps.txt')
            timestamps = []
            with open(timestamps_file, 'r') as ft:   
                for line in ft.readlines():
                    _, t = line.split()
                    timestamps.append(t)
                    
            
            imgfile_names = [f for f in os.listdir(img_dirpath) if Path(f).suffix.lower() in ['.png', '.jpg']]
            imgfile_names.sort()
            path_to_frames = [os.path.join(seq_name,'frames', f) for f in imgfile_names] ###################
            

            event_folder_name = 'events'
            events_dir = os.path.join(path_to_seq, event_folder_name)
            eventfile_names = [f for f in os.listdir(events_dir) if Path(f).suffix.lower() in ['.npz']]
            eventfile_names.sort()
            path_to_events = [os.path.join(seq_name, event_folder_name, f) for f in eventfile_names]
            
            num_events_list = []
            for path_to_cur_events in path_to_events:
                path_to_cur_events = os.path.join(data_dir, path_to_cur_events)
                event_window = np.load(path_to_cur_events)
                num_events = len(event_window["t"])
                num_events_list.append(num_events)

            if len(path_to_events) < len(path_to_frames)-1:
                print(seq_name)
                continue

            if is_flow:
                flow_dirpath = os.path.join(path_to_seq, 'flow')
                flowfile_names = [f for f in os.listdir(flow_dirpath) if Path(f).suffix.lower() in ['.npz']]
                flowfile_names.sort()
                path_to_flow = [os.path.join(seq_name, 'flow', f) for f in flowfile_names]

                
            for frame_idx in range(len(path_to_frames)-1):
                path_to_all_frames = ' '.join( path_to_frames[frame_idx+i] for i in range(2))
                all_ts = ' '.join(timestamps[frame_idx+i] for i in range(2))

                path_to_cur_events = path_to_events[frame_idx]
                num_events = str(num_events_list[frame_idx])
                if not is_flow:
                    f.write(str(video_idx)+' '+num_events+' '+ all_ts +' '+ \
                        path_to_all_frames+' '+path_to_cur_events+'\n')
                else:
                    path_to_cur_flow = path_to_flow[frame_idx]
                    f.write(str(video_idx)+' '+num_events+' '+ all_ts +' '+ \
                        path_to_all_frames+' '+path_to_cur_events+' '+path_to_cur_flow +'\n')
                
            video_idx += 1
    print('Finished! \n')
    f.close()
  


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate train.txt')
    parser.add_argument('--path_to_data', type=str, help='Path to data with frames and events')
    parser.add_argument('--path_to_config_files', type=str, default=None)
    parser.add_argument('--txt_name', type=str)
    parser.add_argument('--is_fused_data', action='store_true', default=False,
                        help='If the data is fused at fixed number of events/frame rate or something else')

    parser.add_argument('--is_flow', action='store_true', default=False)
    
    args = parser.parse_args()
    

    if not args.is_fused_data:
        if args.path_to_config_files is None:
            generate_data_txt(data_dir=args.path_to_data, 
                        txt_name=args.txt_name,
                        num_frames=2,  
                        step=1)
        else:
            generate_data_motion_txt(data_dir=args.path_to_data, 
                        config_dir = args.path_to_config_files,
                        txt_name=args.txt_name, #'train_e2v.txt' #'train_w_events_single.txt', 
                        )
    
    else:
         generate_data_flow_txt(data_dir=args.path_to_data, 
                    txt_name=args.txt_name, #'train_e2v.txt' #'train_w_events_single.txt', 
                    is_flow=args.is_flow)
                   
        
