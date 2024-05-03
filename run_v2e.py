from random import uniform
from v2ecore.emulator import EventEmulator
import os
import numpy as np
import cv2
import torch
import random
from utils.data_io import read_timestamps_file
import argparse
import shutil


def run_v2e(path_to_intensity_data, path_to_save_data, 
            start_id=0, end_id=-1,
            sigma_thres=0.03, cutoff_hz=-1,
                leak_rate_hz=0.1, shot_noise_rate_hz=1,
                leak_jitter_fraction=0.1,
                noise_rate_cov_decades=0.1,
                refractory_period_s=0.001, device='cuda:0'):
    '''Generate events based on HFR frames using v2e '''

    input_cutoff_hz = cutoff_hz
    path_to_sequences = []
    path_to_save_sequences = []
    end_id = end_id if end_id>0 else len(os.listdir(path_to_intensity_data))
    for seq_name in os.listdir(path_to_intensity_data):
        if 'sequence' in seq_name and int(seq_name.split('_')[-1])>=start_id and int(seq_name.split('_')[-1]) < end_id:
            path_to_sequences.append(os.path.join(path_to_intensity_data, seq_name))
            path_to_save_sequences.append(os.path.join(path_to_save_data, seq_name))
    path_to_sequences.sort()
    path_to_save_sequences.sort()
    
    for seq_id, path_to_sequence in enumerate(path_to_sequences):
        path_to_save_sequence = path_to_save_sequences[seq_id]
        path_to_cur_frames = []
        path_to_timestamps = []
        
        for root, dirs, file_names in os.walk(path_to_sequence):
            for file_name in file_names:
                if file_name.split('.')[-1] in ['png', 'jpg']:
                    path_to_cur_frames.append(os.path.join(root, file_name))
                    
                if file_name.split('.')[-1] in ['txt']:
                    path_to_timestamps = os.path.join(root, file_name)
                    timestamps = read_timestamps_file(path_to_timestamps)
        path_to_cur_frames.sort()
        
        if path_to_sequence != path_to_save_sequence:
            dst = os.path.join(path_to_save_sequence, 'frames')
            if not os.path.exists(dst): 
                os.makedirs(dst)
            path_to_save_frames = []
            for _, src in enumerate(path_to_cur_frames):
                shutil.copy(src, dst)
                path_to_save_frames.append(os.path.join(dst, src.split('/')[-1]))
            shutil.copy(path_to_timestamps, dst)
        else:
            path_to_save_frames = path_to_cur_frames
        
        path_to_save_events= os.path.join(path_to_save_sequence, 'events')
        if not os.path.exists(path_to_save_events):
            os.makedirs(path_to_save_events)
        print('Write down events to {}'.format(path_to_save_events))

        
        ###----------- Customize as you like  -----------###
        c1 = random.choice([0.2,0.4,0.6,0.8,1.0]) #0.8 1.0
        c2 = np.random.normal(1, 0.1)*c1
        
        
        if seq_id%2==0:
            pos_thres = c1
            neg_thres = c2
            cutoff_hz = input_cutoff_hz

        else:
            pos_thres = c2
            neg_thres = c1

            cutoff_hz = np.random.normal(1, 0.2) * 150
            cutoff_hz = min(max(cutoff_hz, 30), 200)

            
        pos_thres = min(max(pos_thres, 0.05), 1.5)
        neg_thres = min(max(neg_thres, 0.05), 1.5)
        ###----------- Customize as you like  -----------###

        
        emulator = EventEmulator(
                pos_thres=pos_thres, neg_thres=neg_thres,
                sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,
                leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz,
                leak_jitter_fraction=leak_jitter_fraction,
                noise_rate_cov_decades=noise_rate_cov_decades,
                refractory_period_s=refractory_period_s,
                dvs_npz=path_to_save_events,
                device=device,
                )
        
        for frame_idx, path_to_frame in enumerate(path_to_save_frames):
            frame = cv2.imread(path_to_frame, cv2.IMREAD_GRAYSCALE)
            fr_time = timestamps[frame_idx]
            newEvents = emulator.generate_events(frame, fr_time)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate events from HFR video sequences')
    parser.add_argument('--path_to_intensity_data', type=str, help='folder to read intensity frames')
    parser.add_argument('--path_to_save_data', type=str, default=None)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=-1)
    parser.add_argument('--sigma_thres', type=float, default=0.03)
    parser.add_argument('--cutoff_hz', type=float, default=200)
    parser.add_argument('--leak_rate_hz', type=float, default=0.1)
    parser.add_argument('--shot_noise_rate_hz', type=float, default=1)
    parser.add_argument('--leak_jitter_fraction', type=float, default=0.1)
    parser.add_argument('--noise_rate_cov_decades', type=float, default=0.1)
    parser.add_argument('--refractory_period_s', type=float, default=0.001)
    
    args = parser.parse_args()
    
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        # memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
        # os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
        # os.system('rm tmp')
        # device = torch.device('cuda:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('device: ', device)
    
    
    if args.path_to_save_data is None:
        args.path_to_save_data = args.path_to_intensity_data
    run_v2e(args.path_to_intensity_data, args.path_to_save_data, 
                start_id=args.start_id, end_id=args.end_id,
                sigma_thres=args.sigma_thres, cutoff_hz=args.cutoff_hz,
                leak_rate_hz=args.leak_rate_hz, shot_noise_rate_hz=args.shot_noise_rate_hz,
                leak_jitter_fraction=args.leak_jitter_fraction,
                noise_rate_cov_decades=args.noise_rate_cov_decades,
                refractory_period_s=args.refractory_period_s)

    
  
