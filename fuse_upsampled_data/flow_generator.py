
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data 
import cv2
from .event_utils import *
from upsampling.utils.flow_net import FlowNet
from utils.video_reader import VideoRenderer


class UpsampFlowFixNGenerator(nn.Module):
    '''
        Generate flow using FlowNet between frames at fixed number of events
        The format of path_to_data_txt: 
        seq_id num_events timestamp_0(in seconds) timestamp_1 path_to_frame_0 path_to_frame_1 path_to_events between frame_0 and frame_1

    '''
    def __init__(self, path_to_data_txt, path_to_read_data, cfgs, device):
        self.txt_file = path_to_data_txt
        self.path_to_read_data = path_to_read_data
        self.height, self.width = cfgs.image_dim
        self.limit_num_events = cfgs.num_events
        self.len_sequence = cfgs.len_sequence
        self.max_n_per_seq = cfgs.max_n_per_seq
        self.start_id = cfgs.start_id 
        self.end_id = cfgs.end_id 
        
        self.flow_net = FlowNet(cfgs.image_dim, device=device) 
        
        self.video_cnt = []
        self.event_paths = []
        self.image_paths = []
        self.next_image_paths = []
        self.num_events_list = []
        self.timestamps_list = []

        with open(self.txt_file,'rb') as f:
            for line in f:
                str_list = line.strip().split()
                self.video_cnt.append(int(str_list[0])) #video_cnt
                self.num_events_list.append(int(str_list[1]))
                self.timestamps_list.append([float(str_list[2]), float(str_list[3])])
                self.image_paths.append(str(str_list[4], encoding = "utf-8")) #cur_img_path1
                self.next_image_paths.append(str(str_list[5], encoding = "utf-8")) #cur_next_img_path
                self.event_paths.append(str(str_list[6], encoding = "utf-8"))
        f.close()
        
        self.end_id = self.end_id if self.end_id>0 else self.video_cnt[-1]
        
        if self.len_sequence > 0:
            self.split_sequences()
        else:
            self.split_sequences_keep_org_seq()
        
    
    def __len__(self):
        return len(self.sequence_line_id)   
    
    def split_sequences(self):
        '''if len_sequence > 0, split sequences for each reconstruction based on len_sequence, usually for training'''
        prev_video_id = -1
        sum_num_events = 0
        self.sequence_line_id = []
        line_id_per_reconstruction = []
        line_id_per_sequence = []
        frame_cnt, single_frame_cnt = 0, 0
        same_seq_cnt = 0
        
        self.start_output_seq_id = 0
        self.end_output_seq_id = 0
        for line_id, video_id in enumerate(self.video_cnt):
            if video_id != prev_video_id:
                if len(line_id_per_sequence)>= self.len_sequence: #>=5
                    same_seq_cnt += 1
                    if line_id_per_reconstruction:
                        line_id_per_sequence.append(line_id_per_reconstruction)
                    if self.max_n_per_seq <= 0 or same_seq_cnt < self.max_n_per_seq:
                        self.sequence_line_id.append(line_id_per_sequence)
                        if video_id < self.start_id:
                            self.start_output_seq_id += 1
                        if video_id <= self.end_id:
                            self.end_output_seq_id += 1
                line_id_per_sequence = []
                line_id_per_reconstruction = []
                prev_video_id = video_id
                sum_num_events = 0
                single_frame_cnt = 0
                frame_cnt = 0
                same_seq_cnt = 0
                
            cur_num_event = self.num_events_list[line_id]
            sum_num_events += cur_num_event
            line_id_per_reconstruction.append(line_id)
            single_frame_cnt += 1
            if sum_num_events >= self.limit_num_events or (single_frame_cnt==1 and sum_num_events > 0.8*self.limit_num_events):
                line_id_per_sequence.append(line_id_per_reconstruction)
                frame_cnt += 1
                sum_num_events = 0
                single_frame_cnt = 0
                line_id_per_reconstruction = []
                
            if frame_cnt >= self.len_sequence:
                if self.max_n_per_seq <= 0 or same_seq_cnt < self.max_n_per_seq:
                    self.sequence_line_id.append(line_id_per_sequence)
                    if video_id < self.start_id:
                        self.start_output_seq_id += 1
                    if video_id <= self.end_id:
                        self.end_output_seq_id += 1
                line_id_per_sequence = []
                line_id_per_reconstruction = []
                frame_cnt = 0
                sum_num_events = 0
                same_seq_cnt += 1
        
        print('Number of sequence', len(self.sequence_line_id))
        print('Start / end id of output sequence', self.start_output_seq_id, self.end_output_seq_id)

    def split_sequences_keep_org_seq(self):
        '''if len_sequence <= 0, keep the length of original sequence, usually for testing data'''
        prev_video_id = -1
        sum_num_events = 0
        self.start_output_seq_id = self.start_id
        self.end_output_seq_id = self.end_id
        self.sequence_line_id = []
        line_id_per_reconstruction = []
        line_id_per_sequence = []
        frame_cnt, single_frame_cnt = 0, 0
        for line_id, video_id in enumerate(self.video_cnt):
            if video_id != prev_video_id and video_id !=0:
                if line_id_per_reconstruction:
                    line_id_per_sequence.append(line_id_per_reconstruction)
                self.sequence_line_id.append(line_id_per_sequence)
                line_id_per_sequence = []
                line_id_per_reconstruction = []
                prev_video_id = video_id
                sum_num_events = 0
                single_frame_cnt = 0
                frame_cnt = 0
                
            cur_num_event = self.num_events_list[line_id]
            sum_num_events += cur_num_event
            line_id_per_reconstruction.append(line_id)
            single_frame_cnt += 1
            if sum_num_events >= self.limit_num_events or (single_frame_cnt==1 and sum_num_events > 0.8*self.limit_num_events):
                line_id_per_sequence.append(line_id_per_reconstruction)
                frame_cnt += 1
                sum_num_events = 0
                single_frame_cnt = 0
                line_id_per_reconstruction = []

        if line_id_per_sequence:
            self.sequence_line_id.append(line_id_per_sequence)
        
        print('Number of sequence', len(self.sequence_line_id))
        print('Start / end id of output sequence', self.start_output_seq_id, self.end_output_seq_id)

    
    def warp_events(self, events, F_1_0, is_f10=True):
        # input events: [N,4]: t, x, y, p
        # ----------- warp events directly using generated flow map---------------#
        warped_x, warped_y = warp_events_flow_uv_torch(torch.from_numpy(events).float(), torch.from_numpy(F_1_0).float(), img_size=[self.height,self.width], \
            t0=None, is_f10=is_f10)
        warped_events = np.stack([events[:,0], warped_x, warped_y, events[:,-1]], 1)
        mask = events_bounds_mask(warped_x, warped_y, 0, self.width, 0, self.height)
        warped_events = warped_events[mask.astype(bool)]
        return warped_events
   

    def __getitem__(self, index):
        line_id_per_sequence = self.sequence_line_id[index]
        
        seq_events = []
        seq_frames = [cv2.imread(os.path.join(self.path_to_read_data, self.image_paths[line_id_per_sequence[0][0]]), cv2.IMREAD_GRAYSCALE)]
        seq_timestamps = [self.timestamps_list[line_id_per_sequence[0][0]][0]]
        seq_flows = []
        
        for i, line_id_per_reconstruction in enumerate(line_id_per_sequence):
            event_window = np.empty((0,4),dtype=np.float32)
            for line_id in line_id_per_reconstruction:
                event_path = os.path.join(self.path_to_read_data, self.event_paths[line_id])
                cur_event_window = np.load(event_path, allow_pickle=True) #["arr_0"]
                cur_event_window = np.stack((cur_event_window["t"], cur_event_window["x"], cur_event_window["y"],cur_event_window["p"]), axis=1)
                event_window = np.concatenate((event_window, cur_event_window), 0)
            seq_events.append(event_window)
            seq_frames.append(cv2.imread(os.path.join(self.path_to_read_data, self.next_image_paths[line_id_per_reconstruction[-1]]), cv2.IMREAD_GRAYSCALE)) #########
            seq_timestamps.append(self.timestamps_list[line_id_per_reconstruction[-1]][-1])
            
            # generate optical flow
            F_0_1, F_1_0 = self.flow_net.generate_flow([seq_frames[i], seq_frames[i+1]])
            seq_flows.append(np.concatenate([F_0_1, F_1_0], 0))
            
            # warped_events_10 = self.warp_events(event_window, F_1_0, is_f10=True)
            

        return seq_events, seq_frames, seq_timestamps, seq_flows


class GTFlowFixNGenerator(UpsampFlowFixNGenerator):
    '''
        Generate GT flow using motion configs between frames at fixed number of events
        Using VideoRenderer.generate_gt_flow(t0, t1)
        
        The format of path_to_data_txt: 
        seq_id num_events timestamp_0(in seconds) timestamp_1 path_to_frame_0 path_to_frame_1 path_to_events between frame_0 and frame_1

    '''
    
    def __init__(self, path_to_data_txt, path_to_read_data, cfgs, device, replace_path_pair=None):
        super(GTFlowFixNGenerator, self).__init__(path_to_data_txt, path_to_read_data, cfgs, device)

        self.video_renderer = VideoRenderer(image_dim=cfgs.image_dim, is_save_frames=False, replace_path_pair=replace_path_pair)

        self.config_paths = []

        with open(self.txt_file,'rb') as f:
            for line in f:
                str_list = line.strip().split()
                self.config_paths.append(str(str_list[7], encoding = "utf-8"))
        f.close()
        
        
    def __len__(self):
        return len(self.sequence_line_id)   
    

    def __getitem__(self, index):
        if index < self.start_output_seq_id or index > self.end_output_seq_id:
            return 0
        line_id_per_sequence = self.sequence_line_id[index]
        
        self.video_renderer.initialize(self.config_paths[line_id_per_sequence[0][0]], -1, frame_rate=10) # arbitray value
        
        t0 = self.timestamps_list[line_id_per_sequence[0][0]][0]
        seq_events = []
        seq_frames = [cv2.imread(os.path.join(self.path_to_read_data, self.image_paths[line_id_per_sequence[0][0]]), cv2.IMREAD_GRAYSCALE)]
        seq_timestamps = [t0]
        seq_flows = []
        for i, line_id_per_reconstruction in enumerate(line_id_per_sequence):
            event_window = np.empty((0,4),dtype=np.float32)
            for line_id in line_id_per_reconstruction:
                event_path = os.path.join(self.path_to_read_data, self.event_paths[line_id])
                cur_event_window = np.load(event_path, allow_pickle=True) #["arr_0"]
                cur_event_window = np.stack((cur_event_window["t"], cur_event_window["x"], cur_event_window["y"],cur_event_window["p"]), axis=1)
                event_window = np.concatenate((event_window, cur_event_window), 0)
            
            seq_events.append(event_window)
            seq_frames.append(cv2.imread(os.path.join(self.path_to_read_data, self.next_image_paths[line_id_per_reconstruction[-1]]), cv2.IMREAD_GRAYSCALE)) #########
            t1 = self.timestamps_list[line_id_per_reconstruction[-1]][-1]
            seq_timestamps.append(t1)
            
            # Generate optical flow
            F_0_1, F_1_0 = self.video_renderer.generate_gt_flow(t0, t1)
            seq_flows.append(np.concatenate([F_0_1, F_1_0]))
            t0 = t1
            
            # warped_events_10 = self.warp_events(event_window, F_1_0, is_f10=True)
            # seq_warped_events.append(warped_events_10)
            # warped_events_01 = self.warp_events(event_window, F_0_1, is_f10=False)


        return seq_events, seq_frames, seq_timestamps, seq_flows
