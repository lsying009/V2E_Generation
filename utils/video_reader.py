import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys


# from data_io import read_timestamps_file
# from E2V2E.upsampling.utils.upsamp_sequence import Upsampler required for
from .data_io import read_timestamps_file
from upsampling.utils.upsamp_sequence import Upsampler

'''
scene_config.txt:
image_width, image_height, duration(s)
path_to_background_image median_filter_size gaussian_blur_sigma theta0 theta1 x0 x1 y0 y1 sx0 sx1 sy0 sy1
path_to_foreground_image median_filter_size gaussian_blur_sigma theta0 theta1 x0 x1 y0 y1 sx0 sx1 sy0 sy1
path_to_foreground_image median_filter_size gaussian_blur_sigma theta0 theta1 x0 x1 y0 y1 sx0 sx1 sy0 sy1
...
'''


def replace_data_path(org_paths, old_root_path, new_root_path):
    if isinstance(org_paths, list or tuple):
        replaced_paths = []
        for path in org_paths:
            if old_root_path in path:
                replaced_paths.append(path.replace(old_root_path, new_root_path))
            else:
                replaced_paths.append(path)
        return replaced_paths
    else:
        return org_paths.replace(old_root_path, new_root_path)

def read_motion_config(path_to_config_file, replace_path_pair=None):
    images = []
    theta0_list = []
    theta1_list = [] #vec_angulars
    x0_list, x1_list, y0_list, y1_list = [],[],[],[] # motions
    sx0_list, sx1_list, sy0_list, sy1_list = [],[],[],[] # scales
    with open(path_to_config_file, "r") as f:
        image_width, image_height, duration = f.readline().strip().split()
        image_width, image_height, duration = int(image_width), int(image_height), float(duration)
        for image_line in f:
            path_to_image, \
            median_filter_size, gaussian_blur_sigma, \
            theta0, theta1, \
            x0, x1, y0, y1, \
            sx0, sx1, sy0, sy1 = image_line.strip().split()

            if replace_path_pair is not None:
                path_to_image = replace_data_path(path_to_image, replace_path_pair[0], replace_path_pair[1])
            
            image = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_GRAYSCALE)
            if image.ndim==3 and image.shape[-1]==3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            elif image.ndim==2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
                
            image = cv2.medianBlur(image, int(float(median_filter_size)))
            image = cv2.GaussianBlur(image, [3,3], int(float(gaussian_blur_sigma)))
            images.append(image)
            # median_filter_size_list.append(int(float(median_filter_size)))
            # gaussian_blur_sigma_list.append(float(gaussian_blur_sigma))
            theta0_list.append(float(theta0)), theta1_list.append(float(theta1))
            x0_list.append(float(x0)), x1_list.append(float(x1)), y0_list.append(float(y0)), y1_list.append(float(y1))
            sx0_list.append(float(sx0)), sx1_list.append(float(sx1)), sy0_list.append(float(sy0)), sy1_list.append(float(sy1))

    f.close()
    return images, image_width, image_height, duration, \
         theta0_list, theta1_list, \
            x0_list, x1_list, y0_list, y1_list, sx0_list, sx1_list, sy0_list, sy1_list    



class MotionParameters:
    def __init__(self, texture_image_size, dst_image_size, tmax,
                    theta0_deg, theta1_deg,
                    x0,  x1,
                    y0,  y1,
                    sx0,  sx1,
                    sy0,  sy1):
        self.tmax = tmax
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.theta0 = theta0_deg * np.pi / 180.
        self.theta1 = theta1_deg * np.pi / 180.
        self.sx0 = sx0
        self.sx1 = sx1
        self.sy0 = sy0
        self.sy1 = sy1
        
        # texture_image_size [width, height]
        self.K0 = np.array([[texture_image_size[0], 0, 0.5 * texture_image_size[0]],
                    [0, texture_image_size[1], 0.5 * texture_image_size[1]],
                    [0, 0, 1]])
        self.K1 = np.array([[dst_image_size[0], 0, 0.5 * dst_image_size[0]],
                    [0, dst_image_size[1], 0.5 * dst_image_size[1]],
                    [0, 0, 1]])
        
        
    def getAffineTransformation(self, t):
        dtheta = self.theta1 - self.theta0
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        dsx = self.sx1 - self.sx0
        dsy = self.sy1 - self.sy0

        # computation of parameter(t)
        theta = self.theta0 + t/self.tmax * dtheta
        x = self.x0 + t/self.tmax * dx
        y = self.y0 + t/self.tmax * dy
        sx = self.sx0 + t/self.tmax * dsx
        sy = self.sy0 + t/self.tmax * dsy
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
    
        A = np.array([ [sx * ctheta, -sy * stheta, x],
              [sx * stheta, sy * ctheta,  y],
              [0,           0,            1]])
                    
        return A


    def getAffineTransformationWithJacobian(self, t):
        dtheta = self.theta1 - self.theta0
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        dsx = self.sx1 - self.sx0
        dsy = self.sy1 - self.sy0

        # computation of parameter(t)
        theta = self.theta0 + t/self.tmax * dtheta
        x = self.x0 + t/self.tmax * dx
        y = self.y0 + t/self.tmax * dy
        sx = self.sx0 + t/self.tmax * dsx
        sy = self.sy0 + t/self.tmax * dsy
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
    
        A = np.array([ [sx * ctheta, -sy * stheta, x],
              [sx * stheta, sy * ctheta,  y],
              [0,           0,            1]])

        # computation of dparameter_dt(t)
        dtheta_dt = 1./self.tmax * dtheta
        dx_dt = 1./self.tmax * dx
        dy_dt = 1./self.tmax * dy
        dsx_dt = 1./self.tmax * dsx
        dsy_dt = 1./self.tmax * dsy

        dAdt = np.array([[dsx_dt * ctheta - dtheta_dt * stheta * sx, -dsy_dt * stheta - dtheta_dt * ctheta * sy, dx_dt],
                [dsx_dt * stheta + dtheta_dt * ctheta * sx, dsy_dt * ctheta - dtheta_dt * stheta * sy, dy_dt],
                [0.0,                                       0.0,                                       0.0]])

        return A, dAdt


## parent class for video reader
class VR:
    def __init__(self, image_dim, padding_size=[4,4]):
        self.padding_size = padding_size # padding size before and after sequence
        self.height, self.width = image_dim
        
        self.prev_frame_cache = np.zeros((self.padding_size[0], self.height, self.width), dtype=np.uint8)
        self.after_frame_cache = np.zeros((self.padding_size[1], self.height, self.width), dtype=np.uint8)
        # ts_cache_size = self.padding_size[1] if self.padding_size[1]==0 else self.padding_size[1]+1
        self.prev_ts_cache = np.zeros(self.padding_size[1]+1, dtype=np.float64)
        self.frame_id = 0
        self.num_frames = -1
        self.timestamps = []
    
    def get_frame_rate(self):
        return self.num_frames / self.timestamps[-1]-self.timestamps[0]
    
    def update_frame(self):
        ## Required redefine
        return np.zeros((self.height, self.width),dtype=np.uint8), 0
        
        
    def update_frame_pack(self, num_pack_frames, padding_mode='edge'): # 'edge or reflect'
        # num_update_frames = num_pack_frames + self.padding_size[1] #
        start_frame_id = self.frame_id
        if start_frame_id != 0: #and self.padding_size[1]>0
            num_pack_frames -= (self.padding_size[1]+1)

        num_pack_frames = min(num_pack_frames, self.num_frames-self.frame_id)
        
        frame_pack, timestamps = [], []
        
        # print(num_pack_frames, start_frame_id, self.num_frames)
        for _ in range(num_pack_frames):
            frame, t = self.update_frame()
            frame_pack.append(frame)
            timestamps.append(t)
        gt_frame = frame_pack[-1]
        
        if start_frame_id != 0:
            frame_pack = np.concatenate((self.after_frame_cache, np.stack(frame_pack,0)),0)
            timestamps = np.concatenate((self.prev_ts_cache, np.stack(timestamps,0)),0)
        else:
            frame_pack = np.stack(frame_pack, 0)
            timestamps = np.stack(timestamps, 0)
            self.prev_frame_cache = np.pad(frame_pack, pad_width=((self.padding_size[0],0),(0,0),(0,0)), mode=padding_mode)[:self.padding_size[0],:,:]
        
        if self.num_frames-self.frame_id >= self.padding_size[1]:
            for i in range(self.padding_size[1]):
                frame, t = self.update_frame()
                self.after_frame_cache[i] = frame
                self.prev_ts_cache[i+1] = t
        else:
            self.after_frame_cache = np.pad(frame_pack, pad_width=((0, self.padding_size[1]),(0,0),(0,0)), mode=padding_mode)[-self.padding_size[1]:,:,:]

        self.prev_ts_cache[0] = timestamps[-1]
        
        frame_pack = np.concatenate((self.prev_frame_cache, frame_pack, self.after_frame_cache), 0)
        end = len(frame_pack)-self.padding_size[1]
        self.prev_frame_cache = frame_pack[-(self.padding_size[0]+self.padding_size[1]+1):end]
        # print(timestamps)
        # diff_frame_pack = frame_pack[1:]-frame_pack[:-1]
        # for i, frame in enumerate(diff_frame_pack):
        #     plt.subplot(3,6,i+1)
        #     plt.imshow(diff_frame_pack[i], cmap='gray')
        # plt.show()
        return frame_pack, gt_frame, timestamps
    
    

class VideoRenderer(VR):
    def __init__(self, image_dim, padding_size=[4,4], is_save_frames=False, replace_path_pair=None):
        super(VideoRenderer, self).__init__(image_dim, padding_size)
        self.is_save_frames = is_save_frames
        self.replace_path_pair=replace_path_pair


    def initialize(self, path_to_config_file, num_load_frames=-1, frame_rate=0, save_path=None):
        self.textures, self.width, self.height, self.duration, \
        self.theta0, self.theta1, \
        self.x0,  self.x1, \
        self.y0,  self.y1, \
        self.sx0,  self.sx1, \
        self.sy0,  self.sy1 \
            = read_motion_config(path_to_config_file, self.replace_path_pair)
        
        self.texture_motion_params = []
        
        for i in range(len(self.textures)):
            self.texture_motion_params.append(MotionParameters([self.textures[i].shape[1], self.textures[i].shape[0]],  # W, H
                                                 [self.width, self.height], #self.height, self.height
                                                 self.duration, 
                                                 self.theta0[i], self.theta1[i],
                                                 self.x0[i],  self.x1[i],
                                                 self.y0[i],  self.y1[i],
                                                 self.sx0[i],  self.sx1[i],
                                                 self.sy0[i],  self.sy1[i]))
        
        self.frame_id = 0
        if frame_rate > 0:
            self.num_frames = int(self.duration * frame_rate)
            self.timestamps = np.linspace(0, self.duration, self.num_frames)
        else:
            discrete_num_frames = self.calc_frame_rate()
            self.num_frames = sum(discrete_num_frames)-len(discrete_num_frames)+1
            print('Number of frames: {:d}, Adaptive frame rate: {:.2f}'.format(self.num_frames, self.num_frames / self.duration))
            self.timestamps = []
            for i, N in enumerate(discrete_num_frames):
                if i < len(discrete_num_frames)-1:
                    self.timestamps.extend(np.linspace(i*self.dt, (i+1)*self.dt, N)[:-1])
                else:
                    self.timestamps.extend(np.linspace(i*self.dt, (i+1)*self.dt, N))
        if num_load_frames > 0:
            self.num_frames = min(self.num_frames, num_load_frames)
            self.timestamps = self.timestamps[:num_load_frames]

        self.prev_frame_cache.fill(0)
        self.after_frame_cache.fill(0)
        self.prev_ts_cache.fill(0)
        
        if self.is_save_frames:
            seq_name = path_to_config_file.split('/')[-1].split('.')[0] #.split('_')[2]
            self.path_to_save_frames = os.path.join(save_path, seq_name)
            print('Will save frames in {}'.format(self.path_to_save_frames))
            if not os.path.exists(self.path_to_save_frames):
                os.makedirs(self.path_to_save_frames)
            self.path_to_save_timestamps = os.path.join(self.path_to_save_frames, 'timestamps.txt')
            self.f_timestamps = open(self.path_to_save_timestamps, 'w+')

        
    def __del__(self):
        if self.is_save_frames:
            self.f_timestamps.close()

            
    def calc_frame_rate(self):
        max_layer_displacements = []
        self.discrete = 10
        self.dt = self.duration / self.discrete
        dt_displacement = []
        for k in range(self.discrete):
            max_layer_displacements = []

            for i in range(len(self.textures)):
                A_t0 = self.texture_motion_params[i].getAffineTransformation(k*self.dt)
                A_t1 = self.texture_motion_params[i].getAffineTransformation((k+1)*self.dt)

                A_t1_t0 = np.dot(A_t1, np.linalg.inv(A_t0))
                # M_t1_t0 maps any point on the first image to its position in the last image
                # M_t1_t0 = np.dot(self.texture_motion_params[i].K1, np.dot(A_t1_t0, np.linalg.inv(self.texture_motion_params[i].K1)))

                M_t0 = np.dot(self.texture_motion_params[i].K1, np.dot(A_t0, np.linalg.inv(self.texture_motion_params[i].K0)))
                M_t1 = np.dot(self.texture_motion_params[i].K1, np.dot(A_t1, np.linalg.inv(self.texture_motion_params[i].K0)))
                
                xs = np.arange(0,self.textures[i].shape[1],1)
                ys = np.arange(0,self.textures[i].shape[0],1)
                xx, yy = np.meshgrid(xs,ys,sparse=False)
                xx = np.reshape(xx,(1,-1))
                yy = np.reshape(yy,(1,-1))
                ones = np.ones(xx.shape)
                pixels = np.vstack((xx, yy, ones))
                positions_t0 = np.dot(M_t0, pixels)
                positions_t1 =  np.dot(M_t1, pixels)
                
                mask = ((positions_t1[0] < 0) | (positions_t1[0] >= self.width)) if i!=0 else np.zeros_like(positions_t1[0]).astype(np.bool_)
                displacement_x = (positions_t1[0] -  positions_t0[0])
                displacement_x[mask] = 0
                mask = ((positions_t1[1] < 0) | (positions_t1[1] >= self.height)) if i!=0 else np.zeros_like(positions_t1[1]).astype(np.bool_)
                displacement_y = (positions_t1[1] -  positions_t0[1]) 
                displacement_y[mask] = 0
                
                displacement = max(np.sqrt(displacement_x*displacement_x+displacement_y*displacement_y))
                # displacement = max(max(abs(displacement_x)), max(abs(displacement_y)))
                max_layer_displacements.append(displacement)
            dt_displacement.append(max_layer_displacements)

        max_dt_displacement = np.max(np.array(dt_displacement),1)
        num_frames = np.floor(max_dt_displacement).astype(np.int16)

        return num_frames

    def generate_gt_flow(self, t0, t1, is_back=False):

        F_1_0 = np.zeros((self.height, self.width, 2), dtype=np.float32)
        F_0_1 = np.zeros((self.height, self.width, 2), dtype=np.float32)
        for i in range(len(self.textures)):
            A_t0 = self.texture_motion_params[i].getAffineTransformation(t0)
            A_t1 = self.texture_motion_params[i].getAffineTransformation(t1)

            M_t0 = np.dot(self.texture_motion_params[i].K1, np.dot(A_t0, np.linalg.inv(self.texture_motion_params[i].K0)))
            M_t1 = np.dot(self.texture_motion_params[i].K1, np.dot(A_t1, np.linalg.inv(self.texture_motion_params[i].K0)))
 
            xs = np.arange(0,self.textures[i].shape[1],1)
            ys = np.arange(0,self.textures[i].shape[0],1)
            xx, yy = np.meshgrid(xs,ys,sparse=False)
            xx = np.reshape(xx,(1,-1))
            yy = np.reshape(yy,(1,-1))
            ones = np.ones(xx.shape)
            pixels = np.vstack((xx, yy, ones))
            positions_t0 = np.dot(M_t0, pixels)
            positions_t1 =  np.dot(M_t1, pixels)
            
                
            mask_x_t1= ((positions_t1[0] < 0) | (positions_t1[0] >= self.width)) if i!=0 else np.zeros_like(positions_t1[0]).astype(np.bool_)
            mask_y_t1 = ((positions_t1[1] < 0) | (positions_t1[1] >= self.height)) if i!=0 else np.zeros_like(positions_t1[1]).astype(np.bool_)
            mask_x_t0= ((positions_t0[0] < 0) | (positions_t0[0] >= self.width)) if i!=0 else np.zeros_like(positions_t0[0]).astype(np.bool_)
            mask_y_t0 = ((positions_t0[1] < 0) | (positions_t0[1] >= self.height)) if i!=0 else np.zeros_like(positions_t0[1]).astype(np.bool_)
            displacement_x = (positions_t1[0] -  positions_t0[0])
            displacement_y = (positions_t1[1] -  positions_t0[1]) 
            
            # displacement_x[mask] = 0
            # displacement_y[mask] = 0

            # H x W x 2
            mask_transparent = (self.textures[i][...,-1] == 0)
            flow_0_1 = np.stack((displacement_x, displacement_y), axis=0)
            flow_1_0 = -flow_0_1
            
            flow_0_1[0][mask_x_t0] = 0
            flow_0_1[1][mask_y_t0] = 0
            flow_1_0[0][mask_x_t1] = 0
            flow_1_0[1][mask_y_t1] = 0
            
            flow_0_1 = np.reshape(flow_0_1, (2, self.textures[i].shape[0],self.textures[i].shape[1]))
            flow_0_1[0][mask_transparent] = 0.
            flow_0_1[1][mask_transparent] = 0.
            flow_0_1 = flow_0_1.transpose((1,2,0))
            
            flow_1_0 = np.reshape(flow_1_0, (2, self.textures[i].shape[0],self.textures[i].shape[1]))
            flow_1_0[0][mask_transparent] = 0.
            flow_1_0[1][mask_transparent] = 0.
            flow_1_0 = flow_1_0.transpose((1,2,0))
            
            mask_transparent = self.textures[i][...,-1] 

            
            border_mode = cv2.BORDER_REFLECT101 if i==0 else cv2.BORDER_TRANSPARENT
            if i == 0:
                
                F_0_1 = cv2.warpPerspective(flow_0_1,
                                M_t0,
                                (self.width, self.height),
                                dst=F_0_1,
                                flags=cv2.INTER_LINEAR, #INTER_NEAREST
                                borderMode=border_mode)

                F_1_0 = cv2.warpPerspective(flow_1_0,
                                M_t1,
                                (self.width, self.height),
                                dst=F_1_0,
                                flags=cv2.INTER_LINEAR, #INTER_NEAREST
                                borderMode=border_mode)
                F_0_1[np.isnan(F_0_1)] = 0.
                F_1_0[np.isnan(F_1_0)] = 0.
      
            else:
                fg_canvas_t0 = np.zeros((self.height, self.width,2), dtype=np.float32)
                fg_canvas_t1 = np.zeros((self.height, self.width,2), dtype=np.float32)
                warped_mask_t0 = np.zeros((self.height, self.width), dtype=np.uint8)
                warped_mask_t1 = np.zeros((self.height, self.width), dtype=np.uint8)
                fg_canvas_t0 = cv2.warpPerspective(flow_0_1,
                                    M_t0,
                                    (self.width, self.height),
                                    dst=fg_canvas_t0,
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=border_mode)
                fg_canvas_t1 = cv2.warpPerspective(flow_1_0,
                                    M_t1,
                                    (self.width, self.height),
                                    dst=fg_canvas_t1,
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=border_mode)
   
                fg_canvas_t0[np.isnan(fg_canvas_t0)] = 0.
                fg_canvas_t1[np.isnan(fg_canvas_t1)] = 0.
                fg_canvas_t0[np.isinf(fg_canvas_t0)] = 0.
                fg_canvas_t1[np.isinf(fg_canvas_t1)] = 0.

                warped_mask_t0 = cv2.warpPerspective(mask_transparent,
                                    M_t0,
                                    (self.width, self.height),
                                    dst=warped_mask_t0,
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=border_mode)
                warped_mask_t1 = cv2.warpPerspective(mask_transparent,
                                    M_t1,
                                    (self.width, self.height),
                                    dst=warped_mask_t1,
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=border_mode)

                _weight = warped_mask_t0/255.
                weight = cv2.merge((_weight,_weight))
                F_0_1 = weight*fg_canvas_t0+ (1-weight)*F_0_1
                
                _weight = warped_mask_t1/255.
                weight = cv2.merge((_weight,_weight))
                F_1_0 = weight*fg_canvas_t1+ (1-weight)*F_1_0
            # plt.imshow(F_0_1[:,:,0])
            # plt.show()

        F_0_1 = F_0_1.transpose((2,0,1)).astype(np.float32)
        F_1_0 = F_1_0.transpose((2,0,1)).astype(np.float32)
        
        return F_0_1, F_1_0
        

    def update_frame(self, frame_id=None): 
        canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        
        if frame_id is not None:
            self.frame_id = frame_id
        t = self.timestamps[self.frame_id]
        t = min(t, self.duration)
        for i in range(len(self.texture_motion_params)): 
            A10 = self.texture_motion_params[i].getAffineTransformation(t)
            M_10 = np.dot(self.texture_motion_params[i].K1, np.dot(A10, np.linalg.inv(self.texture_motion_params[i].K0)))

            border_mode = cv2.BORDER_REFLECT101 if i==0 else cv2.BORDER_TRANSPARENT
            if i == 0:
                canvas = cv2.warpPerspective(self.textures[i],
                                    M_10,
                                    (self.width, self.height),
                                    dst=canvas,
                                    flags=cv2.INTER_LINEAR, #INTER_NEAREST
                                    borderMode=border_mode)
            else:
                fg_canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
                fg_canvas = cv2.warpPerspective(self.textures[i],
                                    M_10,
                                    (self.width, self.height),
                                    dst=fg_canvas,
                                    flags=cv2.INTER_LINEAR, # INTER_LINEAR
                                    borderMode=border_mode)
                _weight = fg_canvas[:,:,-1]/255
                weight = cv2.merge((_weight, _weight, _weight, _weight))
                canvas = weight*fg_canvas+ (1-weight)*canvas
                canvas = canvas.astype(np.uint8)
            ######### alpha并没有融合而是覆盖
            # _, mask = cv2.threshold(object, 0,255,cv2.THRESH_BINARY)
            # object = cv2.bitwise_and(object, object, mask=mask)
            # canvas = cv2.bitwise_and(canvas, canvas, mask=~mask)
            # canvas = cv2.add(canvas, object)

        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2GRAY)
        
        
        if self.is_save_frames:
            cv2.imwrite(os.path.join(self.path_to_save_frames, 'frame_{:010d}.png'.format(self.frame_id)), canvas)
            self.f_timestamps.write(str(self.frame_id)+' '+str(t)+'\n')

        self.frame_id += 1
               
        return canvas, t 


class VideoInterpolator(VR):
    def __init__(self, image_dim, padding_size=[4,4], device='cuda:0', is_save_frames=False, is_with_flow=False):
        super(VideoInterpolator, self).__init__(image_dim, padding_size)
        self.upsampler = Upsampler(device=device, image_dim=image_dim, is_train=False, is_with_flow=is_with_flow)
        self.is_save_frames = is_save_frames
        # self.is_with_flow = is_with_flow
        
    def initialize(self, path_to_sequence, num_load_frames=-1, save_path=None):
        path_to_frames = []
        for root, dirs, files in os.walk(path_to_sequence):
            for file_name in files:
                if file_name.split('.')[-1] in ['jpg','png']:
                    path_to_frames.append(os.path.join(root, file_name))
                elif file_name in ['timestamps.txt', 'images.txt']:#.split('.')[-1] in ['txt']:
                    path_to_timestamps = os.path.join(root, file_name)
          
        path_to_frames.sort()
        if num_load_frames > 0:
            path_to_frames = path_to_frames[:num_load_frames]
        
        self.frame_id = 0
        
        timestamps = read_timestamps_file(path_to_timestamps)
        if num_load_frames > 0:
            timestamps = timestamps[:num_load_frames]

        self.prev_frame_cache.fill(0) # = np.zeros((self.padding_size[0], self.height, self.width), dtype=np.uint8)
        self.after_frame_cache.fill(0) #= np.zeros((self.padding_size[1], self.height, self.width), dtype=np.uint8)
        self.prev_ts_cache.fill(0) # = np.zeros((self.padding_size[1]+1), dtype=np.float64)
        
        frames = []
        for path_to_frame in path_to_frames:
            frames.append(cv2.imread(path_to_frame, cv2.IMREAD_GRAYSCALE))

        # if not self.is_with_flow:
        self.frames, self.timestamps = self.upsampler.upsampling(frames, timestamps)
        # else:
        #     self.frames, self.timestamps, self.optical_flows = self.upsampler.upsampling(frames, timestamps)
        self.num_frames = len(self.timestamps)

        if self.is_save_frames:
            # seq_name = path_to_sequence.split('/')[-1]
            for name in path_to_sequence.split('/'):
                if "sequence" in name:
                    seq_name = name
            self.path_to_save_frames = os.path.join(save_path, seq_name, 'frames')
            print('Will save frames in {}'.format(self.path_to_save_frames))
            if not os.path.exists(self.path_to_save_frames):
                os.makedirs(self.path_to_save_frames)
            self.path_to_save_timestamps = os.path.join(self.path_to_save_frames, 'timestamps.txt')
            self.f_timestamps = open(self.path_to_save_timestamps, 'w+')
            
            # if self.is_with_flow:
            #     self.path_to_save_flow = os.path.join(save_path, seq_name, 'flow')
            #     if not os.path.exists(self.path_to_save_flow):
            #         os.makedirs(self.path_to_save_flow)
    
    def __del__(self):
        if self.is_save_frames:
            self.f_timestamps.close() 
        
    def update_frame(self, frame_id=None):
        if frame_id is not None:
            self.frame_id = frame_id
        
        if self.is_save_frames:
            cv2.imwrite(os.path.join(self.path_to_save_frames, 'frame_{:010d}.png'.format(self.frame_id)), self.frames[self.frame_id])
            self.f_timestamps.write(str(self.frame_id)+' '+str(self.timestamps[self.frame_id])+'\n')
            # if self.is_with_flow and self.frame_id < len(self.optical_flows):
            #     np.savez_compressed(os.path.join(self.path_to_save_flow, 'flow_{:010d}.npz'.format(self.frame_id)), flow=self.optical_flows[self.frame_id])
            
        self.frame_id += 1
        return self.frames[self.frame_id-1], self.timestamps[self.frame_id-1]
    
   

class VideoReader(VR):
    def __init__(self, image_dim, padding_size=[4,4], ds=[1/4,1/4]):
        super(VideoReader, self).__init__(image_dim, padding_size)
        self.ds = ds
        
    def initialize(self, path_to_video, num_load_frames=-1):
        cap = cv2.VideoCapture(path_to_video)
    
        if (cap.isOpened()== False):
            assert "Error opening video stream or file"
                
        self.frames, self.timestamps = [], []
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_load_frames = frame_number if num_load_frames < 0 else num_load_frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        while(cap.isOpened()):
            frame_exists, frame = cap.read()
            if frame_exists:
                if frame_count > num_load_frames:
                    break
                # timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                self.timestamps.append(float(frame_count)/fps)
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, dsize=(int(gray.shape[1]*self.ds[1]), int(gray.shape[0]*self.ds[0])))
                if frame.shape[0] > frame.shape[1]:
                    gray = gray.T
                self.frames.append(gray)
                
            else:
                break
        
        cap.release()
        self.num_frames = len(self.frames)

        self.prev_frame_cache.fill(0) # = np.zeros((self.padding_size[0], self.height, self.width), dtype=np.uint8)
        self.after_frame_cache.fill(0) #= np.zeros((self.padding_size[1], self.height, self.width), dtype=np.uint8)
        self.prev_ts_cache.fill(0) # = np.zeros((self.padding_size[1]+1), dtype=np.float64)
        self.frame_id = 0
        
    def update_frame(self, frame_id=None): 
        if frame_id is not None:
            self.frame_id = frame_id
        frame = self.frames[self.frame_id]
        timestamp = self.timestamps[self.frame_id]
        self.frame_id += 1
    
        return frame, timestamp
    
             

class ImageReader(VR):
    def __init__(self, image_dim, padding_size=[4,4]):
        super(ImageReader, self).__init__(image_dim, padding_size)
        
        
    def initialize(self, path_to_sequence, num_load_frames=-1):

        self.path_to_frames = []
        for file_name in os.listdir(path_to_sequence):
            if file_name.split('.')[-1] in ['jpg','png']:
                self.path_to_frames.append(os.path.join(path_to_sequence, file_name))
            elif file_name.split('.')[-1] in ['txt']:
                path_to_timestamps = os.path.join(path_to_sequence, file_name)
        self.path_to_frames.sort()
        if num_load_frames > 0:
            self.path_to_frames = self.path_to_frames[:num_load_frames]
        
        self.num_frames = len(self.path_to_frames)
        self.frame_id = 0
        
        self.timestamps = []
        with open(path_to_timestamps, 'r') as f:
            for line in f:
                self.timestamps.append(float(line.strip().split()[-1]))
        f.close()

        self.prev_frame_cache.fill(0) # = np.zeros((self.padding_size[0], self.height, self.width), dtype=np.uint8)
        self.after_frame_cache.fill(0) #= np.zeros((self.padding_size[1], self.height, self.width), dtype=np.uint8)
        self.prev_ts_cache.fill(0) # = np.zeros((self.padding_size[1]+1), dtype=np.float64)

        
    def update_frame(self, frame_id=None): 
        if frame_id is not None:
            self.frame_id = frame_id
        frame = cv2.imread(self.path_to_frames[self.frame_id], cv2.IMREAD_GRAYSCALE)
        timestamp = self.timestamps[self.frame_id]
        self.frame_id += 1
    
        return frame, timestamp    
