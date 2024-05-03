import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from upsampling.utils.flow_net import FlowNet

def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if len(events) == 0:
        return np.reshape(voxel_grid, (num_bins, height, width))
    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    # events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    xs = events[:, 1].astype(np.uint)
    ys = events[:, 2].astype(np.uint)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.uint)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))


    return voxel_grid

def event_preprocess(event_voxel_grid, mode='std', filter_hot_pixel=False):
# Normalize the event tensor (voxel grid) so that
# the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
    num_bins = event_voxel_grid.shape[0]
    if filter_hot_pixel:
        event_voxel_grid[abs(event_voxel_grid) > 25./num_bins]= 0

    if mode == 'maxmin':
        event_voxel_grid = (event_voxel_grid- event_voxel_grid.min())/(event_voxel_grid.max()- event_voxel_grid.min()+1e-8)
    elif mode == 'std':
        nonzero_ev = (event_voxel_grid != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = event_voxel_grid.sum() / num_nonzeros
            mask = nonzero_ev.astype(np.float32)
            stddev = np.sqrt((event_voxel_grid ** 2).sum() / num_nonzeros - mean ** 2)
            
            event_voxel_grid = mask * (event_voxel_grid - mean) / (stddev + 1e-8)
    else:
        assert mode == 'maxmin' or mode == 'std'
    return event_voxel_grid


def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    """
    Get a mask of the events that are within the given bounds
    """
    mask = np.where(np.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    mask *= np.where(np.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
    return mask

def clip_events_to_bounds(xs, ys, ps, bounds):
    """
    Clip events to the given bounds
    """
    mask = events_bounds_mask(xs, ys, 0, bounds[1], 0, bounds[0])
    return xs*mask, ys*mask, ps*mask


def warp_events_flow_velocity_torch(xt, yt, tt, pt, flow_field, img_size, t0=None):
    """
    Given events and a flow field, warp the events by the flow
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    flow_field : 2D tensor containing the flow at each x,y position
    t0 : the reference time to warp events to. If empty, will use the
        timestamp of the last event
    Returns
    -------
    warped_xt: x coords of warped events
    warped_yt: y coords of warped events
    """
    if len(xt.shape) > 1:
        xt, yt, tt, pt = xt.squeeze(), yt.squeeze(), tt.squeeze(), pt.squeeze()
    if t0 is None:
        t0 = tt[-1]
    while len(flow_field.size()) < 4:
        flow_field = torch.reshape(flow_field, [1, 2, 1, 1]) #flow_field.unsqueeze(0)
    if len(xt.size()) == 1:
        event_indices = torch.transpose(torch.stack((xt, yt), dim=0), 0, 1)
    else:
        event_indices = torch.transpose(torch.cat((xt, yt), dim=1), 0, 1)
    #event_indices.requires_grad_ = False
    event_indices = torch.reshape(event_indices, [1, 1, len(xt), 2])
    
    # Event indices need to be between -1 and 1 for F.gridsample
    event_indices[:,:,:,0] = event_indices[:,:,:,0]/(img_size[0]-1)*2.0-1.0
    event_indices[:,:,:,1] = event_indices[:,:,:,1]/(img_size[1]-1)*2.0-1.0
    
    # flow_field
    # N,C,H,W / N,H2,W2,2 --> N,C,H2,W2
    # 1,2,H,W / 1,1,N,2 --> 1,2,1,N
    flow_at_event = F.grid_sample(flow_field, event_indices, align_corners=True) 

    dt = (tt-t0).squeeze()

    warped_xt = xt+flow_at_event[:,0,:,:].squeeze()*dt
    warped_yt = yt+flow_at_event[:,1,:,:].squeeze()*dt
    

    return warped_xt, warped_yt


def warp_events_flow_uv_torch(events, flow_field, img_size, t0=None, is_f10=True):
    """
    Given events and a flow field, warp the events by the flow
    Parameters
    ----------
    events: t,x,y,p 
    flow_field : 2D tensor containing the flow at each x,y position
    t0 : the reference time to warp events to. If empty, will use the
        timestamp of the last event
    Returns
    -------
    warped_xt: x coords of warped events
    warped_yt: y coords of warped events
    """
    t, x, y, p = events[:,0]-events[0,0], events[:,1], events[:,2], events[:,3]
    if t0 is None:
        t0 = t[-1]
    while len(flow_field.size()) < 4:
        flow_field = flow_field.unsqueeze(0)
    if len(x.size()) == 1:
        event_indices = torch.transpose(torch.stack((x, y), dim=0), 0, 1)
    else:
        event_indices = torch.transpose(torch.cat((x, y), dim=1), 0, 1)
    #event_indices.requires_grad_ = False
    event_indices = torch.reshape(event_indices, [1, 1, len(x), 2])
    
    # Event indices need to be between -1 and 1 for F.gridsample
    event_indices[:,:,:,0] = event_indices[:,:,:,0]/(img_size[0]-1)*2.0-1.0
    event_indices[:,:,:,1] = event_indices[:,:,:,1]/(img_size[1]-1)*2.0-1.0
    
    # flow_field
    # N,C,H,W / N,H2,W2,2 --> N,C,H2,W2
    flow_at_event = F.grid_sample(flow_field, event_indices, align_corners=True) 
    
    dt = (t-t0).squeeze() #/max(abs(t-t0))
    nt = max(dt) - min(dt)
    dt /= nt
    # print(dt.max(), dt.min())
    # ddt = (t-t0).squeeze() /max(abs(t-t0))
    # print(ddt.min(), ddt.max())
    dt = dt if is_f10 else -dt
    warped_xt = x+flow_at_event[:,0,:,:].squeeze()*dt
    warped_yt = y+flow_at_event[:,1,:,:].squeeze()*dt
    

    return warped_xt, warped_yt


def warp_event_list(event_list, flow_list, height, width):
    to_be_warped_events = np.empty((0,4),dtype=np.float32)
    org_timestamps = np.empty((0),dtype=np.float32)

    for events, flow in zip(event_list, flow_list):
        if len(events) == 0:
            continue
        to_be_warped_events = np.concatenate([to_be_warped_events, events], axis=0)
        
        warped_x, warped_y = warp_events_flow_uv_torch(torch.from_numpy(to_be_warped_events).float(), torch.from_numpy(flow).float(), img_size=[height, width])
        warped_events = np.stack([to_be_warped_events[:,0], warped_x, warped_y, to_be_warped_events[:,-1]], 1)
        mask = events_bounds_mask(warped_x, warped_y, 0, width, 0, height)
        warped_events = warped_events[mask.astype(bool)]
        # warped_all_events = np.concatenate([warped_all_events, warped_events], axis=0)
        org_timestamps = np.concatenate([org_timestamps, events[:,0]], axis=0)
        org_timestamps = org_timestamps[mask.astype(bool)]
        
        to_be_warped_events = warped_events
        to_be_warped_events[:,0]==to_be_warped_events[-1,0]
    
    warped_events[:,0] = org_timestamps
    return warped_events   



    

if __name__ == "__main__":
    
    NE = 15000
    height = 180
    width = 240
    path_to_event_data = '/home/sl220/Documents/data/COCO_test2/COCO_upsampled_test2/sequence_0000000000/events/'
    path_to_flow_data = '/home/sl220/Documents/data/COCO_test2/COCO_upsampled_test2/sequence_0000000000/flow/'
    path_to_frame_data = '/home/sl220/Documents/data/COCO_test2/COCO_upsampled_test2/sequence_0000000000/frames/'
    
    event_files = [ os.path.join(path_to_event_data, fn) for fn in os.listdir(path_to_event_data) if fn.split('.')[-1] == 'npz']
    event_files.sort()
    
    flow_files = [ os.path.join(path_to_flow_data, fn) for fn in os.listdir(path_to_flow_data) if fn.split('.')[-1] == 'npz'] 
    flow_files.sort()
    
    frame_files = [ os.path.join(path_to_frame_data, fn) for fn in os.listdir(path_to_frame_data) if fn.split('.')[-1] == 'png'] 
    frame_files.sort()

    event_window = np.empty((0,4),dtype=np.float32)
    flow_list = []
    frame_list = []
    event_list = []
    for event_file, flow_file, frame_file in zip(event_files, flow_files, frame_files):
        cur_event_window = np.load(event_file, allow_pickle=True) #["arr_0"]
        cur_event_window = np.stack((cur_event_window["t"], cur_event_window["x"], cur_event_window["y"],cur_event_window["p"]), axis=1)
        flow = np.load(flow_file, allow_pickle=True)["flow"]
        # img = np.float32((cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE))/255.0)  
        img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
        frame_list.append(img)
        if len(event_window) < NE:
            event_list.append(cur_event_window)
            event_window = np.concatenate((event_window, cur_event_window), 0)
            flow_list.append(flow)
        else:
            break
    # Sum the flow map directly
    fused_flow_map = np.array(flow_list).sum(axis=0)

    
    event_patch = events_to_voxel_grid(event_window, 
                                        num_bins=1,
                                        width=width,
                                        height=height)
    # event_patch = event_preprocess(event_patch, mode='std', filter_hot_pixel=False)

    
    flow_net = FlowNet([height, width], device='cuda:0')
    F_1_0 = flow_net.generate_flow([frame_list[0], frame_list[-1]])
    
    # ----------- warp events directly using summed / generated flow map---------------#
    warped_x, warped_y = warp_events_flow_uv_torch(torch.from_numpy(event_window).float(), torch.from_numpy(F_1_0).float(), img_size=[height,width], t0=None)
    warped_events = np.stack([event_window[:,0], warped_x, warped_y, event_window[:,-1]], 1)
    mask = events_bounds_mask(warped_x, warped_y, 0, width, 0, height)
    warped_events = warped_events[mask.astype(bool)]
    
    # ----------- warp upsampled events iterately  ---------------#
    # warped_events = warp_event_list(event_list, flow_list, height, width)
    
    # ---------- To voxel grid ----------- #
    warped_event_patch = events_to_voxel_grid(warped_events, 
                                        num_bins=1,
                                        width=width,
                                        height=height)

    plt.subplot(1,2,1)
    plt.imshow(flow_list[-1][0])
    plt.subplot(1,2,2)
    plt.imshow(F_1_0[0])
    plt.show()
    
    plt.subplot(1,2,1)
    plt.imshow(event_patch[0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(warped_event_patch[0], cmap='gray')
    plt.show()
    

    plt.subplot(1,2,1)
    plt.imshow(frame_list[0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(frame_list[-1], cmap='gray')
    plt.show()
        