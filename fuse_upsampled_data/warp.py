
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 (at t1) and frame I1,
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` \
            to the backwarping
        block.
    """

    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda).
        """

        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` \
            to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1 at I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """

        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


class forwardWarp(nn.Module):
    """
    A class for creating a forwardwarping object.

    This is used for forwardwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 (at t0) and frame I0,
    it generates I1 <-- forwardwarp(F_0_1, I0).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` \
            to the forwardwarping
        block.
    """

    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda).
        """

        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` \
            to the backwarping
        block.
        I1  = forwardwarp(I0, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I0.
            flow : tensor
                optical flow from I0 and I1 at I0: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """

        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() - u
        y = self.gridY.unsqueeze(0).expand_as(v).float() - v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = F.grid_sample(img, grid)
        return imgOut



def forward_warp(I0, F_0_1, device="cuda:0"):
    # I0: np.array (H,W) or (C,H,W) or tensor, dtype=float, range [0,1]
    # F_0_1: np.array or tensor (2, H, W)
    # Load your optical flow field F0->1 and frame 0 (I0)
    if not isinstance(I0, torch.Tensor):
        I0 = torch.from_numpy(I0).to(device)
        I0 = I0.unsqueeze(0)
        if len(I0.size()) < 4:
            I0 = I0.unsqueeze(0)
        
    if not isinstance(F_0_1, torch.Tensor):
        F_0_1 = torch.from_numpy(F_0_1).to(device)
        if len(F_0_1.size())<4:
            F_0_1 = F_0_1.unsqueeze(0) 
            
    _, _, H, W = F_0_1.shape
    # Calculate the grid of coordinates for frame 1 (I1)
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_x = grid_x.float()
    grid_y = grid_y.float()
    
    
    coordinates = torch.stack((grid_x, grid_y), 0).unsqueeze(0)
    coordinates = coordinates.to(device)
    # Apply optical flow to the coordinates
    warped_coordinates = coordinates - F_0_1
    
    # range -1 to 1
    warped_coordinates[0,0] = 2*(warped_coordinates[0,0]/W - 0.5)
    warped_coordinates[0,1] = 2*(warped_coordinates[0,1]/H - 0.5)
    
    # Sample pixels using bilinear interpolation.
    ## 1x1xHxW 1xHxWx2 
    I1 = F.grid_sample(I0, warped_coordinates.permute(0,2,3,1), align_corners=True, padding_mode='reflection')
    # I1 /= 255.
    
    return I1.squeeze().cpu().numpy()


def back_warp(I1, F_0_1, device="cuda:0"):
    # I0: np.array (H,W) or (C,H,W) or tensor, dtype=float, range [0,1]
    # F_0_1: np.array or tensor (2, H, W)
    # Load your optical flow field F0->1 and frame 0 (I0)
    if not isinstance(I1, torch.Tensor):
        I1 = torch.from_numpy(I1).to(device)
        I1 = I1.unsqueeze(0)
        if len(I1.size()) < 4:
            I1 = I1.unsqueeze(0)
        
    if not isinstance(F_0_1, torch.Tensor):
        F_0_1 = torch.from_numpy(F_0_1).to(device)
        if len(F_0_1.size())<4:
            F_0_1 = F_0_1.unsqueeze(0) 
            
    _, _, H, W = F_0_1.shape
    # Calculate the grid of coordinates for frame 1 (I1)
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_x = grid_x.float()
    grid_y = grid_y.float()
    
    
    coordinates = torch.stack((grid_x, grid_y), 0).unsqueeze(0)
    coordinates = coordinates.to(device)
    # Apply optical flow to the coordinates
    warped_coordinates = coordinates + F_0_1
    
    # range -1 to 1
    warped_coordinates[0,0] = 2*(warped_coordinates[0,0]/W - 0.5)
    warped_coordinates[0,1] = 2*(warped_coordinates[0,1]/H - 0.5)
    
    # Sample pixels using bilinear interpolation.
    ## 1x1xHxW 1xHxWx2 
    I0 = F.grid_sample(I1, warped_coordinates.permute(0,2,3,1), align_corners=True, padding_mode='reflection')
    # I1 /= 255.
    
    return I0.squeeze().cpu().numpy()
