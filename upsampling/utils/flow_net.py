import os
import shutil
from typing import List, Union
import urllib
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from .const import mean, std
from .model import UNet, backWarp
from .dataset import CropParameters


import os
import shutil
import urllib
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image


class FlowNet:
    def __init__(self, image_dim, device: str):
        self.crop = CropParameters(image_dim[1], image_dim[0], 5)
        self.device = torch.device(device)


        self._load_net_from_checkpoint()

        negmean= [x * -1 for x in mean]
        self.negmean = torch.Tensor([x * -1 for x in mean]).view(3, 1, 1)
        self.negmean = self._move_to_device(self.negmean, self.device)
        revNormalize = transforms.Normalize(mean=negmean, std=std)
        self.TP = transforms.Compose([revNormalize])

        normalize = transforms.Normalize(mean=mean, std=std)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    def _load_net_from_checkpoint(self):
        # ckpt_file = '/rds/general/user/sl220/home/projects/202311my_v2e/upsampling/checkpoint/SuperSloMo.ckpt'
        ckpt_file = 'upsampling/checkpoint/SuperSloMo.ckpt'

        if not os.path.isfile(ckpt_file):
            print('Downloading SuperSlowMo checkpoint to {} ...'.format(ckpt_file))
            g = urllib.request.urlopen('http://rpg.ifi.uzh.ch/data/VID2E/SuperSloMo.ckpt')
            with open(ckpt_file, 'w+b') as ckpt:
                ckpt.write(g.read())
            print('Done with downloading!')
        assert os.path.isfile(ckpt_file)
        

        self.flowComp = UNet(6, 4)
        self._move_to_device(self.flowComp, self.device)
        for param in self.flowComp.parameters():
            param.requires_grad = False

        self.flowBackWarp_dict = dict()


        self.ArbTimeFlowIntrp = UNet(20, 5)
        self._move_to_device(self.ArbTimeFlowIntrp, self.device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.ArbTimeFlowIntrp.load_state_dict(checkpoint['state_dictAT'])
        self.flowComp.load_state_dict(checkpoint['state_dictFC'])
    
    def get_flowBackWarp_module(self, width: int, height: int):
        module = self.flowBackWarp_dict.get((width, height))
        if module is None:
            module  = backWarp(width, height, self.device, False)
            self._move_to_device(module, self.device)
            self.flowBackWarp_dict[(width, height)] = module
        assert module is not None
        return module

    def warp_frame(self, I0, F_1_0):
        if not isinstance(I0, torch.Tensor):
            I0 = self.transform(Image.fromarray((I0).astype('uint8')).convert("RGB"))
            I0 = self._move_to_device(I0, self.device).unsqueeze(0)
        if not isinstance(F_1_0, torch.Tensor):
            F_1_0 = torch.from_numpy(F_1_0)
            if len(F_1_0.size())<4:
                F_1_0 = self._move_to_device(F_1_0, self.device).unsqueeze(0)
        height, width  = I0.shape[-2:]
        flow_back_warp = self.get_flowBackWarp_module(width, height)
        # backwarp function to warp I0 to I1
        I1 = flow_back_warp(I0, F_1_0)
        I1 = self.TP(I1[0]).mean(0).squeeze().cpu().numpy()
        return I1


    def generate_flow(self, img_pair):
        # img_sequence: list of uint8, time_sequence: list
        
        img_pair = [self.transform(Image.fromarray((img).astype('uint8')).convert("RGB")) for img in img_pair]
        img_pair = self._move_to_device(img_pair, self.device)
        # img_pair.to(self.device)
        
        if len(img_pair[0].size()) < 4:
            I0 = torch.unsqueeze(img_pair[0], dim=0)
            I1 = torch.unsqueeze(img_pair[-1], dim=0)
        else:
            I0 = img_pair[0]
            I1 = img_pair[-1]
        I0 = self.crop.pad(I0)
        I1 = self.crop.pad(I1)

        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        flowOut = flowOut[:,:, self.crop.iy0:self.crop.iy1,
                        self.crop.ix0:self.crop.ix1]
        F_0_1 = flowOut[:, :2, :, :].squeeze().cpu().numpy()
        F_1_0 = flowOut[:, 2:, :, :].squeeze().cpu().numpy()
        
        
        return F_0_1, F_1_0
    
    def generate_flow_with_res(self, img_pair):
        # img_sequence: list of uint8, time_sequence: list
        
        img_pair = [self.transform(Image.fromarray((img).astype('uint8')).convert("RGB")) for img in img_pair]
        img_pair = self._move_to_device(img_pair, self.device)
        # img_pair.to(self.device)
        
        if len(img_pair[0].size()) < 4:
            I0 = torch.unsqueeze(img_pair[0], dim=0)
            I1 = torch.unsqueeze(img_pair[-1], dim=0)
        else:
            I0 = img_pair[0]
            I1 = img_pair[-1]
        I0 = self.crop.pad(I0)
        I1 = self.crop.pad(I1)

        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]
        

        # optical flow from It-->I0 and It-->I1 
        # F_t_0 = F_1_0
        F_1_1 = torch.zeros_like(F_1_0)
        
        height, width = I0.shape[-2:]
        flow_back_warp = self.get_flowBackWarp_module(width, height)
        # backwarp function to warp I0/I1 to It
        g_I0_F_t_0 = flow_back_warp(I0, F_1_0)
        # g_I1_F_t_1 = flow_back_warp(I1, F_1_1) # I1
        
        # I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0
        intrpOut = self.ArbTimeFlowIntrp(
            torch.cat((I0, I1, F_0_1, F_1_0, F_1_1, F_1_0, I1, g_I0_F_t_0), dim=1))
        
        # Optical flow residuals, intrpOut[:, :2, :, :]: Delta F_t_0 & intrpOut[:, 2:4, :, :]: Delta F_t_1
        F_1_0_f = intrpOut[:, :2, :, :] + F_1_0
        # F_t_1_f = intrpOut[:, 2:4, :, :] + F_1_1
        # visibility map
        
        F_1_0_f = F_1_0_f[:,:, self.crop.iy0:self.crop.iy1,
                        self.crop.ix0:self.crop.ix1]
        F_1_0_f = F_1_0_f.squeeze().cpu().numpy()
        
        return F_1_0_f

    @staticmethod
    def _to_numpy_image(img: torch.Tensor):
        img = np.clip(img.cpu().numpy(), 0, 1) #.astype(np.uint8) # *255
        img = np.transpose(img, (0, 2, 3, 1))
        return img

    @classmethod
    def _move_to_device(
            cls,
            _input,
            device: torch.device,
            dtype: torch.dtype = None):
        if not torch.cuda.is_available() and not device == torch.device('cpu'):
            warnings.warn("CUDA not available! Input remains on CPU!", Warning)

        if isinstance(_input, torch.nn.Module):
            # Performs in-place modification of the module but we still return for convenience.
            return _input.to(device=device, dtype=dtype)
        if isinstance(_input, torch.Tensor):
            return _input.to(device=device, dtype=dtype)
        if isinstance(_input, list):
            return [cls._move_to_device(v, device=device, dtype=dtype) for v in _input]
        warnings.warn("Instance type '{}' not supported! Input remains on current device!".format(type(_input)), Warning)
        return _input
