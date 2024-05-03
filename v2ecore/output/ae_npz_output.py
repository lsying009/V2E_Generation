import numpy as np
import logging
from engineering_notation import EngNumber  # only from pip
import atexit
import os

logger = logging.getLogger(__name__)

class DVSNpzOutput:
    '''
    outputs text format DVS event window to file '.npz' format
    
    The RPG DVS npz file datatset looks like this. 
    
    event voxel grid (B,H,W) B -- number of bins
        
    '''

    def __init__(self, filepath: str):
        self.filepath = filepath
        # edit below to match your device from https://inivation.com/support/software/fileformat/#aedat-20
        logging.info('opening text DVS output file {}'.format(filepath))
        atexit.register(self.cleanup)
        self.sizex=240
        self.sizey=180 # adjust to your needs
        self.t, self.x, self.y, self.p = [],[],[],[]
        # self.events = np.empty((0,4))
        self.event_window_idx = -1
        self.numEventsWritten = 0


    def cleanup(self):
        return

        
    def save_npz(self):
        if self.filepath and self.event_window_idx >= 0:
            # print(self.event_window_idx)
            # print(self.event_window_idx, len(self.t), min(self.t), max(self.t))
            np.savez_compressed(os.path.join(self.filepath, 'events_{:010d}.npz'.format(self.event_window_idx)), t=self.t, x=self.x, y=self.y, p=self.p)
            # np.savez_compressed(os.path.join(self.filepath, 'events_{:010d}.npz'.format(self.event_window_idx)), events=self.events)
            # print("Closing {} after writing {} events".format(os.path.join(self.filepath, 'events_{:010d}.npz'.format(self.event_window_idx)), self.numEventsWritten))
            logger.info("Closing {} after writing {} events".format(self.filepath, EngNumber(self.numEventsWritten)))


    def appendEvents(self, events: np.ndarray, event_window_idx:int):
        if self.filepath is None:
            raise Exception('output file closed already')

        # if len(events) == 0:
        #     return
        # print('self.event_window_idx', self.event_window_idx)
        # if event_window_idx != self.event_window_idx:
        #     self.save_npz()
        #     self.event_window_idx = event_window_idx
        #     self.numEventsWritten = 0
        #     self.t, self.x, self.y, self.p = [],[],[],[]
        #     # self.events = np.empty((0,4))
            
            
        n = events.shape[0]
        t = (events[:, 0]).astype(np.float32)
        x = events[:, 1].astype(np.int16)
        y = events[:, 2].astype(np.int16)
        p = ((events[:, 3] + 1) / 2).astype(np.bool8) # go from -1/+1 to 0,1, before it is int16

        self.t = np.concatenate((self.t, t), 0)
        self.x = np.concatenate((self.x, x), 0)
        self.y = np.concatenate((self.y, y), 0)
        self.p = np.concatenate((self.p, p), 0)
        # self.events = np.concatenate((self.events, events.astype(np.float32)), 0)

        self.numEventsWritten += n

        # print('self.event_window_idx', self.event_window_idx, self.numEventsWritten)
        if event_window_idx != self.event_window_idx:
            self.event_window_idx = event_window_idx
            self.save_npz()
            self.numEventsWritten = 0
            self.t, self.x, self.y, self.p = [],[],[],[]

