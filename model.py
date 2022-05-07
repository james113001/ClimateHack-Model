import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn

class Model(nn.Module):
    
    NUM_WARM_UP_IMAGES = 12
    NUM_PREDICTION_TIMESTEPS = 24
    
    
    def __init__(self, batch_input):
        
        self.batch_input = batch_input
        
    def compute_flows(self, **kwargs):
        
        flows = []
        
        for image_i in range(self.NUM_WARM_UP_IMAGES):
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.batch_input[image_i-1], next=self.batch_input[image_i], flow=None, **kwargs)
            flows.append(flow)
        return np.stack(flows).astype(np.float32)
    
    def weighted_average(self, flow):
        return np.average(flow, axis=0, weights=range(1, self.NUM_WARM_UP_IMAGES+1)).astype(np.float32)
        
    def remap_image(self, image, flow):

        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # x map
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
        remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return remapped_image[32:96, 32:96]
    
    def generate(self):
        
        targets = []
        
        start_image = self.batch_input[-1]
        
        flows_default = self.compute_flows(pyr_scale=0.5, levels=3, winsize=10, 
        iterations=10, poly_n=5, poly_sigma=1.2, ####levels=3, winsize=15, iterations=10, poly_sigma=1.2, 
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_default = self.weighted_average(flows_default)
        
        for i in range(self.NUM_PREDICTION_TIMESTEPS):
            remapped_image = self.remap_image(start_image, flow_default * i)
            
            targets.append(remapped_image)
            
        return np.array(targets)