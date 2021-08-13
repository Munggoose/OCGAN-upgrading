import os 
import time
import numpy as np
import torchvision.utils as vutils
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import visdom
import sys

# sys.append()

class Visualizer():

    def __init__(self,opt):
        self.win_size = 256
        self.opt = opt
        self.ip = '127.0.0.1'
        self.port = '8097'
        self.vis = visdom.Visdom(server=self.ip,port= self.port)
    

    @staticmethod
    def normalize(inp):
        return (inp-inp.min())/(inp.max() - inp.min() + 1e-5)
    
    def plot_current_acc(self,epoch,counter_ratio,acc):
        if not hasattr(self, 'acc_data') or self.acc_data is None:
            self.acc_data = {'X': [], 'Y': [], 'legend': ['acc']}
        self.acc_data['X'].append(epoch/counter_ratio)
        self.acc_data['Y'].append(acc)
        self.vis.line(
            X=np.array(self.acc_data['X']),
            Y=np.array(self.acc_data['Y']),
            opts={
                'title': 'model ACC',
                'legend': self.acc_data['legend'],
                'xlabel': f'Epoch deivice{counter_ratio}',
                'ylabel': 'ACC',
                'markersize': 5
            },
            win=7
        )

    def plot_current_errors(self, epoch:int, counter_ratio:float, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """

        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': 'model loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=4
        )

    def display_current_images(self, reals, fakes, fixed=None):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        # reals = self.normalize(reals.detached.cpu().numpy())
        # fakes = self.normalize(fakes.detached.cpu().numpy())
        # fixed = self.normalize(fixed.detached.cpu().numpy())
        reals = self.normalize(reals.detach().cpu().numpy())
        fakes = self.normalize(fakes.detach().cpu().numpy())
        if fixed is not None:
            fixed = self.normalize(fixed.detach().cpu().numpy())

        self.vis.images(reals, win=1, opts={'title': 'Inputs'})
        self.vis.images(fakes, win=2, opts={'title': 'Reconstruct'})
        if fixed is not None:
            self.vis.images(fixed, win=3, opts={'title': 'L2_generator'})


    def display_fixed_images(self, input, output):
        input = self.normalize(input.detach().cpu().numpy())
        output = self.normalize(output.detach().cpu().numpy())
        self.vis.images(input, win=4, opts={'title': 'Fixed'})
        self.vis.images(output, win=5, opts={'title': 'test_output'})

    def print_current(self):
        pass