# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy


class DirectionCells(object):

    def __init__(self, options, us=None):
        self.Nd = options.Nd
        self.sigma = options.direction_cell_rf
        self.device = options.device
        self.softmax = torch.nn.Softmax(dim=-1)

        self.angle_min = -np.pi
        self.angle_max = np.pi
        self.angle_range = self.angle_max - self.angle_min

        # Randomly tile direction cell centers across circular space
        np.random.seed(0)
        us = np.random.uniform(self.angle_min, self.angle_max, (self.Nd,))
        self.us = torch.tensor(us)
        # If using a GPU, put on GPU
        self.us = self.us.to(self.device)

    def get_activation(self, ang):
        '''
        Get direction cell activations for a given angle.

        Args:
            ang: 1d angle of shape [batch_size, sequence_length] or
                 [batch_size, sequence_length, 1].

        Returns:
            outputs: Direction cell activations with shape
                     [batch_size, sequence_length, Nd].
        '''
        if ang.dim() == 2:
            ang = ang[:, :, None]
        elif ang.dim() == 3:
            ang = ang.squeeze(-1)[:, :, None]

        d = torch.abs(ang - self.us[None, None, :]).float()

        #make tuining profile circular
        d = torch.minimum(d, self.angle_range - d)

        norm2 = d**2

        #Gaussian tuning curves
        outputs = self.softmax(-norm2 / (2 * self.sigma**2))

        return outputs

    def get_nearest_cell_dir(self, activation, k=3):
        '''
        Decode angle using centers of k maximally active direction cells.

        Args:
            activation: Direction cell activations of shape
                        [batch_size, sequence_length, Nd].
            k: Number of maximally active direction cells with which to decode angle.

        Returns:
            pred_ang: Predicted angle with shape [batch_size, sequence_length, 1].
        '''
        _, idxs = torch.topk(activation, k=k)
        
        #!!!!Note: do not use pred_ang = self.us[idxs].mean(-1), since angles are circular quantities. 
        #!!!!Instead, compute mean angle using atan2 of mean sine and cosine values.
        angles = self.us[idxs]
        pred_ang = torch.atan2(torch.sin(angles).mean(-1), torch.cos(angles).mean(-1))

        return pred_ang[:, :, None]

    def grid_dc(self, dc_outputs, res=64):
        ''' Interpolate direction cell outputs onto a 1d grid'''
        coords = np.linspace(self.angle_min, self.angle_max, res)
        grid = coords[:, None]

        # Convert to numpy
        dc_outputs = dc_outputs.reshape(-1, self.Nd)

        T = dc_outputs.shape[0]
        dc = np.zeros([T, res])
        us = self.us.cpu().numpy()
        for i in range(len(dc_outputs)):
            gridval = scipy.interpolate.griddata(us, dc_outputs[i], grid)
            gridval = np.nan_to_num(gridval, nan=0.0)
            dc[i] = gridval.reshape([res])

        return dc

    def compute_covariance(self, res=60):
        '''Compute circular covariance of direction cell outputs'''
        ang = np.linspace(self.angle_min, self.angle_max, res)
        ang = torch.tensor(ang)[None, :, None]

        # Put on GPU if available
        ang = ang.to(self.device)

        dc_outputs = self.get_activation(ang).reshape(-1, self.Nd).cpu()

        C = dc_outputs @ dc_outputs.T

        Cmean = np.zeros([res])
        for i in range(res):
            Cmean += np.roll(C[i].numpy(), -i)

        Cmean = np.roll(Cmean, res // 2)

        return Cmean
