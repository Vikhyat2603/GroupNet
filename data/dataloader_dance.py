import os, random, numpy as np, copy

from torch.utils.data import Dataset
import torch


def seq_collate(data):

    (past_traj, future_traj) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'seq': 'dance',
    }

    return data

class DanceDataset(Dataset):
    """Dataloader for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, node_dim=3, n_samples=None, training=True
    ):
        super(DanceDataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if training:
            data_root = 'datasets/dance/train.npy'
        else:
            data_root = 'datasets/dance/test.npy'

        if n_samples is None:
            self.trajs = np.load(data_root)[:,:,:,:node_dim]
        else:
            self.trajs = np.load(data_root)[:,:,:,:node_dim][:n_samples]
        print(f"Loaded {'Train' if training else 'Test'} data with shape: {self.trajs.shape}")
        
        # if training:
        #     self.trajs = self.trajs#[:1500]
        # else:
        #     self.trajs = self.trajs#[:400]

        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        self.traj_norm = self.traj_norm.permute(0,2,1,3)
        # print(self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        past_traj = self.traj_abs[index, :, :self.obs_len, :]
        future_traj = self.traj_abs[index, :, self.obs_len:, :]
        out = [past_traj, future_traj]
        return out
