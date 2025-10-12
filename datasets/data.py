import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


def get_edge(data):  # for training
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset(data.Dataset):
    def __init__(self, file_path):
        super(Dataset, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        keys = list(data.keys())
        # print(keys)
        if 'gf2' in file_path:
            max_value = 1023.0
        elif 'cave' in file_path:
            max_value = 1.0
        elif 'harvard' in file_path:
            max_value = 1.0
        else:
            max_value = 2047.0
        # tensor type:
        gt1 = data[keys[0]][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / max_value
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        lms1 = data[keys[1]][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / max_value
        self.lms = torch.from_numpy(lms1)

        ms1 = data[keys[2]][...]  # NxCxHxW
        ms1 = np.array(ms1, dtype=np.float32) / max_value
        self.ms = torch.from_numpy(ms1)
        # ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / max_value  # NxHxWxC
        # ms1_tmp = get_edge(ms1)  # NxHxWxC
        # self.ms_hp = torch.from_numpy(ms1_tmp).permute(0, 3, 1, 2)  # NxCxHxW:

        # pan1 = data['pan'][...]  # Nx1xHxW
        # pan1 = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / max_value  # NxHxWx1
        # pan1 = np.squeeze(pan1, axis=3)  # NxHxW
        # pan_hp_tmp = get_edge(pan1)  # NxHxW
        # pan_hp_tmp = np.expand_dims(pan_hp_tmp, axis=3)  # NxHxWx1
        # self.pan_hp = torch.from_numpy(pan_hp_tmp).permute(0, 3, 1, 2)  # Nx1xHxW:

        pan1 = data[keys[3]][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / max_value  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        # print(self.pan.shape)

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), self.lms[index, :, :, :].float(), \
            self.ms[index, :, :, :].float(), self.pan[index, :, :, :].float(), \
            self.pan[index, :, :, :].float()

    def __len__(self):
        return self.gt.shape[0]


def create_loaders(config):
    batch_size = config.batch_size
    dataset_name = config.dataset_name.lower()
    
    train_data_path = Path(config.train_data_path)
    train_data_path = str(train_data_path)
    train_set = Dataset(train_data_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=config.workers, batch_size=batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    print('Train set ground truth shape', train_set.gt.shape)
    validate_data_path = Path(config.valid_data_path)
    validate_data_path = str(validate_data_path)
    validate_set = Dataset(validate_data_path)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)
    print('Validate set ground truth shape', validate_set.gt.shape)
    return training_data_loader, validate_data_loader

if __name__ == '__main__':
    import argparse
    train_set = Dataset('E:/HJM_Datasets/HISR/cave/x4/test_cave(with_up)x4_rgb.h5')
    # train_set = Dataset('E:/HJM_Datasets/new_pan/training_wv3/train_wv3.h5')
    # train_set = Dataset('E:/HJM_Datasets/new_pan/test data/h5/WV3/reduce_examples/test_wv3_multiExm1.h5')
    # train_set = Dataset('E:/HJM_Datasets/new_pan/test data/h5/WV3/full_examples/test_wv3_OrigScale_multiExm1.h5')
