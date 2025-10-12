import h5py
import time
import torch
import os
import scipy.io as sio
from pathlib import Path
from datasets.data import create_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils.util import AverageMeter
from models import *
import numpy as np
from collections import defaultdict
from pytorch_wavelets import DWTForward, DWTInverse
from models.loss_utils import get_loss
from utils.load_weight import module_load

def create_model(config, device, model_name=None):
    if model_name is None:
        model_name = config.model_name
    model = None
    if 'BiWaveDiffSNTBBFDN' == model_name:
        if config.step == 'S1_model':
            model = BiWaveDiffSNTBBFDNS1(config.pan_channels, config.lms_channels, config.hid_channels, config.layers)
        elif config.step == 'S2_model':
            model = BiWaveDiffSNTBBFDNS2(config.pan_channels, config.lms_channels, config.hid_channels, config.layers,
                                        config.n_denoise_res, config.linear_start, config.linear_end, config.timesteps)
        elif config.step == 'S3_model':
            model = BiWaveDiffSNTBBFDNS3(config.pan_channels, config.lms_channels, config.hid_channels, config.layers,
                                        config.n_denoise_res, config.linear_start, config.linear_end, config.timesteps)
    else:
        assert f'{model_name} not supported now.'
    return model.to(device)

class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.writer = None
        dataset_name = config.dataset_name
        self.model_name = config.model_name
        self.debug = config.debug
        if not self.debug:
            run_time = logger.handlers[0].baseFilename.split('/')[-1][:-4]
            self.run_time = run_time
            print(f"run_time: {run_time}")
            weights_save_path = Path(self.config.work_dir) / config.model_name / config.version / config.weights_path / dataset_name
            if config.step == 'S1_model':
                self.weights_save_path = weights_save_path / 'S1_model'
                self.pre_train_path = None
            elif config.step == 'S2_model':
                self.weights_save_path = weights_save_path / 'S2_model'
                self.pre_train_path = weights_save_path / 'S1_model' / config.pre_train_model
            elif config.step == 'S3_model':
                if self.config.wave_loss == True:
                    self.weights_save_path = weights_save_path / 'S3_model' / 'wave_loss'
                else:
                    self.weights_save_path = weights_save_path / 'S3_model'
                self.pre_train_path = weights_save_path / 'S2_model' / config.pre_train_model
            print(f"weights_save_path: {self.weights_save_path}")
            self.weights_save_path.mkdir(exist_ok=True, parents=True)
            tb_log_path = Path(self.config.work_dir) / config.model_name / config.version / config.tb_log_path / dataset_name
            tb_log_path.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(str(tb_log_path))

        self.epoch_num = config.epoch_num
        self.train_loader, self.val_loader = create_loaders(config)
        base_lr = float(config.base_lr)
        weight_decay = float(config.weight_decay)
        self.device = torch.device(config.device)
        self.model = create_model(config, self.device)
        step_size, gamma = int(config.step_size), float(config.gamma)
        self.criterion = nn.L1Loss()
        self.ssim = get_loss('l1ssim', config.lms_channels)
        if config.step == 'S2_model':
            print(f"load_weights_from: {self.pre_train_path}")
            S1_weights = torch.load(self.pre_train_path)
            model_weights = S1_weights['model']
            net_weights_s1 = {k.replace('net.', ''): v for k, v in model_weights.items() if 'net' in k}
            le_weights_s1 = {k.replace('LE.', ''): v for k, v in model_weights.items() if 'LE' in k}
            self.model.net.load_state_dict(net_weights_s1)
            self.model.LE.load_state_dict(le_weights_s1)
        elif config.step == 'S3_model':
            print(f"load_weights_from: {self.pre_train_path}")
            S2_weights = torch.load(self.pre_train_path)
            model_weights = S2_weights['model']
            self.model.load_state_dict(model_weights)
            # if self.config.wave_loss == True:
            #     self.dwt = DWTForward(wave='db1', mode='symmetric').to(self.device)

        if config.step == 'S2_model':
            self.optimizer = optim.Adam(self.model.DM.parameters(), lr=base_lr, betas=(0.9, 0.999))
            for param in self.model.LE.parameters():
                param.requires_grad = False
            for param in self.model.net.parameters():
                param.requires_grad = False
        else:
            if self.config.wave_loss == True:
                self.dwt = DWTForward(wave='db1', mode='symmetric').to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return

    def train_all(self):
        print('Start training...')
        epoch_time = AverageMeter()
        end = time.time()

        ckpt = self.config.save_epoch
        model, optimizer, device = self.model, self.optimizer, self.device
        self.model.train()
        print('-------------------', self.config.step, ' Train-------------------')
        for epoch in range(self.epoch_num):
            epoch += 1
            epoch_train_loss = []
            for iteration, batch in enumerate(self.train_loader, 1):
                gt, lms, ms, pan_hp, pan = Variable(batch[0], requires_grad=False).to(device), \
                    Variable(batch[1]).to(device), \
                    Variable(batch[2]).to(device), \
                    batch[3], \
                    Variable(batch[4]).to(device)
                optimizer.zero_grad()  # fixed
                if self.config.step == 'S1_model':
                    out = model(lms, ms, pan, gt)
                    loss = self.criterion(out, gt)  # compute loss
                elif self.config.step == 'S2_model':
                    prior, prior_DM = model(lms, ms, pan, gt)
                    loss = self.criterion(prior_DM, prior)  # compute loss
                elif self.config.step == 'S3_model':
                    out, prior, prior_DM = model(lms, ms, pan, gt)
                    loss = self.criterion(out, gt)  # compute loss
                epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

                loss.backward()  # fixed
                # nn.utils.clip_grad_norm_(self.model.parameters(), 1e-10)
                optimizer.step()  # fixed
            self.scheduler.step()

            t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
            self.logger.info('Epoch: {}/{} training loss:{:.7f}'.format(epoch, self.epoch_num, t_loss))

            if self.writer:
                self.writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
            self.validate()
            if epoch % ckpt == 0 and not self.debug:
                self.save_checkpoint(epoch, self.model)
            epoch_time.update(time.time() - end)
            end = time.time()
            remain_time = self.calc_remain_time(epoch, epoch_time)
            self.logger.info(f"remain {remain_time}")
        return

    def validate(self):
        epoch_val_loss = []
        model, device = self.model, self.device
        # self.model.eval()
            
        with torch.no_grad():
            for iteration, batch in enumerate(self.val_loader, 1):
                gt, lms, ms, _, pan = Variable(batch[0], requires_grad=False).to(device), \
                    Variable(batch[1]).to(device), \
                    Variable(batch[2]).to(device), \
                    batch[3], \
                    Variable(batch[4]).to(device)
                if self.config.step == 'S1_model':
                    out = model(lms, ms, pan, gt)
                    # loss, _ = self.ssim(out, gt)
                    loss = self.criterion(out, gt)  # compute loss
                elif self.config.step == 'S2_model':
                    prior, prior_DM = model(lms, ms, pan, gt)
                    loss = self.criterion(prior_DM, prior)  # compute loss
                elif self.config.step == 'S3_model':
                    out, prior, prior_DM = model(lms, ms, pan, gt)
                    # loss, _ = self.ssim(out, gt)
                    loss = self.criterion(out, gt)  # compute loss
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        self.logger.info('validate loss: {:.7f}'.format(v_loss))
        return

    def save_checkpoint(self, epoch, model, name=None):
        if name is None:
            model_out_path = str(self.weights_save_path / f'LFormer_epoch{epoch}.pth')
        else:
            model_out_path = str(self.weights_save_path / f'{name}_epoch{epoch}.pth')
        ckpt = {'model': model.state_dict(), 'exp_timestamp': self.run_time}
        torch.save(ckpt, model_out_path)
        return

    def calc_remain_time(self, epoch, epoch_time):
        remain_time = (self.epoch_num - epoch) * epoch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        return remain_time


class Tester:
    def __init__(self, config):
        self.config = config
        dataset_name = config.dataset_name
        assert config.test_mode in ('reduced', 'full')
        test_data_path = str(Path(config.test_data_path))
        self.dataset = h5py.File(test_data_path, 'r')
        self.model_name = config.model_name
        if 'gf2' in test_data_path:
            self.max_value = 1023.0
        elif 'cave' in test_data_path:
            self.max_value = 1.0
        elif 'harvard' in test_data_path:
            self.max_value = 1.0
        else:
            self.max_value = 2047.0
        self.device = torch.device(config.device)
        self.model = create_model(config, self.device)
        if dataset_name == 'WV2':
            weights_save_path = Path(config.work_dir) / config.model_name / config.version / config.weights_path / 'WV3'
        else:
            weights_save_path = Path(config.work_dir) / config.model_name / config.version / config.weights_path / dataset_name
        if config.step == 'S1_model':
            self.weights_save_path = weights_save_path / 'S1_model' / config.test_weight_path
        elif config.step == 'S2_model':
            self.weights_save_path = weights_save_path / 'S2_model' / config.test_weight_path
        elif config.step == 'S3_model':
            self.weights_save_path = weights_save_path / 'S3_model' / config.test_weight_path
            if config.wave_loss ==  True:
                self.weights_save_path = weights_save_path / 'S3_model' / 'wave_loss' / config.test_weight_path
        ckpt = torch.load(self.weights_save_path, map_location=self.device)
        print(f"loading weight: {self.weights_save_path}")
        self.model.load_state_dict(ckpt['model'])
        # self.model.load_state_dict(ckpt['state_dict'])
        # module_load(weight_path, self.model, device)
        save_path = Path(self.config.work_dir) / config.model_name / config.version / config.results_path / dataset_name / config.test_mode / config.test_weight_path.split('.')[0]
        save_path.mkdir(exist_ok=True, parents=True)
        self.save_path = save_path
        return

    def test(self, analyse_fms=False):
        features = defaultdict(list)

        def get_features(name):
            def hook(model, input, output):
                features[name].append(output.detach().cpu().numpy())
            return hook

        dataset, model = self.dataset, self.model
        keys = list(dataset.keys())
        if analyse_fms:
            rijabs = model.rijabs
            for i in range(model.block_num):
                cur_ln = f'rijab_{i}'
                rijab_i = getattr(rijabs, cur_ln)
                rijab_i.register_forward_hook(get_features(cur_ln))
                rijab_i.sat1.register_forward_hook(get_features(cur_ln + '.sat1'))
                rijab_i.sat3.register_forward_hook(get_features(cur_ln + '.sat3'))
                rijab_i.sat5.register_forward_hook(get_features(cur_ln + '.sat5'))


        if self.config.test_mode == 'reduced':
            gt = np.array(dataset[keys[0]], dtype=np.float32) / self.max_value
            gt = torch.from_numpy(gt).float().to(self.device)
            ms = np.array(dataset[keys[2]], dtype=np.float32) / self.max_value
            lms = np.array(dataset[keys[1]], dtype=np.float32) / self.max_value
            pan = np.array(dataset[keys[3]], dtype=np.float32) / self.max_value
        else:
            ms = np.array(dataset[keys[1]], dtype=np.float32) / self.max_value
            lms = np.array(dataset[keys[0]], dtype=np.float32) / self.max_value
            pan = np.array(dataset[keys[2]], dtype=np.float32) / self.max_value

        ms = torch.from_numpy(ms).float().to(self.device)
        lms = torch.from_numpy(lms).float().to(self.device)
        pan = torch.from_numpy(pan).float().to(self.device)
        if self.config.dataset_name == 'cave_x8':
            ms = F.interpolate(ms, size=(128, 128), mode='bilinear', align_corners=True)
        if self.config.dataset_name == 'harvard_x8':
            ms = F.interpolate(ms, size=(250, 250), mode='bilinear', align_corners=True)
        model.eval()
        print(f"save files to {self.save_path}")
        with torch.no_grad():
            for i in range(len(pan)):
                print(i)
                if i == 100:
                    return
                if self.config.step == 'S1_model':
                    # out, y = model(lms[i:i+1], ms[i:i+1], pan[i:i+1], gt[i:i+1])
                    out, y = model(lms[i:i+1], pan[i:i+1])
                elif self.config.step == 'S2_model':
                    out = self.model(lms[i:i+1], ms[i:i+1], pan[i:i+1])
                elif self.config.step == 'S3_model':
                    # out, out_H, out_LL, in_H, input_wavelet_LL = self.model(lms[i:i+1], ms[i:i+1], pan[i:i+1])
                    # out, y = self.model(lms[i:i+1], ms[i:i+1], pan[i:i+1])
                    out = self.model(lms[i:i+1], ms[i:i+1], pan[i:i+1])
                I_SR = torch.squeeze(out * self.max_value).cpu().detach().numpy()  # BxCxHxW
                sio.savemat(str(self.save_path / f'output_mulExm_{i}.mat'), {'I_SR': I_SR.transpose(1, 2, 0)})
                # features = features.cpu().detach().numpy()
                # y = y.cpu().detach().numpy()
                # diff_in = diff_in.cpu().detach().numpy()
                # diff_out = diff_out.cpu().detach().numpy()
                # out_H = out_H.cpu().detach().numpy()
                # out_LL = out_LL.cpu().detach().numpy()
                # in_H = in_H.cpu().detach().numpy()
                # input_wavelet_LL = input_wavelet_LL.cpu().detach().numpy()
                # # print(features.shape)
                # os.makedirs(str(self.save_path / 'features'), exist_ok=True)
                # sio.savemat(str(self.save_path / 'features' / f'blockout_{i}.mat'), {'out_feature': y})
                # sio.savemat(str(self.save_path / 'features' / f'diff_{i}.mat'), {'diff_in': diff_in, 'diff_out': diff_out})
                # sio.savemat(str(self.save_path / 'features' / f'H_{i}.mat'), {'H_in': in_H, 'H_out': out_H})
                # sio.savemat(str(self.save_path / 'features' / f'LL_{i}.mat'), {'LL_in': input_wavelet_LL, 'LL_out': out_LL})
        return
