import os
import random
import numpy as np
import scipy.misc
from PIL import Image
import torch
import scipy.io as sio
import scipy.misc
from adamp import AdamP
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
import datetime as datetimes
import time as times
import math
import sys
from util import *
from ops import *
import shutil
from torchvision.utils import save_image
from gradient_variance_loss import GradientVariance

time = datetimes.datetime.now().strftime('%m.%d-%H:%M:%S')

class Trainer():
    def __init__(self, model, cfg):

        self.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

        self.Network = model(group=cfg.group)

        if cfg.loss_fn in ["MSE"]:
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]:
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.grad_criterion = GradientVariance(patch_size=8, device=self.device).to(self.device)

        self.optim = AdamP(filter(lambda p: p.requires_grad, self.Network.parameters()),cfg.lr)


        self.train_data = TrainDataset(cfg.train_data_path,
                                       scale=cfg.scale,
                                       size=cfg.patch_size)

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)


        self.Network = self.Network.to(self.device)
        self.loss_fn = self.loss_fn

        self.folder_name = str(cfg.loss_fn) + '_' + str(cfg.batch_size) + '_' + str(cfg.max_steps)[0] + 'K' + '_' + \
                           str(cfg.lr) + '_'  +  str(cfg.upscale) + 'to'+ str(cfg.scale)

        checkpoint_folder = 'logs/{}/checkpoints'.format(self.folder_name)
        mkdir(checkpoint_folder)

        if cfg.resume:
            PATH = os.path.join("logs", self.folder_name, "checkpoints")
            all_checkpoints = list(sorted(os.listdir(PATH)))

            if len(all_checkpoints) > 0:
                PATH = os.path.join(PATH, all_checkpoints[-1])
                print("=> loading checkpoint '{}'".format(PATH))
                checkpoint = torch.load(PATH)
                self.Network.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                self.step = checkpoint['step']
                self.best_psnr = checkpoint["best_psnr"]
            else:
                print("=> no checkpoint at '{}'".format(PATH))
                self.best_psnr = 0
                self.step = 0
        else:
            self.best_psnr = 0
            self.step = 0

        self.cfg = cfg


        self.writer = SummaryWriter(log_dir=os.path.join("logs/{}/tensorboard/".format(self.folder_name)))
        if cfg.verbose:
            num_params = 0
            for param in self.Network.parameters():
                num_params += param.nelement()
            print("Number of parameters for scale X{}: {}".format(cfg.scale, num_params))


    def train(self):
        cfg = self.cfg

        #Network = nn.DataParallel(self.Network, device_ids=range(cfg.num_gpu))
        Network = self.Network
        self.mean_content = 0.
        self.mean_l1 = 0.

        learning_rate = cfg.lr
        while True:
            for inputs in self.train_loader:

                self.Network.train()
                total_loss = []

                sc_idx = random.randint(0, len(inputs)-1)
                hr, lr = inputs[sc_idx][0], inputs[sc_idx][1]

                scale1 = hr.size(2) / lr.size(2)
                scale2 = hr.size(3) / lr.size(3)

                hr = hr.to(self.device)
                lr = lr.to(self.device)
                self.Network.set_scale(scale1, scale2)
                sr_main = Network(lr, scale1, scale2)

                l1_loss = self.loss_fn(sr_main, hr)
                loss_grad = 0.01 * self.grad_criterion(sr_main, hr)
                loss = loss_grad + l1_loss

                self.optim.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.Network.parameters(), cfg.clip)
                self.optim.step()

                self.mean_l1 += loss

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate

                self.step += 1
                sys.stdout.write("\r==>>Steps:[%d/ %d] Total:[%.6f] "
                                 % (self.step, cfg.max_steps, loss.item()))
                self.writer.add_scalar('Loss', loss.data.cpu().numpy(), global_step=self.step)

                if cfg.verbose and self.step % cfg.print_interval == 0:
                    with open('logs/{}/'.format(self.folder_name) + 'logs.txt', 'a') as f:
                        PATH = os.path.join('logs/{}/checkpoints/'.format(self.folder_name),
                                            "{}_{:06d}.pth.tar".format(cfg.ckpt_name, self.step))

                        t1 = times.time()


                        mean_psnr = self.evaluate(cfg.valid_data_path, scale=cfg.scale, upscale=cfg.upscale, num_step=self.step)
                        t2 = times.time()

                        self.writer.add_scalar("PSNR_{}x:".format(scale1), mean_psnr, self.step)


                        print('-- PSNR_x{}: {:.5f}  -- Total_Loss: {:.5f}\n'
                                        .format(scale1, mean_psnr, (self.mean_l1) / cfg.print_interval))


                        torch.save({'step': self.step, 'model_state_dict': self.Network.state_dict(),
                                        'optimizer_state_dict': self.optim.state_dict(), 'best_psnr': self.best_psnr}, PATH)
                        f.write('Step: {}'
                                             '--> PSNR_x{}:{:.5f} -->{:.3f}m\n'
                                                .format(self.step, scale1, mean_psnr, ((t2 - t1)/60)))

                    self.mean_l1 = 0.
                    self.mean_content = 0.


            if self.step > cfg.max_steps: break

    def evaluate(self, test_data_dir, scale=2, upscale=3, num_step=0):
        cfg = self.cfg
        mean_psnr = 0

        self.Network.eval()

        test_data = TestDataset(test_data_dir, scale=scale, evaluate=True)
        test_loader = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=True)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0]
            lr = inputs[1]
            name = inputs[2][0]

            lr = lr.to(self.device)
            hr = hr.to(self.device)

            scale1 = hr.size(2) / lr.size(2)
            scale2 = hr.size(3) / lr.size(3)
            self.Network.set_scale(scale1, scale2)
            sr = self.Network(lr, scale1, scale2)

            psnr = calc_psnr(sr, hr, scale, 1, benchmark=True)
            mean_psnr += psnr / len(test_data)

        return mean_psnr


    def load(self, path):
        self.Network.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.Network.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr

    def save_checkpoint(self, is_best, filename='checkpoint.pth.tar'):
        save_path = os.path.join(self.cfg.logdir, self.folder_name) + '/'
        torch.save(self.Network, save_path + filename)
        if is_best:
            shutil.copyfile(save_path + filename, save_path + 'model_best.pth.tar')
