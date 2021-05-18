from __future__ import print_function

import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models_psloss import DnCNN
import nibabel as ni
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from DWIdataset import prepare_data, Dataset

import scipy.io as sio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    ratio = np.amax(data) #/ 255
    data = (data / ratio) #.astype('uint8')/255.
    return data, ratio

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(in_channels=5, out_channels=5, num_of_layers=20)
    device_ids = [0]

    # load data info
    print('Loading data info ...\n')
    dataset_train = Dataset(train=False, fname='p50c5_3')

    # send model to cuda device
    #model = nn.DataParallel(net, device_ids=device_ids).cuds()
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join('logs/DWItraining_P50dc5_MSE_b64_l20_nl50_6', 'net.pth')))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    # process data
    with torch.no_grad():
        img_val = dataset_train[0]
        img_val = img_val.permute(0,3,1,2)
        noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=20/255.)        
        imgn_val = img_val + noise
        out_val = model(imgn_val.cuda())
        denoised = torch.clamp(imgn_val.cuda() - out_val.cuda(), 0., 1.)

    outdir = '/gpfs/data/fieremanslab/Ben/DnCNN/testresults_05_5ch_2'
    mdic = {"Ix": denoised.detach().cpu().numpy()}
    sio.savemat(os.path.join(outdir, '5chcnn_denoised.mat'), mdic)
    mdic = {"Ix": out_val.detach().cpu().numpy()}
    sio.savemat(os.path.join(outdir, '5chcnn_noisemap.mat'), mdic)
    mdic = {"Ix": img_val.detach().cpu().numpy()}
    sio.savemat(os.path.join(outdir, '5chcnn_input.mat'), mdic)

    net = DnCNN(in_channels=1, out_channels=1, num_of_layers=17)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join('logs/DnCNN-S-10', 'net.pth')))
    model.eval()
    with torch.no_grad():
        img_val = dataset_train[0]
        img_val = img_val.permute(0,3,1,2)
        noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=20/255.)        
        imgn_val = img_val + noise
        Denoised = torch.zeros_like(imgn_val)
        Out_val = torch.zeros_like(imgn_val)
        for i in range(5):
            out_val = model(imgn_val[0,i,:,:].unsqueeze(0).unsqueeze(0))
            denoised = torch.clamp(imgn_val[0,i,:,:].cuda() - out_val, 0., 1.)
            print(imgn_val.size())
            print(denoised.size())
            Out_val[0,i,:,:] = out_val
            Denoised[0,i,:,:] = denoised

    mdic = {"Ix": Denoised.detach().cpu().numpy()}
    sio.savemat(os.path.join(outdir, '1chcnn_denoised.mat'), mdic)
    mdic = {"Ix": Out_val.detach().cpu().numpy()}
    sio.savemat(os.path.join(outdir, '1chcnn_noisemap.mat'), mdic)
    mdic = {"Ix": imgn_val.detach().cpu().numpy()}
    sio.savemat(os.path.join(outdir, '1chcnn_noisyinput.mat'), mdic)


if __name__ == "__main__":
    main()
