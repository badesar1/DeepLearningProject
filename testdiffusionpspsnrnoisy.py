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
    net = DnCNN(in_channels=1, out_channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob('/gpfs/data/fieremanslab/Ben/data/train/30_noisy.nii')
    #files_source = glob.glob(os.path.join(opt.test_data, 'FGATIR_MID00550_nophase.mat'))
    files_source.sort()

    print(files_source)
    # send model to cuda device
    #model = nn.DataParallel(net, device_ids=device_ids).cuds()
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    # process data
    psnr_test = 0
    for f in files_source:
        # image
        fname = os.path.splitext(f)[0]

        nii = ni.load(f)
        Img = np.array(nii.dataobj)

        # mat = sio.loadmat(f)
        # Img = mat['Ix']

        print(Img.shape)
        print(Img.dtype)
        #Imgn = np.absolute(Img)
        #Img = np.absolute(Img)

        # nii = ni.load(f)
        # Img = np.array(nii.dataobj)
        #Img = 10*np.random.randn(Img.shape[0],Img.shape[1],Img.shape[2])

        ndims = Img.ndim
        shape = Img.shape
        if ndims == 4:
        	Img = np.reshape(Img, (Img.shape[0], Img.shape[1], Img.shape[2]*Img.shape[3]))

        OutImg = np.zeros(Img.shape)
        NoiseImg = np.zeros(Img.shape)	
        InNoisy = np.zeros(Img.shape)
        Sigma = np.zeros(Img.shape)

        #Img = np.ones(Img.shape)

        for s in tqdm(range(Img.shape[-1])):
            Slice, ratio = normalize(Img[:,:,s])
            Slice = np.float32(Slice[:,:])

            Slice = np.expand_dims(Slice, 0)
            Slice = np.expand_dims(Slice, 1)
            ISource = torch.Tensor(Slice)

            snr = opt.test_noiseL
            sigma = 1./snr
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=sigma)
            #inoise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=sigma)
            #INoisy = torch.sqrt((ISource + rnoise)**2 + inoise**2)
            INoisy = ISource #+ noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            #ISource, INoisy = Variable(ISource), Variable(INoisy)
            with torch.no_grad(): 
                model_out = model(INoisy)
                #print(model_out.size())
                noisemap = model_out[0,0,:,:]
                Out = INoisy-noisemap
                noisemap = np.squeeze(noisemap)
                Outs = np.squeeze(Out)
            #Outs = Outs*255*ratio
            Outs = Outs*ratio
            OutImg[:,:,s] = Outs.cpu()
            NoiseImg[:,:,s] = noisemap.cpu()*ratio
            InNoisy[:,:,s] = INoisy.cpu()*ratio
            Sigma[:,:,s] = ratio/snr*torch.ones(Outs.size()).cpu()

#            if s == 90:
#                import matplotlib.pyplot as plt
#                plt.imshow(np.squeeze(Slice))
#                plt.show()
#                plt.imshow(Outs.cpu())
#                plt.show()

            psnr = batch_PSNR(Out, ISource, 1.)
            psnr_test += psnr
            print(" PSNR %f" % (psnr), end='\r')

        if ndims == 4:
        	OutImg = np.reshape(OutImg,(shape[0],shape[1],shape[2],shape[3]))
        	NoiseImg = np.reshape(NoiseImg,(shape[0],shape[1],shape[2],shape[3]))

        # OutImg = OutImg.astype('uint16')
        # OutImg[OutImg > 65500] = 0    

        # niout = ni.Nifti1Image(OutImg, nii.affine, nii.header)
        # ni.save(niout, os.path.join(opt.test_data, fname + '_dncnnB_FG.nii'))
        # niout = ni.Nifti1Image(NoiseImg, nii.affine, nii.header)
        # ni.save(niout, os.path.join(opt.test_data, fname + '_dncnnB_FG_noisemap.nii'))
        opt.test_noiseL = 'MR'
        outdir = '/gpfs/data/fieremanslab/Ben/DnCNN/testresults_0421'
        mdic = {"Ix": OutImg}
        sio.savemat(os.path.join(outdir, '30noisy_dncnn' + opt.logdir[-4:] + 'AG' + str(opt.test_noiseL) + 'denoised.mat'), mdic)
        mdic = {"Ix": NoiseImg}
        sio.savemat(os.path.join(outdir, '30noisy_dncnn' + opt.logdir[-4:] + 'AG' + str(opt.test_noiseL) + 'noisemap.mat'), mdic)
        mdic = {"Ix": InNoisy}
        sio.savemat(os.path.join(outdir, '30noisy_dncnn' + opt.logdir[-4:] + 'AG' + str(opt.test_noiseL) + 'noisyinput.mat'), mdic)
        mdic = {"Ix": Sigma}
        sio.savemat(os.path.join(outdir, '30noisy_dncnn' + opt.logdir[-4:] + 'AG' + str(opt.test_noiseL) + 'Sigma.mat'), mdic)

    psnr_test /= (len(files_source)*Img.shape[-1])
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
