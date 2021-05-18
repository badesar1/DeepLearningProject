from __future__ import print_function

import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
import nibabel as ni
from utils import *
from tqdm import tqdm

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import pdb

import scipy.io as io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    ratio = np.amax(data) / 255
    data = (data / ratio).astype('uint8')/255.
    return data, ratio

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]

    # load data info
    print('Loading data info ...\n')
    #files_source = glob.glob(os.path.join(opt.test_data, 'IMGRECON_10.mat'))
    files_source = glob.glob(os.path.join(opt.test_data, 'imgreconX_RMT_08182020_10__ysgs_MAG_D0_AC.mat'))
    files_source.sort()

    print(files_source)
    # send model to cuda device
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
        #nii = ni.load(f)
        #Img = np.array(nii.dataobj)
#        import h5py
#        with h5py.File(f, 'r') as file:
#            IPE = np.array(file['orig_mag'])
#            file.close()
            
        mat = io.loadmat(f)
        Imgin = mat['orig_mag']
            #DN1 = np.array(file['DN1'])

        #IPE = IPE.reshape(DN1.shape)
        #IPE = IPE.transpose(4,3,2,1,0)
        #DN1 = DN1.transpose(4,3,2,1,0)

        #DNG = np.stack((DN1['real'], DN1['imag']),axis=-1)
        #Imgin = np.stack((IPE['real'], IPE['imag']),axis=-1)
        ndims = Imgin.ndim
        shape = Imgin.shape
        print(shape)
        print(ndims)

        # import nibabel as nib
        # imgs = nib.Nifti1Image(imgorig_, np.eye(4))
        # nib.save(imgs, os.path.join('/cbi05data/data1/Hamster/Ben/coilstuff','orig.nii'))

        # print(imgorig_.shape)
        # from denoise import MP
        # mp = MP(imgorig_)
        # Signal, Sigma, Npars = mp.process(patch='nonlocal', shrink='frobnorm', nlpatchsize=100)
        # print(Signal.shape)#

        # import scipy.io as sio

        # sio.savemat("/cbi05data/data1/Hamster/Ben/coilstuff/FullySampled_PGSE_TRSE_mppca_nl.mat",{'outimg':Signal})
        # sio.savemat("/cbi05data/data1/Hamster/Ben/coilstuff/FullySampled_PGSE_TRSE_sigma_nl.mat",{'outimg':Sigma})

        # imgspy = nib.Nifti1Image(Signal, np.eye(4))
        # nib.save(imgspy, os.path.join('/cbi05data/data1/Hamster/Ben/coilstuff','testpy.nii'))
        # imgsigpy = nib.Nifti1Image(Sigma, np.eye(4))
        # nib.save(imgsigpy, os.path.join('/cbi05data/data1/Hamster/Ben/coilstuff','testsigpy.nii'))

#        import matplotlib
#        matplotlib.use('TkAgg')
#        import matplotlib.pyplot as plt
#
#        fig, axs = plt.subplots(1,4,figsize=(10,5))
#        axs[0].imshow(np.squeeze(Imgin[:,:,1,0]))
#        # axs[1].imshow(np.squeeze(Signal[:,:,11,10]))
#        # axs[2].imshow(np.squeeze(imgorig[:,:,11,10,1]))
#        # axs[3].imshow(np.squeeze(Sigma[:,:,11]))
#        # plt.show()
#
#        import pdb
#        pdb.set_trace()
        

        if ndims == 4:
            Img = np.reshape(Imgin, (Imgin.shape[0], Imgin.shape[1], Imgin.shape[2]*Imgin.shape[3]))
       
        print(Img.shape)

        OutImg = np.zeros(Img.shape)
        NoiseImg = np.zeros(Img.shape)	

        
        for s in tqdm(range(Img.shape[-1])):
            Slice = np.absolute(Img[:,:,s])
            #Slice = Slice_.view(np.double).reshape((Slice_.shape[0],Slice_.shape[1]))
            # zzj = zz[:,:,:,0] + 1j*zz[:,:,:,1]
            # zzk = f['a'].value.view(np.complex)
            

            Slice, ratio = normalize(np.float32(Slice[:,:]))

            Slice = np.expand_dims(Slice, 0)
            Slice = np.expand_dims(Slice, 1)
            ISource = torch.Tensor(Slice)

            # noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
            INoisy = ISource # + noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            #ISource, INoisy = Variable(ISource), Variable(INoisy)
            with torch.no_grad(): 
                noisemap = model(INoisy)
                Out = INoisy-noisemap
                noisemap = np.squeeze(noisemap)
                Outs = np.squeeze(Out)
            Outs = Outs*255*ratio
            OutImg[:,:,s] = Outs.cpu()
            NoiseImg[:,:,s] = noisemap.cpu()

            psnr = batch_PSNR(Out, ISource, 1.)
            psnr_test += psnr
            #print(" PSNR %f" % (psnr), end='\r')


        if ndims == 4:
            outimg = np.reshape(OutImg, (Imgin.shape[0], Imgin.shape[1], Imgin.shape[2], Imgin.shape[3]))

        #out = np.squeeze(outimg[:,:,:,:,:,0] + 1j*outimg[:,:,:,:,:,1])
        #ins = np.squeeze(Imgin[:,:,:,:,:,0] + 1j*Imgin[:,:,:,:,:,1])

#        import matplotlib
#        matplotlib.use('TkAgg')
#        import matplotlib.pyplot as plt
#        fig, axs = plt.subplots(1,3,figsize=(10,5))
#        axs[0].imshow(np.squeeze(np.absolute(ins[:,:,1,0,49])))
#        axs[1].imshow(np.squeeze(np.absolute(out[:,:,1,0,49])))
#        axs[2].imshow(np.absolute(np.squeeze(DNG[:,:,1,0,49,0]+1j*DNG[:,:,1,0,49,1])))
#        plt.show()

        #gt = np.sqrt(Imgt['real']**2 + Imgt['imag']**2)

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt

        # fig, axs = plt.subplots(1,3,figsize=(10,5))
        # axs[0].imshow(np.abs(np.squeeze(gt[:,:,11,10,0])))
        # axs[1].imshow(np.squeeze(outimg[:,:,11,10,0]))
        # axs[2].imshow(np.squeeze(gt[:,:,11,10,1]))
        # plt.show()

        io.savemat("/cbi05data/data1/Hamster/Ben/coilstuff/imgreconX_RMT_08182020_10__ysgs_MAG_D2dncnn_B55mri_AC.mat",{'outimg':outimg})
        #sio.savemat("/cbi05data/data1/Hamster/Ben/coilstuff/FullySampled_PGSE_TRSE_.mat",{'outimg':imgorig})

        #sio.savemat("/cbi05data/data1/Hamster/Ben/coilstuff/FullySampled_PGSE_TRSE_mppca.mat",{'outimg':Signal})
        #sio.savemat("/cbi05data/data1/Hamster/Ben/coilstuff/FullySampled_PGSE_TRSE_sigma.mat",{'outimg':Sigma})
         # import pdb
        # pdb.set_trace()


        # if ndims == 4:
        #     OutImg = np.reshape(OutImg,(shape[0],shape[1],shape[2],shape[3]))
        #     NoiseImg = np.reshape(NoiseImg,(shape[0],shape[1],shape[2],shape[3]))

        # OutImg = OutImg.astype('uint16')
        # OutImg[OutImg > 65500] = 0    

        

        # # niout = ni.Nifti1Image(OutImg, nii.affine, nii.header)
        # # ni.save(niout, os.path.join(opt.test_data, fname + '_dncnn_blank.nii'))
        # # niout = ni.Nifti1Image(NoiseImg, nii.affine, nii.header)
        # # ni.save(niout, os.path.join(opt.test_data, fname + '_dncnn_noisemap.nii'))

        # niout = ni.Nifti1Image(OutImg, np.eye(4))
        # ni.save(niout, os.path.join(opt.test_data, fname + '_dncnn.nii'))
        # niout = ni.Nifti1Image(NoiseImg, np.eye(4))
        # ni.save(niout, os.path.join(opt.test_data, fname + '_dncnn_noisemap.nii'))

        # psnr_test /= (len(files_source)*Img.shape[-1])
        # print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
