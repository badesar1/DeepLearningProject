
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models_psloss import DnCNN
from DWIdataset import prepare_data, Dataset
from utils import *

from mploss import PowerspectrumLoss
from testmse import testmse
from fullps import ps
import scipy.io as sio
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=15, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--patch_size", type=float, default=[50,50,5], help='patchsize')
parser.add_argument("--fname", type=str, default='mri', help='patchsize')

opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, fname=opt.fname)
    dataset_val = Dataset(train=False, fname=opt.fname)
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(in_channels=5, out_channels=5, num_of_layers=opt.num_of_layers)
    #net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    #criterion = nn.MSELoss(reduction='mean')
    #criterion = PowerspectrumLoss(opt.batchSize, opt.patchSize)
    # criterion = testmse(opt.batchSize, opt.patchSize)
    # Move to GPU
    print(torch.__version__) 
    print(torch.__file__)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    #model = net.cpu()
    criterion.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,50] # ingnored when opt.mode=='S'
    noiseL = 50
    RAVG = []
 
    milestones = [10]
    current_lr = opt.lr
    
    for epoch in range(opt.epochs):
        if any(epoch == m for m in milestones):
            current_lr = current_lr / 10.

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            def handle_batch():
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                img_train = data.permute(0,1,4,2,3).squeeze(1)
                #img_train = data.view(-1,opt.patch_size[2],opt.patch_size[0],opt.patch_size[1])
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)        
                imgn_train = img_train + noise

                img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
                noise = Variable(noise.cuda())
                out_train = model(imgn_train)

                loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # results
                model.eval()
           
                img_train_dn = torch.clamp(imgn_train - out_train, 0., 1.)
                psnr_train = batch_PSNR(img_train_dn, img_train, 1.)
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
                # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
                if step % 10 == 0:
                    # Log the scalar values
                    writer.add_scalar('loss', loss.item(), step)
                    #writer.add_scalar('Gamma', criterion.meanps, step)
                    #writer.add_scalar('sigma MP', criterion.meanmp, step)
                    #writer.add_scalars('ind losses', {'PS': criterion.loss1,
                    #'sigma_mp - noisevar': criterion.loss2,
                    #'sigma1 - sigma2': criterion.loss3,
                    #'MSE': criterion.mseloss}, step)
                    #writer.add_scalar('mean LP', criterion.meanlap, step)
                    writer.add_scalar('PSNR on training data', psnr_train, step)
            handle_batch()
            step += 1
        ## the end of each epoch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        model.eval()
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = dataset_val[k]
                print(img_val.size())
                img_val = img_val.permute(0,3,1,2)
                #img_val = img_val.view(1,img_val.size(3),img_val.size(1),img_val.size(2))
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.noiseL/255.)        
                imgn_val = img_val + noise

                img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                out_val = model(imgn_val)
      
                denoised = torch.clamp(imgn_val - out_val, 0., 1.)
                psnr_val += batch_PSNR(denoised, img_val, 1.)
            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
            writer.add_scalar('PSNR on validation data', psnr_val, epoch)
            
            # log the images
            img_train = iter(loader_train).next()
            img_train = img_train.permute(0,1,4,2,3).squeeze(1)
            #img_train = img_train.view(-1,opt.patch_size[2],opt.patch_size[0],opt.patch_size[1])
            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)        
            imgn_train = img_train + noise

            out_train = model(imgn_train.cuda())
            noisevar = torch.std(out_train,axis=[-3,-2,-1])
            print(noisevar.size())
            #out_train2 = out_train[:,1,:,:].unsqueeze(1)

            res1 = out_train / (noisevar.view(-1,1,1,1).cuda())
            ravg, interval = ps(res1[0,:,:,:].unsqueeze(0).cpu())
            ravg[torch.isnan(ravg)] = 0
            ravg = torch.mean(ravg, axis=0)
            print(ravg.size())
            print(ravg)
            fig = plt.figure()
            plot = plt.plot(interval, ravg)
            plt.ylim([0, 2])
            
            writer.add_figure('powerspectrum', fig, epoch, close=True)
            RAVG.append(ravg.numpy())
            
            denoised = torch.clamp(imgn_train.cuda()-out_train, 0., 1.)
            for i in range(5):
                comp = torch.empty((int(img_train.size(0)*4),1,int(opt.patch_size[0]),int(opt.patch_size[1])), dtype=denoised.dtype)
                comp[0::4,:,:,:] = img_train[:,i,:,:].unsqueeze(1)
                comp[1::4,:,:,:] = imgn_train[:,i,:,:].unsqueeze(1)
                comp[2::4,:,:,:] = denoised[:,i,:,:].unsqueeze(1)
                comp[3::4,:,:,:] = out_train[:,i,:,:].unsqueeze(1)
                Comp = utils.make_grid(comp.data, nrow=4, normalize=True, scale_each=True)
                writer.add_image('compare'+str(i), Comp, epoch)
            

            writer.add_histogram('full power spectrum', ravg, epoch)
            # save model
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
    sio.savemat(os.path.join(opt.outf,'ravg.mat'), {'ravg':RAVG})

if __name__ == "__main__": 
    if opt.preprocess:
        prepare_data(data_path='/gpfs/data/fieremanslab/Ben/DnCNN/data', patch_size=opt.patch_size, stride=10, aug_times=3, fname=opt.fname)
    main()
