
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
from dataset import prepare_data, Dataset
from utils import *
#from powerspectrumloss import PowerspectrumLoss
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
parser.add_argument("--patchSize", type=float, default=30, help='patchsize')

opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(in_channels=1, out_channels=1, num_of_layers=opt.num_of_layers)
    #net.apply(weights_init_kaiming)
    #criterion = nn.MSELoss(size_average=True)
    #criterion = nn.MSELoss(reduction='mean')
    #criterion = PowerspectrumLoss(opt.batchSize, opt.patchSize)
    criterion = testmse(opt.batchSize, opt.patchSize)
    # criterion = resspect()
    # Move to GPU
    print(torch.__version__) 
    print(torch.__file__)
    device_ids = [0, 1]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    #model = net.cpu()
    criterion.cuda()

    # loss = criterion(n = 0.5*np.random.randn(img.shape[0],img.shape[1])
    # sigma = .5*np.ones((50,50))

    # img = torch.Tensor(img) 
    # img = img.unsqueeze(0)
    # imgn = torch.Tensor(imgn)
    # imgn = imgn.unsqueeze(0)
    # sigma = torch.Tensor(sigma)
    # simga = sigma.unsqueeze(0)

    # res = img-imgn)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    noiseL = 20
    RAVG = []
    # for param_group in optimizer.param_groups:
    #         param_group["lr"] = opt.lr
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    #milestones = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    #milestones = [5, 10, 15]
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
            model.train()
            #model.zero_grad()
            optimizer.zero_grad()
            img_train = data
                
            noise = torch.zeros(img_train.size())    
            psize = img_train.size(2)

            sizeN = noise[0,:,:,:].size()
            stdN = torch.Tensor(np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0]))

            # single noise level, uniform noise
            #noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=noiseL/255.)
            # blind
            for n in range(noise.size()[0]):
                noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)

            
            # for n in range(noise.size()[0]):
            #     sig = np.random.uniform(psize//2, psize*2)
            #     coords = np.linspace(-psize//2, psize//2, psize)
            #     x0 = np.random.uniform(0, psize)
            #     y0 = np.random.uniform(0, psize)
            #     x, y = np.meshgrid(coords, coords)
            #     g = 1/(2*np.pi*sig**2) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sig**2))
            #     g = torch.Tensor(g/np.max(g))                
            #     noi = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            #     # testidf = g*noi
            #     # testidf = testidf.numpy()[0,:,:]

            #     # import matplotlib.pyplot as plt
            #     # plt.imshow(testidf)
            #     # plt.show()
            #     # import pdb
            #     # pdb.set_trace()

            #     noise[n,:,:,:] = g*noi

            imgn_train = img_train + noise
            #print(imgn_train.size())
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)

            residual = out_train[:,0,:,:]
            #noisevar = out_train[:,1,:,:]
            #noisevar = noiseL/255.
            noisevar = stdN/255.

            #normresidual = residual / noisevar
 
            #loss = criterion(residual, noisevar, imgn_train, img_train) # / (imgn_train.size()[0]*2)
            loss = criterion(out_train, noise, noisevar.cuda())

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # results
            model.eval()
           
            out_train2 = torch.clamp(imgn_train - residual.unsqueeze(1), 0., 1.)
            psnr_train = batch_PSNR(out_train2, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('Gamma', criterion.meanps, step)
                #writer.add_scalar('sigma MP', criterion.meanmp, step)
                writer.add_scalars('ind losses', {'PS': criterion.ps,
                'MSE': criterion.mse}, step)
                #writer.add_scalar('mean LP', criterion.meanlap, step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            #scheduler.step()
        ## the end of each epoch
        
        model.eval()
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                out_val = model(imgn_val)
                residual_val = out_val[:,0,:,:]
                #noisevar_val = out_val[:,1,:,:]
                out_val2 = torch.clamp(imgn_val-residual_val.unsqueeze(1), 0., 1.)
                psnr_val += batch_PSNR(out_val2, img_val, 1.)
            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
            writer.add_scalar('PSNR on validation data', psnr_val, epoch)
            # log the images
            out_train = model(imgn_train)
            out_train1 = out_train[:,0,:,:].unsqueeze(1)
            #out_train2 = out_train[:,1,:,:].unsqueeze(1)

            res1 = out_train / (noisevar.view(-1,1,1,1).cuda())
            ravg, interval = ps(res1.cpu())
            ravg[torch.isnan(ravg)] = 0
            ravg = torch.mean(ravg, axis=0)
            print(ravg.size())
            print(ravg)
            fig = plt.figure()
            plot = plt.plot(interval, ravg)
            plt.ylim([0, 2])
            
            writer.add_figure('powerspectrum', fig, epoch, close=True)

            # if torch.isnan(ravg).any():
            #     ravg = torch.zeros(1,50)

            RAVG.append(ravg.numpy())
            
            recon = torch.clamp(imgn_train-out_train, 0., 1.)
            #print(img_train.data.size())
            #print(imgn_train.data.size())
            #print(recon.data.size())

            comp = torch.empty((opt.batchSize*4,1,opt.patchSize,opt.patchSize), dtype=recon.dtype)
            print(comp.size())
            print(img_train.size())
            comp[0::4,:,:,:] = img_train
            comp[1::4,:,:,:] = imgn_train
            comp[2::4,:,:,:] = recon
            comp[3::4,:,:,:] = out_train1
            #comp[4::5,:,:,:] = out_train2

            #Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
            #Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
            #Irecon = utils.make_grid(recon.data, nrow=8, normalize=True, scale_each=True)
            #Isigma = utils.make_grid(out_train2.data, nrow=8, normalize=True, scale_each=True)
            Comp = utils.make_grid(comp.data, nrow=4, normalize=True, scale_each=True)
            #writer.add_image('clean image', Img, epoch)
            #writer.add_image('noisy image', Imgn, epoch)
            #writer.add_image('reconstructed image', Irecon, epoch)
            #writer.add_image('sigma', Isigma, epoch)
            writer.add_image('compare', Comp, epoch)
            #writer.add_histogram('full power spectrum', ravg, epoch)
            # save model
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
    sio.savemat(os.path.join(opt.outf,'ravg.mat'), {'ravg':RAVG})

if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path='data', patch_size=opt.patchSize, stride=10, aug_times=2)
    main()
