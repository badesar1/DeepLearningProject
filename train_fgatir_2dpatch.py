
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
from MRIDataset import prepare_data, Dataset
from utils import *
#from powerspectrumloss import PowerspectrumLoss
from mploss import PowerspectrumLoss
from testmse import PS
from fullps import ps
import scipy.io as sio
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=15, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--patchSize", type=float, default=30, help='patchsize')
parser.add_argument("--fname", type=str, default='mri', help='patchsize')

opt = parser.parse_args()

def grad_norm(params):
    """ computes norm of mini-batch gradient
    """
    total_norm = 0
    for p in params:
        param_norm = torch.tensor(0)
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
        total_norm = total_norm + param_norm.item()**2
    return total_norm**(.5)

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, fname=opt.fname)
    dataset_val = Dataset(train=False, fname=opt.fname)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(in_channels=2, out_channels=1, num_of_layers=opt.num_of_layers)
    #net.apply(weights_init_kaiming)
    #criterion = nn.MSELoss(size_average=False)
    #criterion = nn.MSELoss(reduction='mean')
    #criterion = PowerspectrumLoss(opt.batchSize, opt.patchSize)
    criterion = PS()
    # criterion = resspect()
    # Move to GPU
    print(torch.__version__) 
    print(torch.__file__)
    device_ids = [0]

    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    #model.load_state_dict(torch.load('/gpfs/data/fieremanslab/Ben/DnCNN/logs/FGtraining_P2d70_MSE_b64_l20_testAGnoise_2ch_nB0-50/net.pth'))
    #model = net.cpu()
    criterion.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,50] # ingnored when opt.mode=='S'
    noiseL = 10
    RAVG = []
    # for param_group in optimizer.param_groups:
    #         param_group["lr"] = opt.lr
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    #milestones = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    #milestones = [5, 10, 15]
    milestones = [10,20,30]
    current_lr = opt.lr
    psnr_current = [34]
    torch.save(model.state_dict(), os.path.join(opt.outf, 'state_dict_0.pth'))
    
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
                img_train_clean = data[:,0,...]
                img_train_noisy = data[:,1,...]
                
                noise = torch.zeros(img_train_clean.size())  
                stdN = torch.Tensor(np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0]))
                sizeN = noise[0,:,:,:].size()
                for n in range(noise.size()[0]):
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)

                #noise = img_train_clean - img_train_noisy
                #rnoise = torch.FloatTensor(img_train_clean.size()).normal_(mean=0, std=noiseL/255.)
                #inoise = torch.FloatTensor(img_train_clean.size()).normal_(mean=0, std=noiseL/255.)
                img_train_noisy = img_train_clean + noise
                #img_train_noisy = torch.sqrt((img_train_clean + rnoise)**2 + inoise**2)
                #noise = img_train_clean - img_train_noisy
                # noise = torch.sqrt(rnoise**2 + inoise**2)
                #noise = rnoise

                ch2 = stdN.view(-1,1,1,1)*torch.ones_like(img_train_clean)/255.
                minput = torch.cat((img_train_noisy,ch2),dim=1)
                out_train = model(minput.cuda())

                noisevar = stdN/255.
                loss = criterion(out_train, noise.cuda(), noisevar.cuda()) #/ (imgn_train.size()[0]*2)

                loss.backward()

                # gradnorm = grad_norm(model.parameters())
                # if gradnorm > .01:
                #     fig = plt.figure()
                #     plot = plt.imshow(minput[0,0,:,:].cpu())
                #     writer.add_figure('images_with_gradnorm_gt.001', fig, epoch, close=True)
                #     print('batch mean = ' + str(torch.mean(minput[0,0,:,:].flatten())))
                #     print('batch var = ' + str(torch.std(minput[0,0,:,:].flatten())**2))

                #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                optimizer.step()
                # results
                #model.eval()
           
                img_train_dn = torch.clamp(img_train_noisy.cuda() - out_train, 0., 1.)
                psnr_train = batch_PSNR(img_train_dn, img_train_clean, 1.)
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
                    #'sigma_mp - noisevar': criterion.loss2,
                    #'sigma1 - sigma2': criterion.loss3,
                    
                    #writer.add_scalar('mean LP', criterion.meanlap, step)
                    writer.add_scalar('PSNR on training data', psnr_train, step)
                return psnr_train

            # backtracking
            #torch.save(model.state_dict(), os.path.join(opt.outf, 'state_dict_'+str(epoch+1)+'.pth'))
            psnr_batch = handle_batch()
            psnr_current.append(psnr_batch)
            #if psnr_current[-1] < psnr_current[-2]:
            #    model.load_state_dict(torch.load(os.path.join(opt.outf, 'state_dict_'+str(epoch)+'.pth')))
            #    current_lr = current_lr * .9

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
                
                # noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                # imgn_val = img_val + noise
                img_val_clean = img_val[0,...].unsqueeze(0)
                img_val_noisy = img_val[1,...].unsqueeze(0)

                #noise = torch.FloatTensor(img_val_clean.size()).normal_(mean=0, std=noiseL/255.)
                #img_val_noisy = img_val_clean + noise
                rnoise = torch.FloatTensor(img_val_clean.size()).normal_(mean=0, std=noiseL/255.)
                inoise = torch.FloatTensor(img_val_clean.size()).normal_(mean=0, std=noiseL/255.)
                #img_val_noisy = torch.sqrt((img_val_clean + rnoise)**2 + inoise**2)
                img_val_noisy = img_val_clean + rnoise

                #img_val, imgn_val = Variable(img_val_clean.cuda()), Variable(img_val_noisy.cuda())
                ch2 = torch.Tensor([noiseL]).view(1,1,1,1)*torch.ones_like(img_val_noisy)/255.
                minput = torch.cat((img_val_noisy,ch2),dim=1)

                out_val = model(minput.cuda())
                residual_val = out_val[:,0,:,:]
                #noisevar_val = out_val[:,1,:,:]
                denoised = torch.clamp(img_val_noisy.cuda()-residual_val.unsqueeze(1), 0., 1.)
                psnr_val += batch_PSNR(denoised, img_val_clean, 1.)
            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
            writer.add_scalar('PSNR on validation data', psnr_val, epoch)
            
            # log the images
            img_train = iter(loader_train).next()
            img_train_noisy = img_train[:,1,...]
            img_train_clean = img_train[:,0,...]

            noise = torch.zeros(img_train_clean.size())  
            stdN = torch.Tensor(np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0]))
            sizeN = noise[0,:,:,:].size()
            for n in range(noise.size()[0]):
                noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            
            #rnoise = torch.FloatTensor(img_train_clean.size()).normal_(mean=0, std=noiseL/255.)
            #inoise = torch.FloatTensor(img_train_clean.size()).normal_(mean=0, std=noiseL/255.)
            #img_train_noisy = torch.sqrt((img_train_clean + rnoise)**2 + inoise**2)
            img_train_noisy = img_train_clean + noise
            ch2 = stdN.view(-1,1,1,1)*torch.ones_like(img_train_clean)/255.
            minput = torch.cat((img_train_noisy,ch2),dim=1)

            out_train = model(minput.cuda())
            residual = out_train[:,0,:,:].unsqueeze(1)
            noisevar = torch.std(residual,axis=[-2,-1])
            print(noisevar.size())
            #out_train2 = out_train[:,1,:,:].unsqueeze(1)

            res1 = out_train / ((stdN/255.).view(-1,1,1,1).cuda())
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
            
            recon = torch.clamp(img_train_noisy.cuda()-out_train, 0., 1.)
            #print(img_train.data.size())
            #print(imgn_train.data.size())
            #print(recon.data.size())

            comp = torch.empty((int(img_train_clean.size(0)*4),1,int(opt.patchSize),int(opt.patchSize)), dtype=recon.dtype)
            print(comp.size())
            print(img_train.size())
            if img_train_clean.ndim == 4:
                comp[0::4,:,:,:] = img_train_clean
                comp[1::4,:,:,:] = img_train_noisy
                comp[2::4,:,:,:] = recon
                comp[3::4,:,:,:] = residual
            if img_train_clean.ndim == 5:
                comp[0::4,:,:,:,opt.patchSize//2] = img_train_clean
                comp[1::4,:,:,:,opt.patchSize//2] = img_train_noisy
                comp[2::4,:,:,:,opt.patchSize//2] = recon
                comp[3::4,:,:,:,opt.patchSize//2] = residual
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
        prepare_data(data_path='/gpfs/data/fieremanslab/Ben/data', patch_size=opt.patchSize, stride=10, aug_times=1, dim=2, fname=opt.fname)
    main()
