
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
from moments import testRicMoments
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
    net = DnCNN(in_channels=1, out_channels=1, num_of_layers=opt.num_of_layers)
    #net.apply(weights_init_kaiming)
    #criterion = nn.MSELoss(size_average=False)
    criterion = testRicMoments()
    #criterion = nn.MSELoss(reduction='mean')
    #criterion = PowerspectrumLoss(opt.batchSize, opt.patchSize)
    # criterion = testmse(opt.batchSize, opt.patchSize)
    # criterion = resspect()
    # Move to GPU
    print(torch.__version__) 
    print(torch.__file__)
    device_ids = [0]

    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    #model.load_state_dict(torch.load('/gpfs/data/fieremanslab/Ben/DnCNN/logs/FGtraining_P2d50_MSE_b64_l20_testAGnoise/net.pth'))
    #model = net.cpu()
    criterion.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[1,10] # ingnored when opt.mode=='S'
    noiseL = 5
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
            def handle_batch():
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                img_train_clean = data[:,0,...]
                #img_train_noisy = data[:,1,...]
                
                noise = torch.zeros(img_train_clean.size()) 
                img_train_Rnoise = torch.zeros(img_train_clean.size()) 
                img_train_Gnoise = torch.zeros(img_train_clean.size())
                stdN = torch.Tensor(np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0]))
                sizeN = noise[0,:,:,:].size()
                for n in range(noise.size()[0]):
                    rnoise = torch.FloatTensor(sizeN).normal_(mean=0, std=1./stdN[n])
                    inoise = torch.FloatTensor(sizeN).normal_(mean=0, std=1./stdN[n])
                    #gnoise = torch.FloatTensor(sizeN).normal_(mean=0, std=1./stdN[n])
                    img_train_Rnoise[n,:,:,:] = torch.sqrt((img_train_clean[n,:,:,:] + rnoise)**2 + inoise**2)
                    #img_train_Gnoise[n,:,:,:] = img_train_clean[n,:,:,:] + rnoise
                    #noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=1./stdN[n])

                noisevar = 1./stdN
                out_train = model(img_train_Rnoise.cuda())
                loss = criterion(out_train, img_train_clean.cuda(), noisevar.cuda()) #/ (img_train_clean.size()[0]*2)

                loss.backward()
                optimizer.step()
                # results
                model.eval()

                residual = out_train - img_train_clean.cuda()
                mean_train = torch.mean(residual.flatten())
                std_train = residual.view(img_train_clean.size(0),-1).std(1)
                std_diff = torch.mean(1./stdN - std_train.cpu())

                print("[epoch %d][%d/%d] loss: %.4f mean_train: %.4f std_diff %.4f" %
                    (epoch+1, i+1, len(loader_train), loss.item(), mean_train, std_diff))
                # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
                if step % 10 == 0:
                    # Log the scalar values
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('std diff', std_diff, step)
                    writer.add_scalar('mean', mean_train, step)
            handle_batch()
            step += 1
        ## the end of each epoch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        model.eval()
        # validate
        std_diff_val = 0
        mean_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = dataset_val[k]
                
                # noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                # imgn_val = img_val + noise
                img_val_clean = img_val[0,...].unsqueeze(0)
                #img_val_noisy = img_val[1,...].unsqueeze(0)

                #noise = torch.FloatTensor(img_val_clean.size()).normal_(mean=0, std=noiseL/255.)
                #img_val_noisy = img_val_clean + noise

                rnoise = torch.FloatTensor(img_val_clean.size()).normal_(mean=0, std=1./noiseL)
                inoise = torch.FloatTensor(img_val_clean.size()).normal_(mean=0, std=1./noiseL)
                #gnoise = torch.FloatTensor(img_val_clean.size()).normal_(mean=0, std=1./noiseL)
                img_val_Rnoise = torch.sqrt((img_val_clean + rnoise)**2 + inoise**2)
                #img_val_Gnoise = img_val_clean + gnoise

                out_val = model(img_val_Rnoise.cuda())

                residual = out_val - img_val_clean.cuda()
                mean_val += torch.mean(residual.flatten())
                std_val = residual.view(img_val_clean.size(0),-1).std(1)
                std_diff_val += torch.mean(1./noiseL - std_val)
 
            std_diff_val /= len(dataset_val)
            mean_val /= len(dataset_val)
            print("\n[epoch %d] std diff val: %.4f mean_val: %.4f" % (epoch+1, std_diff_val, mean_val))
            writer.add_scalar('std diff on validation data', std_diff_val, epoch)
            writer.add_scalar('mean on validation data', mean_val, epoch)
            
            # log the images
            img_train = iter(loader_train).next()
            #img_train_noisy = img_train[:,1,...]
            img_train_clean = img_train[:,0,...]

            noise = torch.zeros(img_train_clean.size())  
            stdN = torch.Tensor(np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0]))
            sizeN = noise[0,:,:,:].size()
            img_train_Rnoise = torch.zeros(img_train_clean.size()) 
            img_train_Gnoise = torch.zeros(img_train_clean.size())
            for n in range(noise.size()[0]):
                rnoise = torch.FloatTensor(sizeN).normal_(mean=0, std=1./stdN[n])
                inoise = torch.FloatTensor(sizeN).normal_(mean=0, std=1./stdN[n])
                #gnoise = torch.FloatTensor(sizeN).normal_(mean=0, std=1./stdN[n])
                img_train_Rnoise[n,:,:,:] = torch.sqrt((img_train_clean[n,:,:,:] + rnoise)**2 + inoise**2)
                #img_train_Gnoise[n,:,:,:] = img_train_clean[n,:,:,:] + gnoise

            out_train = model(img_train_Rnoise.cuda())
            residual = out_train - img_train_clean.cuda()

            fig = plt.figure()
            plot = plt.hist(residual[0,...].flatten().cpu())
            writer.add_figure('residual hist', fig, epoch, close=True)

            comp = torch.empty((int(img_train_clean.size(0)*4),1,int(opt.patchSize),int(opt.patchSize)), dtype=img_train_clean.dtype)
            print(comp.size())
            print(img_train.size())
            if img_train_clean.ndim == 4:
                comp[0::4,:,:,:] = img_train_clean
                comp[1::4,:,:,:] = img_train_Rnoise
                comp[2::4,:,:,:] = out_train
                comp[3::4,:,:,:] = residual
        
            Comp = utils.make_grid(comp.data, nrow=4, normalize=True, scale_each=True)
            writer.add_image('compare', Comp, epoch)
            # save model
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path='/gpfs/data/fieremanslab/Ben/data', patch_size=opt.patchSize, stride=10, aug_times=1, dim=2, fname=opt.fname)
    main()
