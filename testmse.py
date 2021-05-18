import os
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PS(torch.nn.Module):
    def __init__(self):
        super(PS, self).__init__()
        # self.iscuda = True
        # self.N = patchsize
        # self.batchsize = batchsize
        # self.A = patchsize ** 2
        # self.eps = torch.finfo(torch.float64).eps
        
        # datarange = torch.arange(0, patchsize)
        # if self.iscuda:
        #     datarange = datarange.cuda()
        # self.idatarange = torch.flip(datarange+1, dims=[0])
        # gammp = (patchsize - datarange) / (patchsize)
        # self.rangeMP = 4 * torch.sqrt(gammp)

        # nx = patchsize
        # ny = patchsize
        # kx = torch.arange(-nx/2, nx/2)
        # ky = torch.arange(-ny/2, ny/2)
        # self.kxgrid, self.kygrid = torch.meshgrid(kx, ky)
        # if self.iscuda:
        #     self.kxgrid = self.kxgrid.cuda()
        #     self.kygrid = self.kygrid.cuda()

    def powerspectrum(self, residual):
        sy = residual.size(-1)
        sx = residual.size(-2)
        imarea = sx * sy
        epf = fft.fft2(residual, s=(sx*2-1,sy*2-1))
        gamma = torch.abs(epf * torch.conj(epf)) / imarea
        gamma = fft.fftshift(gamma)
        gamma = torch.clip(gamma,min=-1,max=3)

        # autocorrelation = F.conv2d(residual, residual, padding=sx//2) / imarea
        # fftfromautocorr = torch.abs(fft.fft2(autocorrelation))
        
        #gamma = gamma[:,:,sx//4:3*sx//4,sy//4:3*sy//4]
        
        #mask = torch.ones_like(gamma)
        #mask[:,:,sx//4:3*sx//4,sy//4:3*sy//4] = 0
        #mask[:,:,sx//2,sy//2] = 0
        #gamma = gamma[mask==1]

        return gamma.flatten()

    def mp(self, residual):
        if residual.ndim == 4:
            residual = residual.squeeze()

        u,vals,v = torch.svd(residual)

        vals = (vals**2) / self.N
        csum = torch.cumsum(torch.flip(vals, dims=[1]), 1)
        sigmasq_1 = torch.flip(csum, dims=[1]) / self.idatarange

        rangeData = vals[:,:self.N] - vals[:,self.N-1].unsqueeze(1)
        sigmasq_2 = rangeData / self.rangeMP

        zeros = torch.zeros((sigmasq_1.size()))
        if self.iscuda:
            zeros = zeros.cuda()
        t1 = torch.where(sigmasq_2 < sigmasq_1, torch.sqrt(sigmasq_1), zeros)
        sigma_mp1 = torch.max(t1, axis=-1)[0]
        t2 = torch.where(sigmasq_2 < sigmasq_1, torch.sqrt(sigmasq_2), zeros)
        sigma_mp2 = torch.max(t2, axis=-1)[0]

        print(torch.mean(sigma_mp1))
        print(torch.mean(sigma_mp2))

        # min_idx = torch.argmax(torch.abs(sigmasq_1 - sigmasq_2), dim=-1)
        # sigma_mp1 = torch.sqrt(sigmasq_1[...,min_idx])
        # sigma_mp2 = torch.sqrt(sigmasq_2[...,min_idx])

        self.meanmp = torch.mean(sigma_mp1)
        return sigma_mp1, sigma_mp2

    def laplacian(self, residual, imgn):
        imgdn = imgn - residual
        imk = fft.fftshift(fft.fft2(imgdn, dim=[-2,-1]))
        Lapk = -(self.kxgrid**2 + self.kygrid**2) * imk
        Lapx = torch.abs(fft.ifft2(fft.ifftshift(Lapk), dim=[-2,-1]))
        self.meanlap = torch.mean(Lapx)
        return Lapx.flatten()

    def forward(self, out, noise, noisevar): 
        batchsize = out.size(0)
        patchsize = out.size(-1)

        normres = out / noisevar.view(-1,1,1,1)
        ps = self.powerspectrum(normres)

        meanps = torch.mean(ps)
        self.meanps = meanps

        # dps = torch.mean(torch.abs(ps[1:] - ps[:-1]))
        # self.dps = dps

        loss1 = torch.sum(torch.abs(ps-1)) / (batchsize * patchsize**2)
        #loss1 = torch.abs(ps.mean()-1) / (2 * batchsize)
        self.ps = loss1

        loss7 = torch.sum((out - noise)**2) / (2 * batchsize)
        self.mse = loss7

        #print(loss1)

        # mp1, mp2 = self.mp(out)
        # loss2 = torch.linalg.norm(mp1 - noisevar) / (0.1)
        # loss3 = torch.linalg.norm(mp1 - mp2) / (2 * self.batchsize)
        # self.loss2 = loss2
        # self.loss3 = loss3
        # print(loss2)

        # lap = self.laplacian(residual, imgn)
        # loss3 = 1 / (50 * self.N * np.sqrt(self.batchsize)) * torch.linalg.norm(lap)

        
        #print(loss7)

        # loss4 = 0.05 * torch.linalg.norm(residual.flatten()) #0.3 too much
        # loss5 = 1 * torch.linalg.norm(torch.std(residual.flatten()) - noisevar)
        # loss6 = 1 * torch.linalg.norm(torch.mean(residual.flatten()))
   
        return loss1 + loss7 # + loss3 # loss1 + loss2 + loss6 + loss3 + loss7

