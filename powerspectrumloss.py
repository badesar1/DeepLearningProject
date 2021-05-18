import os
import torch
import torch.fft as fft
from GaussHist import GaussianHistogram
import math

class PowerspectrumLoss(torch.nn.Module):
    def __init__(self):
        super(PowerspectrumLoss, self).__init__()
    
    def forward(self, residual, noisevar): 
        # residual = residual.unsqueeze(0)
        # sigma = sigma.unsqueeze(0)
        # residual = residual.unsqueeze(0)
        # sigma = sigma.unsqueeze(0)
        #residual = torch.clamp(residual.type(torch.float64),-100,100)

        x = residual.size(-1)
        batchs = residual.size(0)
        A = x ** 2

        #print(torch.sum(torch.isnan(ep)))
        #print(ep.size())
        epf = fft.fft2(residual / noisevar, dim=[-2,-1])
        #print(torch.sum(torch.isnan(epf)))
        #epf[torch.isnan(epf)] = 0

        # epf = epf.flatten()
        # epf = epf

        gamma = torch.abs(torch.conj(epf) * epf) / A
        g = gamma.flatten()

        # print(torch.mean(gamma))
        # print(torch.max(gamma))

        #print(torch.linalg.norm(torch.sqrt(gamma.flatten())-1))
        # print(torch.sum(torch.isnan(gamma)))
        #print(gamma.size()

        eps = torch.finfo(torch.float64).eps

        bins = 20
        normres = residual / noisevar
        batchsize = residual.size(0)

        nrf = normres.flatten()
        try:
            rmin = torch.min(nrf[torch.isfinite(nrf)])
            rmax = torch.max(nrf[torch.isfinite(nrf)])
        except:
            print(nrf)
            
        if rmin == rmax:
            bounds = (rmin-1, rmax+1) 
        else:
            bounds = (rmin, rmax)
        #print(bounds)
        #hist = torch.histc(normres[i,:,:], bins=bins)
        gsig = (bounds[1] - bounds[0])/100
       


        # cdf of residual, 
        GH = GaussianHistogram(bins=bins, min=bounds[0], max=bounds[1], sigma=gsig)
        ghist, centers, width = GH(nrf)
        ghist = ghist.div(ghist.sum())
     
        #hist = hist.div(hist.sum())
        #bin_edges = torch.linspace(rmin.item(), rmax.item(), steps=bins).cuda()
        #binwidth = torch.mean(bin_edges[1:]-bin_edges[:-1])
        nsigma = (centers)**2
        logprob = torch.log(ghist/width + eps)
        #print(logprob)
        #print(nsigma)
        
        b = (logprob[1:]-logprob[:-1])/(nsigma[1:]-nsigma[:-1])
        #print(b)


        #print(torch.mean(b))
        bf = torch.isfinite(b)
        a = torch.mean(b[bf])

        # for i in range(0,batchsize):
        #     rmin = torch.min(normres[i,:,:])
        #     rmax = torch.max(normres[i,:,:])
        #     bounds = (rmin, rmax)
        #     #hist = torch.histc(normres[i,:,:], bins=bins)
        #     gsig = (bounds[1] - bounds[0])/100
        #     GH = GaussianHistogram(bins=bins, min=bounds[0], max=bounds[1], sigma=gsig)
        #     ghist, centers, width = GH(normres[i,:,:].flatten())
        #     ghist = ghist.div(ghist.sum())
        #     #hist = hist.div(hist.sum())
        #     #bin_edges = torch.linspace(rmin.item(), rmax.item(), steps=bins).cuda()
        #     #binwidth = torch.mean(bin_edges[1:]-bin_edges[:-1])
        #     nsigma = (centers)**2
        #     logprob = torch.log(ghist/width)
        #     b = (logprob[1:]-logprob[:-1])/(nsigma[1:]-nsigma[:-1])
        #     a[i] = torch.mean(b)
        # print(torch.mean(a))
        # print(torch.mean(g))

        loss4 = 1 * torch.linalg.norm(a + 0.5)
        loss1 = 1 / (math.sqrt(A)*math.sqrt(batchs)) * torch.linalg.norm(g-1)
        loss2 = 0.05 * torch.linalg.norm(residual.flatten()) #0.3 too much
        
        loss3 = 1 * torch.linalg.norm(torch.std(residual.flatten()) - noisevar)
        loss5 = 1 * torch.linalg.norm(torch.mean(residual.flatten()))


        X = residual.reshape(A,batchsize)
        u,vals,v = torch.svd(X)
        vals = (vals**2)/A
        csum = torch.cumsum(torch.flip(vals, dims=[0]), 0)

        datarange = torch.Tensor(range(0, batchsize)).cuda()
        idatarange = torch.flip(datarange+1, dims=[0])

        sigmasq_1 = torch.flip(csum, dims=[0]) / idatarange
        gammp = (batchsize - datarange) / (A - datarange)
        rangeMP = 4*torch.sqrt(gammp[:])
        rangeData = vals[0:batchsize]-vals[batchsize-1]
        sigmasq_2 = rangeData/rangeMP

        t = torch.where(sigmasq_2 < sigmasq_1)
        if t[0].any():
            t = t[0][0]
        else:
            t = batchsize - 1
        sigma_mp = torch.sqrt(sigmasq_1[t])
        print(sigma_mp)

        loss6 = 1 * torch.linalg.norm(sigma_mp - noisevar)
        #print(loss6)


        # print(loss4)
        # print(loss1)
        # print(loss3)
        # print(loss5)
        #loss = torch.linalg.norm(torch.sqrt(gamma.flatten()-1)) + torch.linalg.norm(residual.flatten())
        #loss = torch.mean((gamma-1)**2,dim=[-2,-1])
        # print(loss)
        # print(loss.size())
        # print(torch.mean(loss))
        #print(torch.abs(torch.mean(gamma-1)))
        #loss = torch.abs(torch.mean((gamma-1)))
        #print(torch.mean(loss))
        
        return loss6 + loss5 #loss3 + loss4 + loss5 + loss1 #+ loss3 + loss5

