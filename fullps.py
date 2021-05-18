import os
import torch
import torch.fft as fft

def ps(residual):

    residual = residual.type(torch.float64)
    sx = residual.size(-1)
    sy = residual.size(-2)
    A = sx * sy
    epf = fft.fftshift(fft.fft2(residual, s=(sy*2-1, sx*2-1)))
    gamma = torch.abs(torch.conj(epf) * epf) / A
    
    gx = gamma.size(-1)
    gy = gamma.size(-2)

    xv, yv = torch.meshgrid(torch.arange(-gy//2, gy//2), torch.arange(-gx//2, gx//2))
    xv = xv.double()
    R = torch.floor(torch.sqrt(xv**2 + yv**2))
    rad, ct = torch.unique(R, return_counts=True)
    nbins = len(rad)

    # minrad = torch.min(rad)
    # maxrad = torch.max(rad)
    # delta = (maxrad - minrad) / nbins
    # interval = torch.linspace(minrad, maxrad, nbins)-delta/2
   
    #Rq = torch.bucketize(R, interval)
    #newrads = torch.unique(Rq)

    batchsize = gamma.size(0)
    ghist = torch.zeros(batchsize,nbins)
    count = torch.zeros(nbins)

    for i in range(0, nbins):
        indsx, indsy = torch.where(R == rad[i])
        count[i] += indsx.size(0)
        for b in range(batchsize):
            ghist[b,i] += torch.sum(gamma[b, 0, indsx, indsy]) 
            
    ravg = ghist/count
    return ravg, rad

