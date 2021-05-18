import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import nibabel as ni
from scipy.ndimage import zoom
import scipy.io as sio

def data_augmentation(image, mode):
    if image.ndim == 3:
        out = np.transpose(image, (1,2,0))
    elif image.ndim == 4:
        out = np.transpose(image,(1,2,3,0))
    elif image.ndim == 5:
        out = np.transpose(image,(1,2,3,4,0))
        
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)

    if image.ndim == 3:
        out = np.transpose(out, (2,0,1))
    elif image.ndim == 4:
        out = np.transpose(out,(3,0,1,2))
    elif image.ndim == 5:
        out = np.transpose(out,(4,0,1,2,3))
    return out

def Im2Patch(img, win, stride=1):

    indrem = []
    for i in range(img.shape[-1]):
        img_ = img[0,:,:,i]/np.max(img[0,:,:,i])
        if np.sum(img_)>img_.shape[0]*img_.shape[1]-500:
            indrem.append(i)
    img = np.delete(img,indrem,axis=3)

    l = 0
    endc = int(img.shape[0])
    endw = int(img.shape[1])
    endh = int(img.shape[2])
    endd = int(img.shape[3])
    winx = int(win[0])
    winy = int(win[1])
    winz = int(win[2])
    stride = int(stride)

    patch = img[:, 0:endw-winx+0+1:stride, 0:endh-winy+0+1:stride, 0:endd-winz+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2] * patch.shape[3]
    Y = np.zeros([endc, winx*winy*winz, TotalPatNum], np.float32)
    for i in range(winx):
        for j in range(winy):
            for k in range(winz):
                patch = img[:,i:endw-winx+i+1:stride,j:endh-winy+j+1:stride,k:endd-winz+k+1:stride]
                Y[:,l,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
                l = l + 1
    out_patch = Y.reshape([endc, winx, winy, winz, TotalPatNum]) 
    return out_patch

def prepare_data(data_path, patch_size, stride, aug_times=1, fname='dwi'):
    # train
    print('process training data')
    scales = [1]
    files = glob.glob(os.path.join(data_path, 'train_dwi', '*.mat'))
    files.sort()
    
    h5f = h5py.File(fname + '_train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        mat = sio.loadmat(files[i])
        img = np.array(mat['dwi']).astype('float32').squeeze()
        h, w, d = img.shape
        for k in range(len(scales)):
            Img = zoom(img, scales[k])
            Img = np.expand_dims(Img.copy(), 0)
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[-1]*aug_times))
            for n in range(patches.shape[-1]):
                data = patches[...,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    aug_mode = np.random.randint(1,8)
                    data_aug = data_augmentation(data, aug_mode)
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'set12_dwi', '*.mat'))
    files.sort()
                                  
    h5f = h5py.File(fname + '_val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        mat = sio.loadmat(files[i])
        img = np.array(mat['dwi']).astype('float32').squeeze()
        img = np.expand_dims(img, 0)

        indrem = []
        for i in range(img.shape[-1]):
            img_ = img[0,:,:,i]/np.max(img[0,:,:,i])
            if np.sum(img_)>=img_.shape[0]*img_.shape[1]-500:
                indrem.append(i)
        img = np.delete(img,indrem,axis=3)

        data = img[...,:patch_size[2]].copy()
        h5f.create_dataset(str(val_num), data=data)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True, fname='mri'):
        super(Dataset, self).__init__()
        self.train = train
        self.fname = fname
        if self.train:
            h5f = h5py.File(fname + '_train.h5', 'r')
        else:
            h5f = h5py.File(fname + '_val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.fname + '_train.h5', 'r')
        else:
            h5f = h5py.File(self.fname + '_val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)