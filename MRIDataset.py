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

def normalize(data):
    ratio = np.amax(data) / 255
    data = (data / ratio).astype('uint8')/255.
    data = data.astype('float32')
    return data, ratio

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

def Im2Patch(img, win, stride=1, dim=3):
    l = 0
    endc = int(img.shape[0])
    endw = int(img.shape[1])
    endh = int(img.shape[2])
    endd = int(img.shape[3])
    win = int(win)
    stride = int(stride)

    xlen = range(endw//4,3*endw//4)
    ylen = range(endh//4,3*endh//4)
    zlen = range(endd//4,3*endd//4)

    print(img.shape)
    img = img[:,endw//4:3*endw//4, endh//4:3*endh//4, endd//4:3*endd//4]
    endc = int(img.shape[0])
    endw = int(img.shape[1])
    endh = int(img.shape[2])
    endd = int(img.shape[3])
    print(img.shape)
    if dim==3:
        patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride, 0:endd-win+0+1:stride]
        TotalPatNum = patch.shape[1] * patch.shape[2] * patch.shape[3]
        Y = np.zeros([endc, win*win*win, TotalPatNum], np.float32)
        for i in range(win):
            for j in range(win):
                for k in range(win):
                    patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride,k:endd-win+k+1:stride]
                    Y[:,l,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
                    l = l + 1
        out_patch = Y.reshape([endc, win, win, win, TotalPatNum]) 
    if dim==2:
        patch = img[:, 0:int(endw-win+0+1):stride, 0:int(endh-win+0+1):stride, 0]
        TotalPatNum = patch.shape[1] * patch.shape[2]
        Y = np.zeros([endc, win*win*len(zlen), TotalPatNum], np.float32)
        for i in range(win):
            for j in range(win):
                for k in range(endd):
                    patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride, k]
                    Y[:,l,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
                    l = l + 1
        out_patch = Y.reshape([endc, win, win, TotalPatNum*len(zlen)])
    return out_patch

def prepare_data(data_path, patch_size, stride, aug_times=1, dim=3, fname='mri'):
    # train
    print('process training data')
    scales = [1]
    files_clean = glob.glob(os.path.join(data_path, 'train', '*clean.nii'))
    files_clean.sort()
    files_noisy = glob.glob(os.path.join(data_path, 'train', '*noisy.nii'))
    files_noisy.sort()

    files_clean = files_clean[:100]
    file_noisy = files_noisy[:100]
    
    h5f = h5py.File(fname + '_train.h5', 'w')
    train_num = 0
    for i in range(len(files_clean)):
        print(files_clean[i])
        nii = ni.load(files_clean[i])
        img_clean = np.array(nii.dataobj).astype('float32')
        nii = ni.load(files_noisy[i])
        img_noisy = np.array(nii.dataobj).astype('float32')
        h, w, d = img_clean.shape
        for k in range(len(scales)):
            Img_clean = zoom(img_clean, scales[k])
            Img_clean = np.expand_dims(Img_clean.copy(), 0)
            Img_clean, ratio_c = normalize(Img_clean)
            patches_clean = Im2Patch(Img_clean, win=patch_size, stride=stride, dim=dim)
            
            Img_noisy = zoom(img_noisy, scales[k])
            Img_noisy = np.expand_dims(Img_noisy.copy(), 0)
            Img_noisy, ratio_n = normalize(Img_noisy)
            patches_noisy = Im2Patch(Img_noisy, win=patch_size, stride=stride, dim=dim)
            
            print("file: %s scale %.1f # samples: %d" % (files_clean[i], scales[k], patches_clean.shape[-1]*aug_times))
            for n in range(patches_clean.shape[-1]):
                data_clean = patches_clean[...,n].copy()
                data_noisy = patches_noisy[...,n].copy()
                h5f.create_dataset(str(train_num), data=(data_clean, data_noisy))
                train_num += 1
                for m in range(aug_times-1):
                    aug_mode = np.random.randint(1,8)
                    data_aug_clean = data_augmentation(data_clean, aug_mode)
                    data_aug_noisy = data_augmentation(data_noisy, aug_mode)
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=(data_aug_clean, data_aug_noisy))
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files_clean.clear()
    files_clean = glob.glob(os.path.join(data_path, 'val', '*clean.nii'))
    files_clean.sort()
    files_noisy.clear()
    files_noisy = glob.glob(os.path.join(data_path, 'val', '*noisy.nii'))
    files_noisy.sort()                                
    files_clean = files_clean[:20]
    files_noisy = files_noisy[:20]
    print(files_clean)
    print(files_noisy)
    h5f = h5py.File(fname + '_val.h5', 'w')
    val_num = 0
    for i in range(len(files_clean)):
        print("file: %s" % files_clean[i])
        nii = ni.load(files_clean[i])
        img_clean = np.array(nii.dataobj).astype('float32')
        img_clean = np.expand_dims(img_clean, 0)
        img_clean, ratio_c = normalize(img_clean)
                                       
        nii = ni.load(files_noisy[i])
        img_noisy = np.array(nii.dataobj).astype('float32')
        img_noisy = np.expand_dims(img_noisy, 0)
        img_noisy, ratio_n = normalize(img_noisy)

        if dim == 2:
            #for n in range(img_clean.shape[-1]):
            data_clean = img_clean[...,124].copy()
            data_noisy = img_noisy[...,124].copy()
            h5f.create_dataset(str(val_num), data=(data_clean, data_noisy))
            val_num += 1
        if dim == 3:
            print(img_clean.shape)
            print(val_num)
            data_clean = img_clean.copy()
            data_noisy = img_noisy.copy()
            h5f.create_dataset(str(val_num), data=(data_clean, data_noisy))
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