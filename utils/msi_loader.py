import torch.utils.data as data
import scipy.ndimage as ndimage
import numpy as np
import torch
import h5py
import os


def interp23tap(image, ratio):
    if (2**round(np.log2(ratio)) != ratio):
        print('Error: only resize factors of power 2')
        return

    b,r,c = image.shape

    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    for z in range(1,np.int32(np.log2(ratio))+1):
        I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2]=image
            first = 0
        else:
            I1LRU[:,0:I1LRU.shape[1]:2,0:I1LRU.shape[2]:2]=image
        
        for ii in range(0,b):
            t = I1LRU[ii,:,:]
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            I1LRU[ii,:,:]=t
        image=I1LRU
        
    return image


class MSI_Dataset(data.Dataset):
    def __init__(self, file_path):
        super(MSI_Dataset, self).__init__()
        data = h5py.File(file_path,'r+')  # NxCxHxW = 0x1x2x3=nx191x64x64   channel height width

        self.gt   = data.get("HRMSS")
        self.upms = data.get("USMSS")
        self.pan  = data.get("PANS")

    def __getitem__(self, index):
        reference  = torch.from_numpy(self.gt[index, :, :, :]).float()
        PAN_image  = torch.from_numpy(self.pan[index, :, :, :]).float()
        UpMS_image = torch.from_numpy(self.upms[index, :, :, :]).float()
        return UpMS_image, PAN_image, reference

    def __len__(self):
        return self.gt.shape[0]


class MSI_Dataset_FR(data.Dataset):
    def __init__(self, file_path):
        super(MSI_Dataset_FR, self).__init__()
        data = h5py.File(file_path,'r+')  # NxCxHxW = 0x1x2x3=nx191x64x64   channel height width

        self.ms  = data.get("HRMSS")
        self.pan = data.get("FPANS")

    def __getitem__(self, index):
        upms = interp23tap(self.ms[index, :, :, :], 4)
        PAN_image = torch.from_numpy(self.pan[index, :, :, :]).float()
        MS_image  = torch.from_numpy(upms).float()
        return MS_image, PAN_image

    def __len__(self):
        return self.pan.shape[0]
