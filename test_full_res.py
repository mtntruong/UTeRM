import os
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

from model.UTeRM_CS import LRTC_Net as UCS
from model.UTeRM_MRA import LRTC_Net as UMRA
from model.UTeRM_CNN import LRTC_Net as UCNN
from utils.msi_loader import MSI_Dataset_FR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch',
        required=True,
        help='Architecture. UTeRM_CS, UTeRM_MRA, or UTeRM_CNN',
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Multispectral data in H5 format.',
    )
    parser.add_argument(
        '--weight',
        required=True,
        help='Weight for testing.',
    )
    parser.add_argument(
        '--save_path',
        default='HRMS',
        help='Path to save images.',
    )
    return parser.parse_args()


if __name__ == '__main__':

    opt = parse_args()
    os.makedirs(opt.save_path, exist_ok=True)

    data_test = MSI_Dataset_FR(file_path=opt.data)
    data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, num_workers=4)

    tmp_msi, _ = next(iter(data_test_loader))
    HSI_channels = tmp_msi.shape[1]

    if opt.arch == 'UTeRM_CS':
        model = UCS(HSI_channels=HSI_channels).cuda()
    elif opt.arch == 'UTeRM_MRA':
        model = UMRA(HSI_channels=HSI_channels).cuda()
    elif opt.arch == 'UTeRM_CNN':
        model = UCNN(HSI_channels=HSI_channels).cuda()
    else:
        print('Incorrect architecture.')

    checkpoint = torch.load(opt.weight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Grid
    w_grid = [0, 256, 512, 768]
    h_grid = [0, 256, 512, 768]

    w_grid = np.array(w_grid, dtype=np.uint16)
    h_grid = np.array(h_grid, dtype=np.uint16)

    with torch.no_grad():
        for idx, (ms_image, pan_image) in enumerate(tqdm(data_test_loader)):
            hrhs = torch.zeros(HSI_channels, 1024, 1024)
            i = 0
            j = 0
            while i < len(h_grid):
                while j < len(w_grid):
                    h = h_grid[i]
                    w = w_grid[j]

                    ms_patch = copy.deepcopy(ms_image[:, :, w:w + 256, h:h + 256])
                    pan_patch = copy.deepcopy(pan_image[:, :, w:w + 256, h:h + 256])
                    
                    ms_patch, pan_patch = ms_patch.cuda(), pan_patch.cuda()
                    _, out = model(ms_patch, pan_patch)

                    # Stitching
                    hrhs[:, w:w + 256, h:h + 256] = torch.squeeze(out.cpu())
                    j = j + 1
                i = i + 1
                j = 0
            
            hrhs = hrhs.permute(2, 1, 0)
            hrhs = hrhs.numpy()
            savemat(os.path.join(opt.save_path, str(idx+1).zfill(3)+'.mat'), {'Xhat': hrhs})
