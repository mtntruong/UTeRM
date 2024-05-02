import os
import torch
import argparse
from tqdm import tqdm
from scipy.io import savemat

from model.UTeRM_CS import LRTC_Net as UCS
from model.UTeRM_MRA import LRTC_Net as UMRA
from model.UTeRM_CNN import LRTC_Net as UCNN
from utils.msi_loader import MSI_Dataset

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

    data_test = MSI_Dataset(file_path=opt.data)
    data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, num_workers=4)

    tmp_msi, _, _ = next(iter(data_test_loader))
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

    with torch.no_grad():
        for i, (upms_image, pan_image, reference) in enumerate(tqdm(data_test_loader)):
            upms_image, pan_image, reference = upms_image.cuda(), pan_image.cuda(), reference.cuda()
            _, hrhs = model(upms_image, pan_image)
            hrhs = torch.squeeze(hrhs).permute(2, 1, 0)
            hrhs = hrhs.cpu().numpy()
            savemat(os.path.join(opt.save_path, str(i+1).zfill(3)+'.mat'), {'Xhat': hrhs})
