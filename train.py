import os
import torch
import argparse
from tqdm import tqdm
from torchinfo import summary

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
        '--save_path',
        default='ckpts',
        help='Path for checkpointing.',
    )
    parser.add_argument(
        '--resume',
        help='Resume training from saved checkpoint(s).',
    )
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='Finetune with L1-loss',
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=2,
        help='Checkpoint model every x epochs.',
    )
    parser.add_argument(
        '--N_iter',
        type=int,
        default=10,
        help='Number of unrolled iterations.',
    )
    parser.add_argument(
        '--set_lr',
        type=float,
        default=-1,
        help='Set new learning rate.',
    )
    return parser.parse_args()


def train(opt):

    torch.backends.cudnn.benchmark = True

    data_train = MSI_Dataset(file_path=opt.data)
    data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, num_workers=4)

    _, PAN_image, reference = next(iter(data_train_loader))
    HSI_channels = reference.shape[1]
    
    if opt.arch == 'UTeRM_CS':
        model = UCS(HSI_channels=HSI_channels).cuda()
    elif opt.arch == 'UTeRM_MRA':
        model = UMRA(HSI_channels=HSI_channels).cuda()
    elif opt.arch == 'UTeRM_CNN':
        model = UCNN(HSI_channels=HSI_channels).cuda()
    else:
        print('Incorrect architecture.')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss = torch.nn.L1Loss()
    summary(model, input_size=[reference.shape, PAN_image.shape])

    if opt.resume is not None:
        print('Resume training from' + opt.resume)
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_0 = checkpoint['epoch'] + 1
        model.train()
    else:
        print('Start training from scratch.')
        epoch_0 = 1

    if not opt.set_lr == -1:
        for groups in optimizer.param_groups: groups['lr'] = opt.set_lr; break
        print('New learning rate:', end=" ")
        for groups in optimizer.param_groups: print(groups['lr']); break

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    else:
        print('WARNING: save_path already exists. Checkpoints may be overwritten.')

    avg_loss = 0
    if opt.finetune:
        max_epoch = 100
    else:
        max_epoch = 90
    for epoch in tqdm(range(epoch_0, max_epoch+1), desc='Training'):
        for i, (upms_image, pan_image, reference) in enumerate(tqdm(data_train_loader, desc=f'Epoch {epoch}')):

            upms_image, pan_image, reference = upms_image.cuda(), pan_image.cuda(), reference.cuda()

            cs_comp, hrhs = model(upms_image, pan_image)
            if not opt.finetune:
                total_loss = loss(hrhs, reference) + 0.25 * loss(cs_comp, reference)
            else:
                total_loss = loss(hrhs, reference)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            avg_loss += total_loss.item()
            if ((i + 1) % 20) == 0:
                rep = (
                    f'Epoch: {epoch:>5d}, '
                    f'Iter: {i+1:>6d}, '
                    f'Loss: {avg_loss/20:>6.2e}'
                )
                tqdm.write(rep)
                avg_loss = 0

        if (epoch % opt.checkpoint_freq) == 0:
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(opt.save_path, f'epoch_{epoch}.pth')
            )


if __name__ == '__main__':
    opt_args = parse_args()
    train(opt_args)
