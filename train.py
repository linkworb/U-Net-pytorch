import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from eval import eval_net, save_eval_vis
from model import UNet
from utils.dataset import BasicDataset


def train_net(cfg,
              model,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1):
    dataset = BasicDataset(cfg, args.img_dir, args.mask_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    # train, val = dataset, dataset
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    if cfg.log_dir:
        writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.exp_name))
    else:
        writer = SummaryWriter(comment='_' + cfg.exp_name)
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                 eps=cfg.eps, weight_decay=cfg.weight_decay)
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            imgs = batch['image']
            true_masks = batch['mask']
            assert imgs.shape[1] == model.in_channels, \
                f'Network has been defined with {model.in_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            masks_pred = model(imgs)
            if net.n_classes == 1:
                masks_pred = torch.sigmoid(masks_pred)
                true_masks = true_masks.unsqueeze(dim=1)
            else:
                masks_pred = torch.softmax(masks_pred, dim=1)
            loss = criterion(masks_pred, true_masks)
            '''print('_:',(true_masks.unique()))
            print('0:',(true_masks == 0).sum().item())
            print('1:',(true_masks == 1).sum().item())
            print('2:',(true_masks == 2).sum().item())
            exit(0)'''
            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            print(f'\repoch: {epoch + 1:3d}/{epochs:<3d}\ttrain loss: {loss.item():<6.2f}', end='')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            global_step += 1
        print()
        if (epoch + 1) % 5 == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            imgs, true_masks, masks_pred, val_score = eval_net(model, val_loader, device, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if model.n_classes > 1:
                logging.info('Validation cross entropy: {}'.format(val_score))
                writer.add_scalar('Loss/test', val_score, global_step)
            else:
                logging.info('Validation Dice Coeff: {}'.format(val_score))
                writer.add_scalar('Dice/test', val_score, global_step)

            if model.n_classes == 1:
                writer.add_images('images', imgs, global_step)
                writer.add_images('masks/true', true_masks, global_step)
                writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
            else:
                save_eval_vis(masks_pred, true_masks, epoch + 1)
                # # masks_pred = torch.softmax(masks_pred, dim=1)
                # masks_pred = torch.argmax(masks_pred, dim=1)
                # res = torch.cat((masks_pred, true_masks), dim=1)
                # res = res.data.cpu().numpy()
                # for i in res:
                #     i = PIL.Image.fromarray(i.astype(np.uint8), mode="P")
                #     colormap = imgviz.label_colormap()
                #     i.putpalette(colormap.flatten())
                #     i.save(f'./res/pred_{global_step}.png')

                '''masks_pred = masks_pred.data.cpu().squeeze().numpy()
                # print(masks_pred.shape)
                masks_pred = PIL.Image.fromarray(masks_pred.astype(np.uint8), mode="P")
                colormap = imgviz.label_colormap()
                masks_pred.putpalette(colormap.flatten())
                masks_pred.save(f'./res/pred_{global_step}.png')'''

                # masks_pred //= max(masks_pred)
                # masks_pred = masks_pred.repeat(1, 3, 1, 1)
                # writer.add_images('masks/true', true_masks.unsqueeze(0), global_step)
                # writer.add_images('masks/pred', masks_pred * 255, global_step)

            if cfg.save_dir:
                try:
                    os.makedirs(cfg.save_dir)
                    logging.info(f'Created checkpoint directory {cfg.save_dir}')
                except OSError:
                    pass
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, f'{epoch + 1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved to {cfg.save_dir}!')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # model
    parser.add_argument('--model_dir', type=str, help='trained model path to load')
    parser.add_argument('--save_dir', type=str, help='trained model path to save')
    # dataset
    parser.add_argument('--img_dir', type=str, default='./data/imgs', help='dataset images path')
    parser.add_argument('--mask_dir', type=str, default='./data/masks', help='dataset mask path')
    parser.add_argument('--img_w', type=int, default=400, help='width of image and mask')
    parser.add_argument('--img_h', type=int, default=400, help='height of image and mask')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--val_percent', type=float, default=0.1, help='percent of the validation data (0-1)')
    # optimizer
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam')
    # logger
    parser.add_argument('--exp_name', type=str, default='exp', help='name of a specific experiment')
    parser.add_argument('--log_dir', type=str, help='path to save log')
    # others
    parser.add_argument('--rgb', action='store_true', help='use rgb images')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of class. For 1 class and background use classes=1. '
                             'For n>1 class and background use classes=n+1')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.rgb:
        args.in_channels = 3
    else:
        args.in_channels = 1
    net = UNet(in_channels=args.in_channels, n_classes=args.classes, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.model_dir:
        net.load_state_dict(
            torch.load(args.model_dir, map_location=device)
        )
        logging.info(f'Model loaded from {args.model_dir}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(cfg=args,
                  model=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val_percent)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
