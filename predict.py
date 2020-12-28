import argparse
import logging
import os

import imgviz
import numpy as np
import torch
from PIL import Image

from model import UNet
from utils.data_vis import plot_img_and_mask


def preprocess(pil_img, cfg):
    pil_img = pil_img.resize((cfg.img_w, cfg.img_h))
    img_np = np.array(pil_img)
    if len(img_np.shape) == 2:
        img_np = np.expand_dims(img_np, axis=2)
    # HxWxC to CxHxW
    img_np = img_np.transpose((2, 0, 1))
    img_np = img_np / 255
    return img_np


@torch.no_grad()
def predict_img(cfg, net_, img_, device_, mask_threshold):
    net_.eval()
    img_ = torch.from_numpy(preprocess(img_, cfg))
    img_ = img_.unsqueeze(0)
    img_ = img_.to(device=device_, dtype=torch.float32)

    output = net_(img_)
    if net_.n_classes > 1:
        mask_pred_ = torch.softmax(output, dim=1)
    else:
        pred = torch.sigmoid(output)
        mask_pred_ = (pred > mask_threshold).float()
    return mask_pred_


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/model.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input_dir', type=str, default='./demo', help='path of input images')
    parser.add_argument('--output_dir', type=str, default='./output', help='path of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--rgb', action='store_true', help='use rgb images')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of class. For 1 class and background use classes=1. '
                             'For n>1 class and background use classes=n+1')
    parser.add_argument('--mask_threshold', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--img_w', type=int, default=400, help='width of image and mask')
    parser.add_argument('--img_h', type=int, default=400, help='height of image and mask')
    # parser.add_argument('--scale', '-s', type=float,
    #                     help="Scale factor for the input images",
    #                     default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    input_files = os.listdir(args.input_dir)

    if args.rgb:
        args.in_channels = 3
    else:
        args.in_channels = 1
    net = UNet(in_channels=args.in_channels, n_classes=args.classes)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    for i, file_name in enumerate(input_files):
        logging.info("\nPredicting image {} ...".format(file_name))
        img = Image.open(os.path.join(args.input_dir, file_name))
        mask = predict_img(args, net_=net, img_=img, device_=device, mask_threshold=args.mask_threshold)

        if net.n_classes > 1:
            mask_pred = torch.argmax(mask, dim=1)
            res = mask_pred
        else:
            res = mask.squeeze(0)
        res = res.squeeze(0).data.cpu().numpy()
        res = Image.fromarray(res.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        res.putpalette(colormap.flatten())
        if not args.no_save:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            res.save(os.path.join(args.output_dir, f'{file_name}.png'))
        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(file_name))
            plot_img_and_mask(img, np.array(res))
