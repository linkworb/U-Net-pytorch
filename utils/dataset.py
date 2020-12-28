import logging
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, cfg, imgs_dir, masks_dir):
        self.cfg = cfg
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.imgs = os.listdir(self.imgs_dir)
        self.masks = [img.replace('jpg', 'png') for img in self.imgs]
        logging.info(f'Creating dataset with {len(self.imgs)} examples')

    def __len__(self):
        return len(self.imgs)

    def preprocess(self, pil_img, is_mask=False):
        pil_img = pil_img.resize((self.cfg.img_w, self.cfg.img_h))
        img_np = np.array(pil_img)
        if not is_mask and self.cfg.rgb:
            if len(img_np.shape) == 2:
                img_np = np.expand_dims(img_np, axis=2)
        if not is_mask:
            # HxWxC to CxHxW
            img_np = img_np.transpose((2, 0, 1))
            img_np = img_np / 255
        return img_np

    def __getitem__(self, index):
        img_name = self.imgs[index]
        mask_name = self.masks[index]

        # mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        # img_file = glob(self.imgs_dir + idx + '.*')

        # assert len(mask_file) == 1, \
        #     f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        # assert len(img_file) == 1, \
        #     f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(os.path.join(self.imgs_dir, img_name))
        mask = Image.open(os.path.join(self.masks_dir, mask_name))

        assert img.size == mask.size, \
            f'Image and mask {index} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        return {
            'image': torch.from_numpy(img).type(torch.float32),
            'mask': torch.from_numpy(mask).type(torch.float32)
        }
