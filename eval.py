from PIL import Image
import imgviz
import torch
import torch.nn.functional as F
import numpy as np

from dice_loss import dice_coeff


@torch.no_grad()
def eval_net(net, loader, device, epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    img = None
    mask = None
    pred = None
    for index, batch in enumerate(loader):
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)
        mask_pred = net(imgs)
        if net.n_classes > 1:
            mask_pred = torch.softmax(mask_pred, dim=1)
            tot += F.cross_entropy(mask_pred, true_masks).item()
            if index == n_val - 1:
                save_eval_vis(mask_pred, true_masks, epoch)
                img = imgs
                mask = true_masks
                pred = mask_pred
                # mask_pred = torch.argmax(mask_pred, dim=1)
                # res = torch.cat((mask_pred, true_masks), dim=1)
                # res = res.data.cpu().numpy()
                # for i in res:
                #     i = PIL.Image.fromarray(i.astype(np.uint8), mode="P")
                #     colormap = imgviz.label_colormap()
                #     i.putpalette(colormap.flatten())
                #     i.save(f'./res/pred_{epoch}.png')
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            true_masks = true_masks.unsqueeze(dim=1)
            tot += dice_coeff(pred, true_masks).item()
            if index == n_val - 1:
                img = imgs
                mask = true_masks
                pred = pred

    net.train()
    return img, mask, pred, tot / n_val


def save_eval_vis(mask_pred, true_masks, epoch):
    mask_pred = torch.argmax(mask_pred, dim=1)
    res = torch.cat((mask_pred, true_masks), dim=1)
    res = res.data.cpu().numpy()
    for i in res:
        i = Image.fromarray(i.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        i.putpalette(colormap.flatten())
        i.save(f'./res/pred_{epoch}.png')
