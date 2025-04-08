import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            if net.n_classes > 1:
                # 多分类问题，将目标张量转换为 torch.long 类型
                true_masks = true_masks.argmax(dim=1).to(device=device, dtype=torch.long)
            else:
                true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)  # 输出范围[0,1]
                tot += F.mse_loss(pred, true_masks).item()  # 使用MSE
            pbar.update()

    net.train()
    return tot / n_val