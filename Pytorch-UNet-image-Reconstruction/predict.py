import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

import glob  # 添加 glob 模块

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    print(f"Predicted mask min: {full_mask.min()}, max: {full_mask.max()}")  # 检查预测输出的像素值范围

    # 暂时去掉二值化步骤，直接返回概率图
    # return full_mask > out_threshold
    return full_mask

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    # return parser.parse_args()
    args = parser.parse_args()

    # 处理通配符，获取所有匹配的文件路径
    if args.input:
        expanded_inputs = []
        for input_pattern in args.input:
            if input_pattern.endswith('.tif'):
                expanded_inputs.extend(glob.glob(input_pattern))
        args.input = expanded_inputs

    return args


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    # 如果 --output 是一个文件夹，自动生成输出文件名
    if len(args.output) == 1 and os.path.isdir(args.output[0]):
        for in_file in in_files:
            file_name = os.path.basename(in_file)
            out_files.append(os.path.join(args.output[0], file_name))
    elif len(args.output) == len(in_files):
        out_files = args.output
    else:
        raise ValueError("输出文件参数的数量不匹配输入文件的数量，或者指定的输出不是有效的文件夹。")

    return out_files

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=3, n_classes=3)  # 根据实际情况修改输入输出通道数
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device=device)

    in_files = args.input
    out_files = get_output_filenames(args)

    for i, (in_file, out_file) in enumerate(zip(in_files, out_files)):
        img = Image.open(in_file)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # 进一步确保 mask 数组形状正确
        while mask.ndim > 2:
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            elif mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            else:
                # 如果无法通过 squeeze 处理，尝试取第一个通道
                mask = mask[0]

        if not args.no_save:
            result = Image.fromarray((mask * 255).astype(np.uint8))
            result.save(out_file)
            print(f'Mask saved to {out_file}')

        if args.viz:
            print('Visualizing results for image, close to continue...')
            plot_img_and_mask(img, mask)

if __name__ == '__main__':
    main()