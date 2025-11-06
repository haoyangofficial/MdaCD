import torch
import os
from PIL import Image
import os
import torch.nn.functional as F
from CLIP_tools import *
from Fill_blocks import *

def loss(a, b):
    zeros = torch.relu(a[:, 0, :, :] - a[:, 1, :, :])
    ones = torch.relu(a[:, 1, :, :] - a[:, 0, :, :])
    large_loss = b.cpu() * zeros.cpu()
    small_loss = .1 * (1-b).cpu() * ones.cpu()
    # total_loss = large_loss.sum()
    total_loss = (large_loss.sum() + small_loss.sum())
    # print(f'a = {a}')
    # input('')
    return total_loss

def remove_png_files(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith(('.png', '.json')):
                file_path = os.path.join(directory_path, filename)
                os.remove(file_path)

def CLIPloss(batch_output, batch_label, prefix, penalty, 
             batch_time1, batch_time2, device, fill_nearest):
    output_argmax = torch.argmax(batch_output, dim=1, keepdim=False)

    if not os.path.exists(prefix):
        os.mkdir(prefix)
    temp1_dir = os.path.join(prefix, 'temp1')
    temp2_dir = os.path.join(prefix, 'temp2')
    if not os.path.exists(temp1_dir):
        os.mkdir(temp1_dir)
    if not os.path.exists(temp2_dir):
        os.mkdir(temp2_dir)
    remove_png_files(temp1_dir)
    remove_png_files(temp2_dir)
    for j in range(batch_output.shape[0]):
        # (3, H, W), bool
        mask = 1-output_argmax[j].unsqueeze(0).repeat(3, 1, 1)
        mask = F.interpolate(mask.to(torch.float).unsqueeze(0), scale_factor=32,
                             mode='nearest').squeeze(0)
        mask1 = mask * batch_time1[j] * 255
        mask2 = mask * batch_time2[j] * 255
        single_image = Image.fromarray(mask1.byte().cpu().permute(1, 2, 0).numpy())
        single_image.save(os.path.join(temp1_dir, f'{j}.png'))
        single_image = Image.fromarray(mask2.byte().cpu().permute(1, 2, 0).numpy())
        single_image.save(os.path.join(temp2_dir, f'{j}.png'))
    if fill_nearest is True:
        fill_empty_batches_in_images(temp1_dir)
        fill_empty_batches_in_images(temp2_dir)
    CLIP_forward(src_path=prefix, device=device)
    clip_loss = penalty * json_difference(os.path.join(temp1_dir, 'temp1.json'),
                                          os.path.join(temp2_dir, 'temp2.json'))
    num_loss = loss(batch_output, batch_label)
    return -clip_loss + num_loss