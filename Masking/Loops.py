import torch
import torch.optim as optim
from torchvision import transforms
import copy
import os
from PIL import Image
from tqdm import tqdm
from Loss import *
import torch.nn.functional as F

def downsample_and_threshold(label, threshold=0.3):
    b, h, w = label.shape
    patch_size = 32

    output = torch.zeros((b, 8, 8), dtype=torch.uint8)

    for i in range(b):
        label_img = label[i]
        
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = label_img[y:y+patch_size, x:x+patch_size]

                non_black_pixels = torch.sum(patch > 0)
                
                if non_black_pixels >= patch_size * patch_size * threshold:
                    output[i, y // patch_size, x // patch_size] = 1
                else:
                    output[i, y // patch_size, x // patch_size] = 0
    return output

def kmax_mask(tensor, k):
    # keep only the first channel → [B, 8, 8]
    first_channel = tensor[:, 0, :, :]

    # flatten to [B, 64]
    flat = first_channel.view(first_channel.size(0), -1)

    # get top-k indices along dimension 1
    topk_vals, topk_idx = torch.topk(flat, k, dim=1)

    # build binary mask initialized to zeros
    mask = torch.zeros_like(flat)

    # set top-k positions to 1
    mask.scatter_(1, topk_idx, 1.0)

    # reshape back to [B, 8, 8]
    mask = mask.view_as(first_channel)

    return 1. - mask

def main_loop(model, device, args):
    if args.get('weight') is not None:
        model.load_state_dict(args['weight'])

    if 'train' == args['mode']:
        root_dir = args['train_dir']
        model.train()
    else:
        model.eval()

    if 'val' == args['mode']:
        root_dir = args['val_dir']
    
    if 'test' == args['mode']:
        root_dir = args['root_dir']
        mask1_dir = os.path.join(root_dir, f"{args['mask_dir']}1")
        mask2_dir = os.path.join(root_dir, f"{args['mask_dir']}2")
        if not os.path.exists(mask1_dir):
            print(f'mkdir {mask1_dir}')
            os.mkdir(mask1_dir)
        if not os.path.exists(mask2_dir):
            print(f'mkdir {mask2_dir}')
            os.mkdir(mask2_dir)
    else:
        optimizer = args['optimizer']
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = args['criterion']
        penalty = args['penalty']
        tv_loss = 0.0
        TP = TN = FP = FN = 0
        samples = 0

    time1_dir = os.path.join(root_dir, "time1")
    time2_dir = os.path.join(root_dir, "time2")
    label_dir = os.path.join(root_dir, "label")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    batch_size = 50
    file_names = sorted(os.listdir(time1_dir))

    for i in tqdm(range(0, len(file_names), batch_size), desc="Processing batches", unit="batch"):
    # for i in range(0, len(file_names), batch_size):
        batch_time1 = []
        batch_time2 = []
        
        for file_name in file_names[i:i + batch_size]:
            img_time1 = Image.open(os.path.join(time1_dir, file_name)).convert('RGB')
            img_time2 = Image.open(os.path.join(time2_dir, file_name)).convert('RGB')
            
            img_time1 = transform(img_time1)
            img_time2 = transform(img_time2)
            
            batch_time1.append(img_time1)
            batch_time2.append(img_time2)
        
        batch_time1 = torch.stack(batch_time1).to(device)  # (batch_size, 3, 256, 256)
        batch_time2 = torch.stack(batch_time2).to(device)  # (batch_size, 3, 256, 256)

        # (batch_size, 256, 256)
        batch_output = (model(batch_time1 - batch_time2)).to(device)
        # (batch_size, 256, 256), binary
        output_argmax = torch.argmax(batch_output, dim=1, keepdim=False)
        # kmax masking
        #output_argmax = kmax_mask(batch_output, args['k']) 

        if 'train' == args['mode']:
            batch_label = []
            for file_name in file_names[i:i + batch_size]:
                img_label = Image.open(os.path.join(label_dir, file_name)) # Binary
                img_label = transform(img_label).squeeze(0)
                batch_label.append(img_label)
            batch_label = torch.stack(batch_label).to(device).long()
            batch_label = downsample_and_threshold(batch_label).to(device)

            optimizer.zero_grad()
            loss = criterion(batch_output, batch_label, args['prefix'], penalty,
                             batch_time1, batch_time2, device, args['fill_nearest'])
            loss.backward()
            optimizer.step()
            # scheduler.step()

            tv_loss += loss.item() * batch_label.numel()
            TP += (output_argmax * batch_label).sum().item()
            TN += ((1-output_argmax) * (1-batch_label)).sum().item()
            FP += ((1-output_argmax) * batch_label).sum().item()
            FN += (output_argmax * (1-batch_label)).sum().item()
            samples += batch_label.numel()

        if 'val' == args['mode']:
            with torch.no_grad():
                batch_label = []
                for file_name in file_names[i:i + batch_size]:
                    img_label = Image.open(os.path.join(label_dir, file_name))
                    img_label = transform(img_label).squeeze(0)
                    batch_label.append(img_label)
                batch_label = torch.stack(batch_label).to(device).long()
                batch_label = downsample_and_threshold(batch_label).to(device)
                loss = criterion(batch_output, batch_label, args['prefix'], penalty,
                             batch_time1, batch_time2, device, args['fill_nearest'])

                tv_loss += loss.item() * batch_label.numel()
                TP += (output_argmax * batch_label).sum().item()
                TN += ((1-output_argmax) * (1-batch_label)).sum().item()
                FP += ((1-output_argmax) * batch_label).sum().item()
                FN += (output_argmax * (1-batch_label)).sum().item()
                samples += batch_label.numel()

        if 'test' == args['mode']:
            output_argmax = F.interpolate(output_argmax.to(torch.float).unsqueeze(0),
                                          scale_factor=32, mode='nearest').squeeze(0)
            for j in range(batch_output.shape[0]):
                # (3, H, W), bool
                mask = output_argmax[j].unsqueeze(0).repeat(3, 1, 1)
                mask1 = mask * batch_time1[j] * 255
                mask2 = mask * batch_time2[j] * 255
                single_image = Image.fromarray(mask1.byte().cpu().permute(1, 2, 0).numpy())
                single_image.save(os.path.join(mask1_dir, file_names[i + j]))
                single_image = Image.fromarray(mask2.byte().cpu().permute(1, 2, 0).numpy())
                single_image.save(os.path.join(mask2_dir, file_names[i + j]))

    if 'train' == args['mode']:
        tv_loss /= samples
        acc = (TP + TN) / samples
        pre = TP / (TP + FP)
        rec = TP / (TP + FN)
        print(f"Train Loss: {tv_loss:.4f}, Acc = {acc:.4f}, Pre = {pre:.4f}, Rec = {rec:.4f}")
        return acc, copy.deepcopy(model.state_dict())

    if 'val' == args['mode']:
        if 0 != samples:
            tv_loss /= samples
            acc = (TP + TN) / samples
            pre = TP / (TP + FP)
            rec = TP / (TP + FN)
        else:
            tv_loss = acc = pre = rec = 0.0
        print(f"Val Loss: {tv_loss:.4f}, Acc = {acc:.4f}, Pre = {pre:.4f}, Rec = {rec:.4f}")
        return pre

    if 'test' == args['mode']:
        if args['fill_nearest'] is True:
            fill_empty_batches_in_images(mask1_dir)
            fill_empty_batches_in_images(mask2_dir)
        print(f'Mask complete')
        return
    
def LoadBestWeight(model, device, args):
    loaded_model = model
    loaded_model.load_state_dict(torch.load(args['pthfile']))
    loaded_model = loaded_model.to(device)
    print("Model weights loaded.")
    return loaded_model

def TrainValidate(model, device, args):
    best_pre = 0.
    args['optimizer'] = optim.Adam(model.parameters(), lr=1e-6)
    args['criterion'] = CLIPloss
    args['train_dir'] = os.path.join(args['root_dir'], 'train')
    args['val_dir'] = os.path.join(args['root_dir'], 'val')

    for e in range(args['num_epochs']):
        print(f'Epoch {e}')
        args['mode'] = 'train'
        _, args['weight'] = main_loop(model, device=device, args=args)
        args['mode'] = 'val'
        pre = main_loop(model, device=device, args=args)
        if pre >= best_pre:
            best_pre = pre
            best_weight = args['weight']
        torch.save(best_weight, args['pthfile'])

def Mask(model, device, args):
    model = LoadBestWeight(model=model, device=device, args=args)
    args['mode'] = 'test'
    main_loop(model, device=device, args=args)