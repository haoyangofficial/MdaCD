import torch
import Backbone
import Loss
import Loops
import os


def main():
    device = torch.device('cuda:0')
    print(f'device={device}')
    model = Backbone.ResNetFeatureExtractor().to(device)

    root_dir = '/home/haoyang/data/data/MdaCD/Dataset/LEVIRCD'
    penalties = [10, 100]
    directories = ['train', 'val', 'test']
    directories = [os.path.join(root_dir, directory) for directory in directories]
    for penalty in penalties:
        print(f'Penalty {penalty}, training & validating')
        args = {'prefix': 'LEVIR',
                'root_dir': root_dir,
                'pthfile': f'penalty_LEVIRCD_{penalty}.pth',
                'mask_dir': f'penalty_{penalty}_mask',
                'num_epochs': 50,
                'fill_nearest': False,
                'penalty': penalty}
        # Train and Validate
        Loops.TrainValidate(model=model, device=device, args=args)

        # Mask
        for directory in directories:
            print(f'Penalty {penalty}, masking {directory}')
            args['root_dir'] = directory
            Loops.Mask(model, device=device, args=args)
    print('Done')

if __name__ == "__main__":
    main()