import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2]) # (b, 2048, 8, 8)
        self.conv = nn.Conv2d(2048, 1, kernel_size=1)
        self.norm = nn.BatchNorm2d(1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.norm(x)
        x = torch.cat((x, 1-x), dim=1)
        # x = F.interpolate(x, scale_factor=32, mode='nearest')
        return x

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(DenseNetFeatureExtractor, self).__init__()
        # Load DenseNet121 model (pretrained=False for feature extraction purposes)
        densenet = models.densenet121(pretrained=False)
        self.features = nn.Sequential(*list(densenet.children())[:-1])  # Remove the classifier layer
        
        # Adding a convolution layer to reduce the output to [1, 2, 8, 8]
        self.conv = nn.Conv2d(1024, 2, kernel_size=1)  # 1024 is the output size of DenseNet121

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        # Ensure the output shape is [1, 2, 8, 8]
        x = F.interpolate(x, size=(8, 8), mode='nearest')
        return x

class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetV2FeatureExtractor, self).__init__()
        # Load MobileNetV2 model (pretrained=False for feature extraction purposes)
        mobilenet_v2 = models.mobilenet_v2(pretrained=False)
        self.features = mobilenet_v2.features
        
        # Adding a convolution layer to reduce the output to [1, 2, 8, 8]
        self.conv = nn.Conv2d(1280, 2, kernel_size=1)  # 1280 is the output size of MobileNetV2

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        # Ensure the output shape is [1, 2, 8, 8]
        x = F.interpolate(x, size=(8, 8), mode='nearest')
        return x

from thop import profile
def main():
    device = torch.device("cuda:1")

    input_tensor = torch.randn(1, 3, 256, 256, device=device)
    model = DenseNetFeatureExtractor().to(device)
    model.eval()

    # Forward once (no grad) just like you had
    with torch.no_grad():
        output = model(input_tensor)
    print("Output shape:", tuple(output.shape))

    # ---- THOP: FLOPs & params ----
    # Note: THOP returns FLOPs for the given input shape (includes batch dim).
    flops, params_total = profile(model, inputs=(input_tensor,), verbose=False)

    # Only learnable parameters
    params_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Convert units
    params_m = params_learnable / 1e6
    flops_g = flops / 1e9  # FLOPs for the provided batch (here batch=1)

    # If you want per-image FLOPs regardless of batch size:
    bs = input_tensor.shape[0]
    flops_per_image_g = (flops / max(bs, 1)) / 1e9

    print(f"Learnable parameters: {params_m:.3f} M")
    print(f"FLOPs (forward, batch={bs}): {flops_g:.3f} G")
    print(f"FLOPs per image: {flops_per_image_g:.3f} G")

if __name__ == '__main__':
    main()