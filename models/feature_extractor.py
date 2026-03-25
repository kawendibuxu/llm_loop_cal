import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    """
    使用预训练的CNN骨干网络提取图像特征
    """

    def __init__(self, backbone_name='resnet18', pretrained=True):
        super(FeatureExtractor, self).__init__()

        # Load a pre-trained ResNet model
        self.backbone = getattr(models, backbone_name)(pretrained=pretrained)

        # Remove the final classification layer to get features
        # For ResNet, the last layer is `fc`. We'll take the output of the layer before it.
        # The output dimension of resnet18's avgpool is 512.
        self.backbone.fc = nn.Identity()  # Replace fc layer with identity

        # Add a normalization layer (optional but often helpful)
        self.normalize = nn.functional.normalize

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Normalized feature vector of shape (B, D)
        """
        features = self.backbone(x)
        features = self.normalize(features, p=2, dim=1)  # L2 normalize
        return features

    def extract_feature(self, image_rgb):
        """
        从单张RGB图像中提取特征
        Args:
            image_rgb (np.ndarray): Shape (H, W, C), values in [0, 255]
        Returns:
            torch.Tensor: Shape (1, D)
        """
        # Preprocess: ToTensor and Normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize to model's expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])

        image_tensor = transform(image_rgb).unsqueeze(0).to(next(self.parameters()).device)  # Add batch dimension

        self.eval()
        with torch.no_grad():
            feature = self(image_tensor)
        return feature.squeeze(0)  # Remove batch dimension


# Note: You need to import transforms at the top of the file
from torchvision import transforms