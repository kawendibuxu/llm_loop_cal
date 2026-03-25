# models/attribute_predictor.py
import torch
import torch.nn as nn
import torchvision.models as models


class AttributePredictor(nn.Module):
    """
    属性预测器：预测图像的多个离散属性，如物体、颜色、场景类型等。
    这是一个更易于实现和训练的过渡方案。
    """

    def __init__(self, backbone_name='resnet18', num_objects=10, num_colors=8, num_scenes=5):
        super(AttributePredictor, self).__init__()

        self.backbone = getattr(models, backbone_name)(pretrained=True)
        self.backbone.fc = nn.Identity()

        feature_dim = self.backbone.layer4[-1].conv2.out_channels  # e.g., 512 for ResNet18

        # 分别预测不同的属性
        self.object_classifier = nn.Linear(feature_dim, num_objects)
        self.color_classifier = nn.Linear(feature_dim, num_colors)
        self.scene_classifier = nn.Linear(feature_dim, num_scenes)

        # 定义标签名称（这些需要根据你的训练数据来设定）
        self.object_names = ["car", "building", "tree", "person", "traffic_light", "road", "sky", "sign",
                             "parking_meter", "lamppost"]
        self.color_names = ["red", "blue", "green", "yellow", "black", "white", "gray", "brown"]
        self.scene_names = ["urban_street", "highway", "intersection", "parking_lot", "residential"]

    def forward(self, x):
        features = self.backbone(x)  # Shape: (B, 512)

        obj_logits = self.object_classifier(features)  # (B, num_objects)
        col_logits = self.color_classifier(features)  # (B, num_colors)
        scn_logits = self.scene_classifier(features)  # (B, num_scenes)

        return {
            'objects': obj_logits,
            'colors': col_logits,
            'scenes': scn_logits
        }

    def predict_and_format(self, image):
        """
        预测属性并格式化为结构化文本。
        """
        self.eval()
        with torch.no_grad():
            image_tensor = self.preprocess_image(image).unsqueeze(0).to(next(self.parameters()).device)

            outputs = self(image_tensor)

            # 获取预测的类别ID
            pred_obj_id = torch.argmax(outputs['objects'], dim=1).item()
            pred_col_id = torch.argmax(outputs['colors'], dim=1).item()
            pred_sce_id = torch.argmax(outputs['scenes'], dim=1).item()

            # 将ID映射为名称
            pred_obj = self.object_names[pred_obj_id]
            pred_col = self.color_names[pred_col_id]
            pred_sce = self.scene_names[pred_sce_id]

            # 格式化为结构化文本
            description = f"SCENE_TYPE: {pred_sce}. KEY_OBJECT: A {pred_col} {pred_obj}."

        return description

    def preprocess_image(self, image_rgb):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image_rgb)