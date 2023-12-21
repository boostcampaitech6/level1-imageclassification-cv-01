import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm # needed library
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform



class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class ConvNextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()


        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        self.convnext = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        self.convnext.classifier = nn.Sequential(
            nn.LayerNorm((768,1,1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            # nn.Linear(in_features=768, out_features=1024, bias=True),
            #nn.Linear(in_features=1024, out_features=num_classes, bias=True)
            nn.Linear(in_features=768, out_features=num_classes, bias=True)
        )


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.convnext(x)
        
        return x
    

class VITmodel(nn.Module): 
    def __init__(self, num_classes):
        super().__init__()

        # Create the ViT model
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Get the number of features in the ViT model's final layer
        num_features = self.model.head.in_features

        # Modify the final layer for classification *** 중요
        self.model.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the ViT model
        x = self.model(x)
        return x