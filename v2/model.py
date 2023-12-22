import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm 
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torchvision.models as models


class BaseModel(nn.Module):
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

class MyModel(nn.Module):
    def __init__(self, num_age_classes,num_gender_classes,num_mask_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        resnext50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        # Remove the last fully connected layer of ResNeXt-50
        self.resnext = nn.Sequential(*list(resnext50.children())[:-1])
        
        n=resnext50.fc.in_features
        self.fc_age= nn.Linear(n, num_age_classes)
        self.fc_gender= nn.Linear(n, num_gender_classes)
        self.fc_mask= nn.Linear(n, num_mask_classes)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.resnext(x)
        age=self.fc_age(x)
        gender=self.fc_gender(x)
        mask=self.fc_mask(x)
        
        return age,gender,mask

## Efficient 
class MyModel_efficient_v2_s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        efficient = models.efficientnet_v2_s(pretrained=True)
        self.features = efficient.features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2,inplace = True),
            nn.Linear(in_features=1280, out_features=18, bias=True)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

class MyModel_efficient_v2_l(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        efficient = models.efficientnet_v2_l(pretrained=True)
        self.features = efficient.features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4,inplace = True),
            nn.Linear(in_features=1280, out_features=18, bias=True)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
    

## ConvNext
class ConvNextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        self.convnext = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        self.convnext.classifier = nn.Sequential(
            nn.LayerNorm((768,1,1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            # nn.Linear(in_features=768, out_features=1024, bias=True),
            # nn.BatchNorm1d(1024),s
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=1024, out_features=num_classes, bias=True)
            nn.Linear(in_features=768, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.convnext(x)
        
        return x

class ConvNextModel_3fc(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        self.convnext = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        self.convnext.classifier = nn.Sequential(
            nn.LayerNorm((768,1,1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=1024, bias=True),
            nn.Dropout(0.2),
        )
        
        self.classifier_age = nn.Linear(in_features=1024, out_features=3, bias=True)
        self.classifier_mask = nn.Linear(in_features=1024, out_features=3, bias=True)
        self.classifier_gender = nn.Linear(in_features=1024, out_features=2, bias=True)

    def forward(self, x):

        x=self.convnext(x)
        age=self.classifier_age(x)
        mask=self.classifier_mask(x)
        gender=self.classifier_gender(x)
        
        return age,mask,gender

class ConvNextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        self.convnext = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        self.convnext.classifier = nn.Sequential(
            nn.LayerNorm((768,1,1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            # nn.Linear(in_features=768, out_features=1024, bias=True),
            # nn.BatchNorm1d(1024),s
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=1024, out_features=num_classes, bias=True)
            nn.Linear(in_features=768, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x=self.convnext(x)
        return x

class ConvNext_timm(nn.Module):
    def __init__(self, num_classes, pretrained=True):

        super(ConvNext_timm, self).__init__()

        self.model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=pretrained)
        # if pretrained:
        #     self.model.load_state_dict(torch.load("./level1-imageclassification-cv-01/v2/input/convnext_small_22k_1k_384.pth"))
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        
        return x


## VIT16
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