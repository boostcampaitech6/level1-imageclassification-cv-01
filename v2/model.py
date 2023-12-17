import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from transformers import AutoImageProcessor, ConvNextForImageClassification
import timm # needed library
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Callable, List, Optional, Sequence, Tuple, Union

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


# Custom Model Template
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


class ConvNextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()


        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        self.convnext = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        self.convnext.classifier = nn.Sequential(
            nn.LayerNorm((768,1,1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=num_classes, bias=True)
        )
                        


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
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




#convnext_model = ConvNextModel(3)#torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')


# Print the model architecture
#print(convnext_model.convnext)
# print(convnext_model)