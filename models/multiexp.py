from typing import List, Dict, Union
from utils.general import Log
import torch
from torch import nn
from tricks.resnet import ResNet, BasicBlock, Bottleneck
from tricks.senet import SENet, SEResNeXtBottleneck, SEResNetBottleneck, SEBottleneck

try:  # try loading from URL
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ModelBuilder(nn.Module):
    in_feat_size = 2048

    def __init__(self, exp_config: Dict[str, Union[str, tuple, bool]], num_classes=1000):
        # available key: dataset, backbone, input_size, loss,
        # bnn_neck, warmup, random_erasing, label_smoothing, last_stride
        super(ModelBuilder, self).__init__()
        last_stride: int = exp_config['last_stride']
        self.model_name: str = exp_config['backbone']
        pretrained: Union[bool, str] = exp_config['pretrained']
        self.neck: bool = exp_config['bnn_neck']
        self.neck_feat: str = exp_config['neck_feat']
        self.num_classes = num_classes

        self.backbone: Union[None, nn.Module] = None

        if self.model_name == 'resnet18':
            self.in_feat_size = 512
            self.backbone = ResNet(last_stride=last_stride,
                                   block=BasicBlock,
                                   layers=[2, 2, 2, 2])
        elif self.model_name == 'resnet34':
            self.in_feat_size = 512
            self.backbone = ResNet(last_stride=last_stride,
                                   block=BasicBlock,
                                   layers=[3, 4, 6, 3])
        elif self.model_name == 'resnet50':
            self.backbone = ResNet(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[2, 2, 2, 2])
        elif self.model_name == 'resnet101':
            self.backbone = ResNet(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[3, 4, 23, 3])
        elif self.model_name == 'resnet152':
            self.backbone = ResNet(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[3, 8, 36, 3])
        elif self.model_name == 'se_resnet50':
            self.backbone = SENet(block=SEResNetBottleneck,
                                  layers=[3, 4, 6, 3],
                                  groups=1,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
        elif self.model_name == 'se_resnet101':
            self.backbone = SENet(block=SEResNetBottleneck,
                                  layers=[3, 4, 23, 3],
                                  groups=1,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
        elif self.model_name == 'se_resnet152':
            self.backbone = SENet(block=SEResNetBottleneck,
                                  layers=[3, 8, 36, 3],
                                  groups=1,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
        elif self.model_name == 'se_resnext50':
            self.backbone = SENet(block=SEResNeXtBottleneck,
                                  layers=[3, 4, 6, 3],
                                  groups=32,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
        elif self.model_name == 'se_resnext101':
            self.backbone = SENet(block=SEResNeXtBottleneck,
                                  layers=[3, 4, 23, 3],
                                  groups=32,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
        elif self.model_name == 'senet154':
            self.backbone = SENet(block=SEBottleneck,
                                  layers=[3, 8, 36, 3],
                                  groups=64,
                                  reduction=16,
                                  dropout_p=0.2,
                                  last_stride=last_stride)

        elif self.model_name == 'CNN-GA':
            pass  # CNN-GA method goes to cnn_ga_exp.py

        if isinstance(pretrained, str):  # if load pretrained model
            self._load_pretrain(pretrained)
        else:
            assert not pretrained, "Pretrained can only be model path or false!"

        # Network Part
        self.gap = nn.AdaptiveAvgPool2d(1)  # prepare to flatten

        if self.neck:
            self.bottleneck = nn.BatchNorm1d(self.in_feat_size)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(self.in_feat_size, self.num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        else:
            self.classifier = nn.Linear(self.in_feat_size, self.num_classes)

    def forward(self, x, add_logits=True):
        # add_logits is True when training, False when generating feature
        # x: (N, 3, *sz)
        g_feat = self.gap(self.backbone(x))  # (N, feat_size, 1, 1)
        g_feat = g_feat.view(g_feat.shape[0], -1)  # flatten

        if self.neck:
            feat = self.bottleneck(g_feat)  # if bnn_neck
        else:
            feat = g_feat  # if no bnn_neck, feat remains identical

        if add_logits:
            logits = self.classifier(feat)
            return logits, g_feat
        else:
            return feat if self.neck_feat == 'after' else g_feat

    def forward_once(self, x):
        """
        `forward_once` is necessary for evaluating MOT-related metrics
        """
        return self.forward(x, add_logits=False)

    def _load_pretrain(self, pretrain_model_path: str):
        # Done: Load ImageNet from torchvision
        if pretrain_model_path == 'imagenet':
            if isinstance(self.backbone, ResNet):  # if resnet, trying to load from torchvision
                state_dict = load_state_dict_from_url(ResNet.model_urls[self.model_name])
                self.backbone.load_param(state_dict)
                return
            elif isinstance(self.backbone, SENet):
                import torch.utils.model_zoo as model_zoo
                from tricks.senet import pretrained_settings as senet_settings
                state_dict = model_zoo.load_url(senet_settings[self.model_name]['imagenet']['url'])
                self.backbone.load_param(state_dict)
                return
            else:
                raise NotImplementedError(
                    "Sorry, ImageNet pretrained model for %s not supported yet!" % self.model_name)
        Log.warn("You are trying to load non-imagenet pretrained weight, use at your own risk!")
        self.backbone.load_param(pretrain_model_path)
