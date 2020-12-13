import math
import torch.nn as nn
# from torchvision.models.resnet import BasicBlock
from benchmarks.models.layers import OctConvBn, OctConvBnAct


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        alpha=0.5,
        first_block=False,
        last_block=False,
    ):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = OctConvBnAct(
            inplanes, planes, 3,
            stride=stride, padding=1,
            alpha=alpha if not first_block else (0., alpha), 
            bias=True,
        )
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = OctConvBn(
            planes, planes, 3, padding=1,
            alpha=alpha if not last_block else (alpha, 0.)
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # identity = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        out = self.conv2((x_h, x_l))
        x_h, x_l = out if isinstance(out, tuple) else (out, None)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity_h, identity_l = identity if isinstance(identity, tuple) else (identity, None)

        # print('identity_h:', identity_h.shape)
        # print('identity_l:', identity_l.shape if identity_l is not None else None)

        # out += identity
        # out = self.relu(out)
        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        # return out
        return x_h, x_l


class ResNetSmall(nn.Module):

    def __init__(self, block, layers, num_classes=10, alpha=0.5, yuv=False):
        super(ResNetSmall, self).__init__()
        self.inplanes = 16
        self.alpha = alpha
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.relu = nn.ReLU(inplace=True)
        if yuv:
            self.conv1 = OctConvBnAct(3, 16, 3, stride=1, padding=1, alpha=(2/3, alpha))
        else:
            self.conv1 = OctConvBnAct(3, 16, 3, stride=1, padding=1, alpha=(0., alpha))
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_layer=True)
        # self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, first_layer=False, last_layer=False):
        assert not (first_layer and last_layer), "mutually exclusive options"
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion)
            # )
            if last_layer:
                downsample = nn.Sequential(
                    OctConvBn(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                              alpha=(self.alpha, 0.))
                )
            else:
                downsample = nn.Sequential(
                    OctConvBn(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                              alpha=self.alpha if not first_layer else (0., self.alpha))
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, first_block=first_layer, last_block=last_layer,
                alpha=self.alpha,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes,
                    alpha=self.alpha if not last_layer else 0.,
                    last_block=last_layer
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h, x_l))
        x_h, x_l = self.layer3((x_h, x_l))

        # x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)
        x = self.avg_pool(x_h)
        x = x.flatten(1)
        x = self.fc(x)

        return x


def oct_resnet20_small(**kwargs):
    model = ResNetSmall(BasicBlock, [3, 3, 3], **kwargs)
    return model


def oct_resnet32_small(**kwargs):
    model = ResNetSmall(BasicBlock, [5, 5, 5], **kwargs)
    return model


def oct_resnet44_small(**kwargs):
    model = ResNetSmall(BasicBlock, [7, 7, 7], **kwargs)
    return model


def oct_resnet56_small(**kwargs):
    model = ResNetSmall(BasicBlock, [9, 9, 9], **kwargs)
    return model


if __name__ == "__main__":
    import torch
    # x = torch.rand(2, 3, 32, 32)
    x = (
        torch.rand(2, 1, 32, 32),
        torch.rand(2, 2, 16, 16)
    )
    model = oct_resnet20_small(alpha=0.5, yuv=True)
    y = model(x)
    print('y:', y.shape)
