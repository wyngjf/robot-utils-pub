import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 fully_conv=False,
                 remove_avg_pool_layer=False,
                 output_stride=32,
                 additional_blocks=0,
                 multi_grid=(1, 1, 1)):

        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1

        self.remove_avg_pool_layer = remove_avg_pool_layer

        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)

        self.additional_blocks = additional_blocks

        if additional_blocks == 1:
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)

        if additional_blocks == 2:
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)

        if additional_blocks == 3:
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer7 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)

        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
            # In the latest unstable torch 4.0 the tensor.copy_
            # method was changed and doesn't work as it used to be
            # self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    multi_grid=None):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:

                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:

                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride

            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        dilation = multi_grid[0] * self.current_dilation if multi_grid else self.current_dilation

        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            dilation = multi_grid[i] * self.current_dilation if multi_grid else self.current_dilation
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.additional_blocks == 1:
            x = self.layer5(x)

        if self.additional_blocks == 2:
            x = self.layer5(x)
            x = self.layer6(x)

        if self.additional_blocks == 3:
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)

        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)

        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:

        if model.additional_blocks:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

            return model

        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained:

        if model.additional_blocks:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)

            return model

        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:

        if model.additional_blocks:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

            return model

        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:

        if model.additional_blocks:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)

            return model

        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def adjust_input_image_size_for_proper_feature_alignment(input_img_batch, output_stride=8):
    """Resizes the input image to allow proper feature alignment during the
    forward propagation.

    Resizes the input image to a closest multiple of `output_stride` + 1.
    This allows the proper alignment of features.
    To get more details, read here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159

    Parameters
    ----------
    input_img_batch : torch.Tensor
        Tensor containing a single input image of size (1, 3, h, w)

    output_stride : int
        Output stride of the network where the input image batch
        will be fed.

    Returns
    -------
    input_img_batch_new_size : torch.Tensor
        Resized input image batch tensor
    """

    input_spatial_dims = np.asarray( input_img_batch.shape[2:], dtype=np.float )

    # Comments about proper alignment can be found here
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159
    new_spatial_dims = np.ceil(input_spatial_dims / output_stride).astype(np.int) * output_stride + 1

    # Converting the numpy to list, torch.nn.functional.upsample_bilinear accepts
    # size in the list representation.
    new_spatial_dims = list(new_spatial_dims)

    input_img_batch_new_size = nn.functional.upsample_bilinear(input=input_img_batch,
                                                               size=new_spatial_dims)

    return input_img_batch_new_size


class Resnet101_8s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet101_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet101_8s = models.resnet101(fully_conv=True,
                                        pretrained=True,
                                        output_stride=8,
                                        remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet101_8s.fc = nn.Conv2d(resnet101_8s.inplanes, num_classes, 1)

        self.resnet101_8s = resnet101_8s

        self._normal_initialization(self.resnet101_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet101_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet18_8s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet18_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)
        self.resnet18_8s = resnet18_8s
        self._normal_initialization(self.resnet18_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]
        x = self.resnet18_8s(x)
        x = nn.functional.interpolate(x, size=input_spatial_dim, mode="bilinear", align_corners=False)
        return x


class Resnet18_16s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet18_16s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet18_16s = models.resnet18(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_16s.fc = nn.Conv2d(resnet18_16s.inplanes, num_classes, 1)

        self.resnet18_16s = resnet18_16s

        self._normal_initialization(self.resnet18_16s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet18_16s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet18_32s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet18_32s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_32s = models.resnet18(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_32s.fc = nn.Conv2d(resnet18_32s.inplanes, num_classes, 1)

        self.resnet18_32s = resnet18_32s

        self._normal_initialization(self.resnet18_32s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet18_32s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet34_32s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet34_32s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_32s = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_32s.fc = nn.Conv2d(resnet34_32s.inplanes, num_classes, 1)

        self.resnet34_32s = resnet34_32s

        self._normal_initialization(self.resnet34_32s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet34_32s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet34_16s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet34_16s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_16s = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_16s.fc = nn.Conv2d(resnet34_16s.inplanes, num_classes, 1)

        self.resnet34_16s = resnet34_16s

        self._normal_initialization(self.resnet34_16s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet34_16s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet34_8s(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet34_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True, pretrained=True, output_stride=8, remove_avg_pool_layer=True)
        # from torchsummary import summary
        # summary(resnet34_8s.to('cuda'), input_size=(3, 480, 640), batch_size=1)
        # exit()

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)
        self.resnet34_8s = resnet34_8s
        self._normal_initialization(self.resnet34_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]
        x = self.resnet34_8s(x)
        x = nn.functional.interpolate(x, size=input_spatial_dim, mode="bilinear", align_corners=False)
        return x


class Resnet50_32s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet50_32s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = models.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_32s.fc = nn.Conv2d(resnet50_32s.inplanes, num_classes, 1)

        self.resnet50_32s = resnet50_32s

        self._normal_initialization(self.resnet50_32s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet50_32s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet50_16s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet50_16s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = models.resnet50(fully_conv=True,
                                      pretrained=True,
                                      output_stride=16,
                                      remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)

        self.resnet50_8s = resnet50_8s

        self._normal_initialization(self.resnet50_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet50_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet50_8s(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet50_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = resnet50(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        self.resnet50_8s = resnet50_8s
        self._normal_initialization(self.resnet50_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = self.resnet50_8s(x)
        # x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        x = nn.functional.interpolate(x, size=input_spatial_dim, mode="bilinear", align_corners=False)
        return x


class Resnet9_8s(nn.Module):

    # Gets ~ 46 MIOU on Pascal Voc

    def __init__(self, num_classes=1000):
        super(Resnet9_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = models.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)

        self.resnet18_8s = resnet18_8s

        self._normal_initialization(self.resnet18_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet18_8s.conv1(x)
        x = self.resnet18_8s.bn1(x)
        x = self.resnet18_8s.relu(x)
        x = self.resnet18_8s.maxpool(x)

        x = self.resnet18_8s.layer1[0](x)
        x = self.resnet18_8s.layer2[0](x)
        x = self.resnet18_8s.layer3[0](x)
        x = self.resnet18_8s.layer4[0](x)

        x = self.resnet18_8s.fc(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x
