from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class Conv1BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_,out,1)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x

class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=True):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x

class NetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class BsiNet_2(nn.Module):
    def __init__(self, config, head, num_class):
        super().__init__()

        self.conv1 = self._make_conv(3, 32, 64)
        self.conv2 = NetModule(64, 128)
        self.conv3 = NetModule(128, 256)
        self.conv4 = NetModule(256, 512)
        self.conv5 = NetModule(512, 1024)

        self.conv6 = NetModule(1536, 512)#512+256
        self.conv7 = NetModule(768, 256)#256+128
        self.conv8 = NetModule(384, 128)#128+64
        self.conv9 = NetModule(192, 64)#64+32

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.CA = CoordAtt(64,64,4)
        self.head = head(64,num_class)


    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)

        x_6 = self.upsample2(x5)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x9 = self.conv9(torch.cat([x8, x1], 1))
        x = self.CA(x9)

        out = self.head(x)
        return out,x

    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer






