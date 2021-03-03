import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1(in_planes, out_planes, stride=2):
    '''
    The first convolution layer of ResNet
    :param in_planes: input channel
    :param out_planes: out channel
    :param stride:
    :return:
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class RBCA(nn.Module):
    def __init__(self, in_planes, out_planes, text_feature_shape, stride=1, expansion=4, downsampling=False):
        '''
        :param in_planes:
        :param out_planes:
        :param text_feature_shape:
        :param stride:
        :param expansion:
        :param downsampling:
        '''
        super(RBCA, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.text_feature_shape = text_feature_shape
        self.text_feature = None
        self.selForAtt = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_planes // 2),
            nn.ReLU(inplace=True)
        )
        self.selForOri = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_planes // 2),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes*expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes * expansion)
        )
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=out_planes*expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_planes*expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def fuse(self, x):
        text_vec = self.text_feature
        channel = text_vec.size(1)
        batch_size = text_vec.size(0)
        text_vec = text_vec.reshape(batch_size, channel, 1, 1)  # TODO: make sure the order of the size is right
        temp_x = torch.mul(x, text_vec)
        temp_x = F.adaptive_avg_pool2d(temp_x, (1, 1))  # squeeze to batch_size*channel*1*1
        temp_x = torch.sigmoid(temp_x)
        x = torch.mul(x, temp_x)
        return x

    def forward(self, x, text_feature):
        residual = x
        self.text_feature = text_feature
        x1 = self.selForAtt(x)
        x2 = self.selForOri(x)

        x1 = self.fuse(x1)
        x = torch.cat((x1, x2), 1)

        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BottleneckNoAtt(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(BottleneckNoAtt, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class FusedResNet50(nn.Module):
    def __init__(self, cp, text_info_dim, t_feature_shape, num_classes=2, expansion=4):
        super(FusedResNet50, self).__init__()
        self.expansion = expansion
        self.cp = cp
        self.num_classes = num_classes
        self.t_feature_shape = t_feature_shape

        self.inc_channel64 = nn.Sequential(
            nn.Conv1d(in_channels=t_feature_shape[1], out_channels=64, kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.inc_channel256 = nn.Sequential(
            nn.Conv1d(in_channels=t_feature_shape[1], out_channels=256, kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.inc_channel512 = nn.Sequential(
            nn.Conv1d(in_channels=t_feature_shape[1], out_channels=512, kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.inc_channel1024 = nn.Sequential(
            nn.Conv1d(in_channels=t_feature_shape[1], out_channels=t_feature_shape[1] // 2, kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=t_feature_shape[1] // 2),
            nn.ReLU(inplace=True)
        )

        self.inc_channel2048 = nn.Sequential(
            nn.Conv1d(in_channels=t_feature_shape[1], out_channels=t_feature_shape[1], kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=t_feature_shape[1]),
            nn.ReLU(inplace=True)
        )

        self.conv1 = conv1(in_planes=1, out_planes=64)

        # Layer1: 3 x RBCA
        self.bnk1_1 = BottleneckNoAtt(in_places=64, places=64, stride=1, downsampling=True)
        self.bnk1_2 = BottleneckNoAtt(in_places=256, places=64, stride=1)
        self.bnk1_3 = BottleneckNoAtt(in_places=256, places=64, stride=1)

        # Layer2: 4 x RBCA
        shape2 = list(t_feature_shape)
        shape2[1] = shape2[1] * expansion
        self.bnk2_1 = RBCA(in_planes=256, out_planes=128, text_feature_shape=list(t_feature_shape), stride=2,
                                 downsampling=True)
        self.bnk2_2 = RBCA(in_planes=512, out_planes=128, text_feature_shape=shape2, stride=1)
        self.bnk2_3 = RBCA(in_planes=512, out_planes=128, text_feature_shape=shape2, stride=1)
        self.bnk2_4 = RBCA(in_planes=512, out_planes=128, text_feature_shape=shape2, stride=1)

        # Layer3: 6 x RBCA
        shape3 = list(t_feature_shape)
        shape3[1] = shape3[1] * expansion
        self.bnk3_1 = RBCA(in_planes=512, out_planes=256, text_feature_shape=list(t_feature_shape), stride=2,
                                 downsampling=True)
        self.bnk3_2 = RBCA(in_planes=1024, out_planes=256, text_feature_shape=shape3, stride=1)
        self.bnk3_3 = RBCA(in_planes=1024, out_planes=256, text_feature_shape=shape3, stride=1)
        self.bnk3_4 = RBCA(in_planes=1024, out_planes=256, text_feature_shape=shape3, stride=1)
        self.bnk3_5 = RBCA(in_planes=1024, out_planes=256, text_feature_shape=shape3, stride=1)
        self.bnk3_6 = RBCA(in_planes=1024, out_planes=256, text_feature_shape=shape3, stride=1)

        # Layer4: 3 x RBCA
        shape4 = list(t_feature_shape)
        shape4[1] = shape4[1] * expansion
        self.bnk4_1 = RBCA(in_planes=1024, out_planes=512, text_feature_shape=list(t_feature_shape), stride=2,
                                 downsampling=True)
        self.bnk4_2 = RBCA(in_planes=2048, out_planes=512, text_feature_shape=shape4, stride=1)
        self.bnk4_3 = RBCA(in_planes=2048, out_planes=512, text_feature_shape=shape4, stride=1)

        # pooling
        self.avgpool = nn.AvgPool2d(8, stride=1)

        # fuse the text info
        self.text_interface = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.bn = nn.BatchNorm1d(num_features=1)
        self.relu = nn.ReLU()

        # classification
        self.fc = nn.Linear(2048 + text_info_dim, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Conv1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, text):
        t_feature = self.cp(text)
        t_feature256 = self.inc_channel256(t_feature)
        t_feature512 = self.inc_channel512(t_feature)
        t_feature1024 = self.inc_channel1024(t_feature)  # 1024
        t_feature2048 = self.inc_channel2048(t_feature)  # 2048
        x = self.conv1(x)
        # layer 1
        x = self.bnk1_1(x)
        x = self.bnk1_2(x)
        x = self.bnk1_3(x)
        # layer 2
        x = self.bnk2_1(x, t_feature256)
        x = self.bnk2_2(x, t_feature512)
        x = self.bnk2_3(x, t_feature512)
        x = self.bnk2_4(x, t_feature512)
        # layer 3
        x = self.bnk3_1(x, t_feature512)
        x = self.bnk3_2(x, t_feature1024)
        x = self.bnk3_3(x, t_feature1024)
        x = self.bnk3_4(x, t_feature1024)
        x = self.bnk3_5(x, t_feature1024)
        x = self.bnk3_6(x, t_feature1024)
        # layer 4
        x = self.bnk4_1(x, t_feature1024)
        x = self.bnk4_2(x, t_feature2048)
        x = self.bnk4_3(x, t_feature2048)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # TODO concat t_feature:
        t_feature = t_feature.view(t_feature.size(0), -1)
        x = torch.cat((x, t_feature), 1)

        feature = x
        x = self.fc(x)
        return feature, x


class ClinicalPath(nn.Module):
    def __init__(self, input_shape):
        super(ClinicalPath, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 1024),
            nn.Linear(1024, 1024)
        )

    def forward(self, x):
        out = self.net(x)
        out = out.reshape(out.size(0), out.size(1), 1)
        return out
