import torch
import torch.nn as nn
import torch.nn.functional as F


from tensorboardX import SummaryWriter
from torchviz import make_dot


class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.ReLU(),
        )

        self.conv_13 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=13, dilation=13),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x =  self.conv_in(x)
        x_1 = self.conv_1(x)
        x_7 = self.conv_7(x)
        x_13 = self.conv_13(x)
        x_c17 = torch.cat((x_1, x_7),1)
        x_c1713 = torch.cat((x_c17, x_13),1)
        x_out = self.conv_out(x_c1713)
        return x_out




class Net_aspp_block(nn.Module):
    def __init__(self):
        super(Net_aspp_block, self).__init__()
        self.aspp = ASPP()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1
            nn.ReLU()
        )

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        x = input
        for i in range(1):  # 迭代次数，不改变网络参数量

            x = torch.cat((input, x), 1)
            x = self.aspp(x)
            x = self.conv0(x)
            x = F.relu(self.res_conv1(x) + x)
            x = F.relu(self.res_conv2(x) + x)
            x = F.relu(self.res_conv3(x) + x)
            x = F.relu(self.res_conv4(x) + x)
            x = F.relu(self.res_conv5(x) + x)
            x = self.conv(x)
            x = x + input

        return x

class Net_aspp(nn.Module):
    def __init__(self):
        super(Net_aspp, self).__init__()
        self.block_1 = Net_aspp_block()
        self.block_2 = Net_aspp_block()
        self.block_3 = Net_aspp_block()


    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        return x


if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))
    net = Net_aspp()
    out = net(x)
    print(net)
    #g = make_dot(out)
    #g.render('espnet_model', view=False)

    with SummaryWriter(comment='resnet') as w:
        w.add_graph(net, x)
