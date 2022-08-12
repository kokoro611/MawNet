import torch
from torch import nn
from net.basic_block import BaseConv
from tensorboardX import SummaryWriter
class Network_class(nn.Module):
    def __init__(self, class_num):
        super(Network_class, self).__init__()
        self.class_num = class_num

        self.Conv_1 = BaseConv(in_channels=3, out_channels=16, ksize=11, stride=4)
        self.Maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Conv_2 = BaseConv(in_channels=16, out_channels=32, ksize=5, stride=2)
        self.Maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Conv_3 = BaseConv(in_channels=32, out_channels=64, ksize=3, stride=1)
        self.Maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self.Globepool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, class_num),
        )
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Conv_1(x)
        x = self.Maxpool_1(x)
        x = self.Conv_2(x)
        x = self.Maxpool_2(x)
        x = self.Conv_3(x)
        x = self.Maxpool_3(x)
        x = self.Globepool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.Sigmoid(x)
        return x

from torchsummary import summary

if __name__ == "__main__":
    net = Network(2)
    x = torch.rand((1, 3, 224, 224))
    y = net(x)
    print(net)
    print(y)
    with SummaryWriter(comment='net') as w:
        w.add_graph(net, x)
    summary(net.cuda(), input_size=(3, 224, 224))
