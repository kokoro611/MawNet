import torch
import torch.nn as nn
import torch.nn.functional as F
#from tensorboardX import SummaryWriter
#from torchviz import make_dot

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

        self.Conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )


        # self.Conv_r1=nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1),
        #     nn.ReLU()
        # )

        # self.Conv_r3=nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=2, dilation=2),
        #     nn.ReLU()
        # )

        self.Conv_r5=nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU()
        )

        self.Conv_r0 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU()
        )

        self.Conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU()
        )

    def forward(self,input):
        x = self.Conv_0(input)
        x1 = self.Conv_r0(x)
        # x2 = self.Conv_r1(x)
        # x3 = self.Conv_r3(x)
        x4 = self.Conv_r5(x)

        # x_12 = torch.cat((x1, x2), 1)
        # x_34 = torch.cat((x3, x4), 1)
        # x_all = torch.cat((x_12,x_34),1)

        # x_34 = torch.cat((x3, x4), 1)
        x_all = torch.cat((x1,x4),1)

        x_out = self.Conv_1(x_all)

        return x_out


if __name__ == "__main__":
    x = torch.rand((1, 6, 224, 224))
    net = ASPP()
    out = net(x)
    print(net)

    '''
    g = make_dot(out)
    g.render('espnet_model', view=False)

    with SummaryWriter(comment='resnet') as w:
        w.add_graph(net, x)
    '''
