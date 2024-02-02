import torch
import torch.nn as nn
from torchinfo import summary

class Extractor(nn.Module):

    def __init__(self):
        super().__init__()

        # [1,5000] → [64,1250]
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64,
                      kernel_size=7, stride=1, 
                      padding='same', bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, 
                      kernel_size=7, stride=1,
                      padding='same', bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=7, stride=4, padding=3),
        )

        # [64,1250] → [128,250]
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=7, stride=1, 
                      padding='same', bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128,
                      kernel_size=7, stride=1, 
                      padding='same', bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=9, stride=5, padding=4),
        )

        # [128,250] → [256,50]
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=7, stride=1,
                      padding='same', bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=7, stride=1, 
                      padding='same', bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=9, stride=5, padding=4),
        )

        # [256,50] → [512,10]
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512,
                      kernel_size=7, stride=1, 
                      padding='same', bias=False),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512,
                      kernel_size=7, stride=1, 
                      padding='same', bias=False),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=9, stride=5, padding=4),
        )

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out

if __name__=='__main__':
    inp = torch.randn((10, 1, 5000))
    net = Extractor()
    out = net(inp)
    print(out.shape)
    summary(net, (10, 1, 5000))
