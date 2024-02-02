import torch
import torch.nn as nn
from torchinfo import summary

class SELayer(nn.Module):
    
    def __init__(self, channel=512, reduction=16):
        super(SELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c = x.shape[1] 
        y = self.avg_pool(x).view(1,c)    
        y = self.fc(y).view(1,c,1)
        return y

class TMU(nn.Module):

    def __init__(self, way):
        super().__init__()
             
        self.conv_activate = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=64, 
                      kernel_size=1, stride=1, 
                      padding=0, bias=False), 
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=64, out_channels=64,  
                      kernel_size=1, stride=1, 
                      padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), 
            
            nn.Conv1d(in_channels=64, out_channels=512,  
                      kernel_size=1, stride=1, 
                      padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        
        self.conv_attention = nn.Sequential(
            nn.Conv1d(in_channels=512*way, out_channels=512, 
                      kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            SELayer(512,16)
        )

    def forward(self, support_map, query_map):
        
        # spt:[100,5,512,10]
        way, shot, d, l = support_map.shape
        # prt:[100,512,10]
        prt = support_map.mean(1)
        
        # m1:[1,100,512,10] → [100,100,512,10]
        m1 = prt.reshape(1,way,d,l).repeat(way,1,1,1)
        # m2:[100,1,512,10] → [100,100,512,10]
        m2 = prt.reshape(way,1,d,l).repeat(1,way,1,1)
        # diff_info:[100,100,512,10] → [10000,512,10]
        diff_info = (m1-m2).reshape(way*way,d,l) 
        
        # [10000,512,10] → [10000,64,10] → [10000,64,10] → [10000,512,10]  
        diff_info = self.conv_activate(diff_info)
        # [10000,512,10] → [100,100,512,10] 
        diff_info = diff_info.reshape(way,way,d,l)
        # [100,100,512,10] → [100,512,10] → [1,51200,10] 
        diff_info = diff_info.mean(1).reshape(1,way*d,l)
        
        # [1,51200,10] → [1,512,10] → [1,1,512,1]
        channel_attention = self.conv_attention(diff_info).unsqueeze(1)
        
        new_support_map = channel_attention*support_map
        new_query_map = channel_attention*query_map 
             
        return new_support_map,new_query_map

if __name__=='__main__':
    support_map = torch.randn((100, 5, 512, 10))
    query_map = torch.randn((1, 1500, 512, 10))
    net = TMU(100)
    new_support_map,new_query_map = net(support_map,query_map)
    print(new_support_map.shape)
    print(new_query_map.shape)