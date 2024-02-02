import torch
import torch.nn as nn

class Regression(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.alpha = nn.Parameter(torch.zeros(1),requires_grad=True)      
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
    
    def forward(self, reg, s, q):
       
        lamda = reg * self.alpha.exp() + 1e-6
        rho = self.beta.exp()
        
        #  s:[way,shot*l,d]       
        # st:[way,d,shot*l] 
        st = s.permute(0, 2, 1)
        # sst:[way,shot*l,shot*l] = [way,shot*l,d] * [way,d,shot*l]     
        sst = s.matmul(st)
        # inv:[way,shot*l,shot*l]
        inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lamda)).inverse()
        # W_bar:[way,d,d] = [way,d,shot*l] * [way,shot*l,shot*l] * [way,shot*l,d]
        W_bar = st.matmul(inv).matmul(s)
        
        # q:[1,way*query*l,d]
        # Q_bar:[way,way*query*l,d] = [1,way*query*l,d] * [way,d,d] 
        Q_bar = q.matmul(W_bar).mul(rho)
        
        return Q_bar
    