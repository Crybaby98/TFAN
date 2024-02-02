import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import Extractor
from models.feature_modulation import TMU
from models.ridge_regression import Regression

class TFAN(nn.Module):

    def __init__(self,way):
        super().__init__()

        self.extractor = Extractor()
        self.modulation = TMU(way)
        self.regression = Regression()
    
    def get_distance(self, inp, way, shot, query):
        #===================================================================================
        # 100way-5shot-15query
        #===================================================================================
        #         inp:[way*(shot+query),1,L]                  [2000,1,5000]
        # feature_map:[way*(shot+query),d,l]                  [2000,512,10]                 
        # support_map:[way*shot,c,l] → [way,shot,c,l]         [500,512,10] → [100,5,512,10]
        #   query_map:[way*query,c,l] → [1,way*query,c,l]     [1500,512,10] → [1,1500,512,10]
        #===================================================================================

        # Get feature map of a task
        feature_map = self.extractor(inp)
        _, d, l = feature_map.shape
        
        # divide to support and query
        support_map = feature_map[:way * shot].view(way, shot, d, l)
        query_map = feature_map[way * shot:].view(1, way*query, d, l)
        
        # task-adaptive modulation
        support_map, query_map = self.modulation(support_map,query_map)

        # reshape across locations and instances to construct support latent space
        support_map = support_map.permute(0, 1, 3, 2).contiguous().view(way, shot*l, d)
        query_map = query_map.permute(0, 1, 3, 2).contiguous().view(1, way*query*l, d)
           
        # ridge regression
        reg = shot * l / d
        Q_bar = self.regression(reg, support_map, query_map)

        # compute distance 
        distance = (Q_bar - query_map).pow(2).sum(2).permute(1, 0) 
        distance = distance.neg().view(way * query, l, way).mean(1)

        return distance

    # meta-train
    def forward(self, inp, way, shot, query): 
        distance = self.get_distance(inp, way, shot, query)
        logits = distance * self.regression.gamma
        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction
    
    # meta-val or meta-test
    def val_or_test(self, inp, way, shot, query):
        distance = self.get_distance(inp, way, shot, query)
        _, max_index = torch.max(distance, 1)
        return max_index