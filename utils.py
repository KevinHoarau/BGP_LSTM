import pandas as pd
from sklearn.preprocessing import StandardScaler

def loadData(folder):
    data = {}
    
    names = ['bitcanal_3', 'petersburg_unused_1', 'petersburg_unused_2', 'petersburg_1', 'torg_1', 'torg_3', 
             'backconnect_3', 'backconnect_5', 'backconnect_6', 'france_1', 'enzu_1', 'defcon_1', 'carlson_1', 'facebook_1'] 
    
    for name in names:
        d = pd.read_json(folder+"/"+name+"/transform/GraphFeatures/GraphFeatures_2.json")
        d = pd.DataFrame(StandardScaler().fit_transform(d), columns=d.columns)

        data[name] = d
            
    return(data)

import torch
from torch import nn
import torch.nn.functional as F

class RNNClassifier(nn.Module):
    def __init__(
            self,
            input_dim=13,
            rec_layer_type='lstm',
            num_units=16,
            num_layers=2,
            dropout=0,
            seed=0
    ):
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        super().__init__()
        
        self.input_dim = input_dim
        self.num_units = num_units
        self.num_layers = num_layers

    
        self.rec = nn.LSTM(
            self.input_dim,
            self.num_units,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True
        )

        self.drop = nn.Dropout(dropout)
        self.output = nn.Linear(self.num_units, 2)
        
    def forward(self, X):
        
        X = X.float()
        _, (rec_out, _) = self.rec(X)
        rec_out = rec_out[-1]

        drop = self.drop(rec_out)
        out = F.softmax(self.output(drop), dim=-1)
        return out