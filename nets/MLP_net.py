import torch
import torch.nn as nn
import dgl
from torch_geometric.nn.norm import GraphNorm


class Seq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

class MLPNet(nn.Module):
    def __init__(self, model_params):
        super().__init__()  
        in_dim     = model_params['in_dim']
        hid_dim    = model_params['hid_dim']
        out_dim    = model_params['out_dim']
        num_layers = model_params['n_layer']
        tail_act   = model_params['tail_act']
        dropout    = model_params['dropout']
        gn         = model_params['gn']
        bias       = model_params['bias']
        if model_params['act']   == "relu":
            activation = nn.ReLU(inplace=True) 
        elif model_params['act'] == "elu":
            activation = nn.ELU(inplace=True) 
        modlist = nn.ModuleList()
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(in_dim, out_dim, bias=bias))
            if tail_act:
                if gn:
                    modlist.append(GraphNorm(out_dim))
                modlist.append(nn.Dropout(p = dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(in_dim, hid_dim, bias=bias))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hid_dim))
                modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(nn.Linear(hid_dim, hid_dim))
            if gn:
                modlist.append(GraphNorm(hid_dim))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hid_dim, out_dim))                        
            if tail_act:
                if gn:
                    modlist.append(GraphNorm(out_dim))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)
    
    def loss(self, scores, targets):
        loss_a = nn.CrossEntropyLoss()(scores, targets)
        loss = loss_a
        return loss


class PMLPNet(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        in_dim     = model_params['in_dim']
        hid_dim    = model_params['hid_dim']
        out_dim    = model_params['out_dim']
        num_layers = model_params['n_layer']
        tail_act   = model_params['tail_act']
        dropout    = model_params['dropout']
        gn         = model_params['gn']
        bias       = model_params['bias']
        if model_params['act']   == "relu":
            activation = nn.ReLU(inplace=True) 
        elif model_params['act'] == "elu":
            activation = nn.ELU(inplace=True) 
        modlist_b = nn.ModuleList()
        modlist_f = nn.ModuleList()
        self.seq_b = None
        self.seq_f = None
        if num_layers == 1:
            modlist_b.append(nn.Linear(in_dim, hid_dim, bias=bias))
            modlist_f.append(nn.Linear(in_dim, hid_dim, bias=bias))
            if tail_act:
                if gn:
                    modlist_b.append(GraphNorm(hid_dim))
                    modlist_f.append(GraphNorm(hid_dim))
                modlist_b.append(nn.Dropout(p = dropout, inplace=True))
                modlist_f.append(nn.Dropout(p = dropout, inplace=True))
                modlist_b.append(activation)
                modlist_f.append(activation)
            self.seq_b = Seq(modlist_b)
            self.seq_f = Seq(modlist_f)
        else:
            modlist_b.append(nn.Linear(in_dim, hid_dim, bias=bias))
            modlist_f.append(nn.Linear(in_dim, hid_dim, bias=bias))
            for _ in range(num_layers - 2):
                if gn:
                    modlist_b.append(GraphNorm(hid_dim))
                    modlist_f.append(GraphNorm(hid_dim))
                modlist_b.append(nn.Dropout(p=dropout, inplace=True))
                modlist_f.append(nn.Dropout(p=dropout, inplace=True))
                modlist_b.append(nn.Linear(hid_dim, hid_dim))
                modlist_f.append(nn.Linear(hid_dim, hid_dim))
            if gn:
                modlist_b.append(GraphNorm(hid_dim))
                modlist_f.append(GraphNorm(hid_dim))
            if dropout > 0:
                modlist_b.append(nn.Dropout(p=dropout, inplace=True))
                modlist_f.append(nn.Dropout(p=dropout, inplace=True))
            modlist_b.append(activation)
            modlist_f.append(activation)
            modlist_b.append(nn.Linear(hid_dim, hid_dim))
            modlist_f.append(nn.Linear(hid_dim, hid_dim))                                 
            if tail_act:
                if gn:
                    modlist_b.append(GraphNorm(hid_dim))
                    modlist_f.append(GraphNorm(hid_dim))
                if dropout > 0:
                    modlist_b.append(nn.Dropout(p=dropout, inplace=True))
                    modlist_f.append(nn.Dropout(p=dropout, inplace=True))
                modlist_b.append(activation)
                modlist_f.append(activation)
            self.seq_b = Seq(modlist_b)
            self.seq_f = Seq(modlist_f)
        
        self.ego_trans = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, 1),
                                  nn.Sigmoid())

        self.out_trans = nn.Linear(hid_dim, out_dim)

    def forward(self, x, y):
        ego_weight = self.ego_trans(x)
        x = self.ego_trans * self.seq_b(x) + (1-self.ego_trans) * self.seq_f(x)
        out = self.out_trans(x)

        return 