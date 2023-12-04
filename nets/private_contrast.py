import torch.nn as nn
import torch

class PrivateSpace(nn.Module):
    def __init__(self, config, g):
        super(PrivateSpace, self).__init__()
        self.config           = config
        labels                = g.ndata['label']
        feat_dim              = g.ndata['feat'].shape[1]
        self.space_generator  = nn.ModuleList()
        self.n_layers         = config['private_space']['n_layers']
        in_dim                = feat_dim
        hid_dim               = config['private_space']['hid_dim']
        for lyr in range(self.n_layers):
            if lyr == self.n_layers - 1:
                hid_dim  = 1
            self.space_generator.append(nn.Linear(in_features  = in_dim,
                                                  out_features = hid_dim))
            in_dim = hid_dim



    def forward(self, cls_spec_avg_feats):
        sigmas = cls_spec_avg_feats
        for generator_layer in self.space_generator:
            sigmas = generator_layer(sigmas)
        assert sigmas.shape[0] == self.config['num_cls']
        return sigmas