import dgl
from nets.backbones import GCNLayer, GATLayer, SAGELayer
import torch.nn as nn
import torch
import torch.nn.functional as F

class GImbNet(nn.Module):
    def __init__(self, config, in_dim, out_dim, train_cls_masks = None, private_space = None, module_name = None):
        super(GImbNet, self).__init__()
        self.num_layers = config['n_layer']
        self.in_dim          = in_dim
        self.train_cls_masks = train_cls_masks
        self.hid_dim         = config['hid_dim']
        self.out_dim         = out_dim
        self.n_cls           = config['num_cls']
        self.private_space   = private_space
        self.dropout         = config['dropout']
        self.bn              = config['gn']
        self.residual        = config['residual']
        self.out_mlp         = config['layer_mlp']
        self.act             = getattr(F, config['act'])
        self.backbone        = config['backbone']
        self.in_feat_dropout = nn.Dropout(config['in_feat_dropout'])
        self.proj            = config['proj']
        hid_dim              = self.hid_dim
        self.layerwise_embs  = []
        self.layers = nn.ModuleList()
        if self.proj:
            temp_out       = out_dim
            out_dim        = self.hid_dim
            self.proj_head = nn.Conv1d(in_channels = hid_dim, out_channels = temp_out, kernel_size = 1)
        if self.num_layers == 1:
            hid_dim = out_dim

            
        if self.backbone  == 'gcn':
            self.layers             .append(GCNLayer(in_dim     = in_dim, 
                                                     out_dim    = hid_dim, 
                                                     activation = self.act if self.num_layers > 1 else None, 
                                                     dropout    = self.dropout if self.num_layers > 1 else 0, 
                                                     bn         = self.bn  if self.num_layers > 1 else None, 
                                                     residual   = self.residual, 
                                                     out_mlp    = self.out_mlp))
            if self.num_layers >1:
                self.layers.extend([GCNLayer(in_dim     = hid_dim,
                                             out_dim    = hid_dim,
                                             activation = self.act,
                                             dropout    = self.dropout, 
                                             bn         = self.bn, 
                                             residual   = self.residual,
                                             out_mlp    = self.out_mlp ) for _ in range(self.num_layers-2)])
                
                self.layers.append(GCNLayer(in_dim      = hid_dim,
                                            out_dim     = out_dim,
                                            activation  = None,
                                            dropout     = 0, 
                                            bn          = None, 
                                            residual    = None,
                                            out_mlp     = self.out_mlp ) )
        elif self.backbone == 'gat':
            self.layers           .append([GATLayer(in_dim     = in_dim, 
                                                    out_dim    = hid_dim, 
                                                    num_heads  = config[self.backbone]['num_heads'],
                                                    dropout    = self.dropout if self.num_layers > 1 else 0,
                                                    bn         = self.bn if self.num_layers > 1 else None,
                                                    residual   = self.residual)])
            
            if self.num_layers >1:
                self.layers.extend([GATLayer(in_dim     = hid_dim * config[self.backbone]['num_heads'],
                                             out_dim    = hid_dim,
                                             num_heads  = config[self.backbone]['num_heads'],
                                             dropout    = self.dropout,
                                             bn         = self.bn,
                                             residual   = self.residual) for _ in range(self.num_layers-2)])
                self.layers.append(GATLayer(in_dim      = hid_dim * config[self.backbone]['num_heads'], 
                                            out_dim     = out_dim,
                                            num_heads   = 1,
                                            activation  = None,
                                            dropout     = self.dropout, 
                                            bn          = None, 
                                            residual    = None))

        elif self.backbone == 'sage':
            for i in range(self.num_layers):
                if  i == self.num_layers - 1:
                    hid_dim  = out_dim
                self.layers.append(SAGELayer(in_feats        = in_dim,
                                             out_feats       = hid_dim,
                                             aggregator_type = config[self.backbone]['agg'],
                                             feat_drop       = self.dropout if i != self.num_layers - 1 else 0,
                                             bias            = True,
                                             norm            = nn.BatchNorm1d(hid_dim) if self.bn and self.num_layers < self.num_layers - 1 else None,
                                             activation      = self.act if self.num_layers < self.num_layers - 1 else None))
                in_dim = hid_dim


    def forward(self, g, feat, cls_spec_avg_feats):
        self.sigmas = self.private_space(cls_spec_avg_feats)  # shape [7,1]
        h = self.in_feat_dropout(feat)
        layerwise_embs = []
        for i, conv in enumerate(self.layers):
            h = conv(g, h, last_layer = False) if i < self.num_layers-1 else conv(g, h, last_layer = True)
            layerwise_embs.append(h)
        if self.proj:
            if self.dropout > 0:
                h = F.dropout(h, self.dropout, training=self.training)
            h = h.unsqueeze(0)
            h = h.permute((0,2,1))
            h = self.proj_head(h)
            h = h.permute((0,2,1)).squeeze()
        return h, layerwise_embs
    
    def sim(self, h):
        z = F.normalize(h)
        return torch.mm(z, z.t())

    

    def contrast_loss(self, layerwise_embs):
        contrast_loss = 0
        for h in layerwise_embs:
            for i in range(self.n_cls):
                cls_mask       = self.train_cls_masks[i]
                sigma          = self.sigmas[i]
                h_cls          = h[cls_mask]
                f              = lambda x: torch.exp(x / sigma)
                intra_cls_sim  = f(self.sim(h_cls))  # nxn
                cls_loss = -torch.log(intra_cls_sim.sum(1)).mean()
                contrast_loss += cls_loss
        return contrast_loss

    def get_sigma(self):
        sigmas = None
        with torch.no_grad():
            sigmas = self.sigmas.detach().cpu().numpy()
        return sigmas

    def loss(self, pred, label, criterion = nn.CrossEntropyLoss()):
        loss = criterion(pred, label)
        return loss
    
    