import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv, SAGEConv, GraphConv

class GNN(nn.Module):
    def __init__(self,config, in_dim, out_dim):
        super(GNN, self).__init__()
        self.gnn_layers = nn.ModuleList()
        self.backbone = config['backbone']
        self.hid_dim         = config['hid_dim']
        self.out_dim         = out_dim
        self.n_cls           = config['num_cls']
        self.dropout         = config['dropout']
        self.num_layers      = config['n_layer']
        self.tail_act        = getattr(F, config[self.backbone]['tail_act'], None)
        hid_dim = out_dim if self.num_layers == 1 else self.hid_dim
        self.act = self.tail_act if self.num_layers == 1 else getattr(F, config[self.backbone]['act'], None)
        if self.backbone == 'gcn':
            self.gnn_layers.append(GraphConv(in_feats   = in_dim,
                                             out_feats  = hid_dim, 
                                             activation = self.act))
            if self.num_layers >1:
                self.gnn_layers.extend([GraphConv(in_feats   = hid_dim,
                                                  out_feats  = hid_dim,
                                                  activation = self.act) for _ in range(self.num_layers-2)])
                self.gnn_layers.append(GraphConv(in_feats=hid_dim,
                                                 out_feats=out_dim,
                                                 activation=self.tail_act))
        elif self.backbone == 'gat':
            self.gnn_layers.append(GATConv(in_feats  = in_dim,
                                           out_feats = hid_dim,
                                           num_heads = config[self.backbone]['num_heads'],
                                           feat_drop = config[self.backbone]['feat_drop'],
                                           attn_drop = config[self.backbone]['attn_drop'],
                                           residual  = False,
                                           activation= self.act))
            if self.num_layers > 1:
                self.gnn_layers.extend([GATConv(in_feats  = hid_dim * config[self.backbone]['num_heads'],
                                                out_feats = hid_dim,
                                                num_heads = config[self.backbone]['num_heads'],
                                                feat_drop = config[self.backbone]['feat_drop'],
                                                attn_drop = config[self.backbone]['attn_drop'],
                                                residual  = False,
                                                activation= self.act) for _ in range(self.num_layers-2)])
                
                self.gnn_layers.append(GATConv(in_feats  = hid_dim * config[self.backbone]['num_heads'],
                                               out_feats = out_dim,
                                               num_heads = 1,
                                               feat_drop = config[self.backbone]['feat_drop'],
                                               attn_drop = config[self.backbone]['attn_drop'],
                                               residual  = False,
                                               activation= self.tail_act))
        elif self.backbone == 'sage':
            self.gnn_layers.append(SAGEConv(in_feats        = in_dim,
                                            out_feats       = hid_dim,
                                            aggregator_type = config[self.backbone]['agg'],
                                            feat_drop       = config[self.backbone]['feat_drop'],
                                            activation      = self.act))
            if self.num_layers > 1:
                self.gnn_layers.extend([SAGEConv(in_feats       = hid_dim,
                                                out_feats       = hid_dim,
                                                aggregator_type = config[self.backbone]['agg'],
                                                feat_drop       = config[self.backbone]['feat_drop'],
                                                activation      = self.act) for _ in range(self.num_layers-2)])
                
                self.gnn_layers.append(SAGEConv(in_feats        = hid_dim,
                                                out_feats       = out_dim,
                                                aggregator_type = config[self.backbone]['agg'],
                                                feat_drop       = config[self.backbone]['feat_drop'],
                                                activation      = self.tail_act))
    def forward(self, inputs: torch.Tensor, g: dgl.graph):
        h = inputs
        for i, conv in enumerate(self.gnn_layers):
            h = conv(g, h).flatten(1) if self.backbone == 'gat' else conv(g, h)
        return h

class SupConGraph(nn.Module):
    def __init__(self, config, in_dim):
        super(SupConGraph, self).__init__()
        self.hid_dim         = config['hid_dim']
        self.n_cls           = config['num_cls']
        self.encoder = GNN(config, in_dim, out_dim = self.hid_dim)
        head                 = config['head']
        if head == 'linear':
            self.proj = nn.Linear(self.hid_dim, self.hid_dim)
        elif head == 'mlp':
            self.proj = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim),
                                      nn.ReLU(inplace = True),
                                      nn.Linear(self.hid_dim, self.hid_dim))
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
    
    def forward(self, feat, g):
        emb = self.encoder(feat, g)
        return F.normalize(self.proj(emb), dim = 1)

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_dim, out_dim)
    
    def forward(self, embs):
        return self.classifier(embs)

class SupConLoss(nn.Module):
    def __init__(self, config, device, contrast_mode = 'all'):
        super(SupConLoss, self).__init__()
        self.temperature = config['temperature']
        self.contrast_mode = config['contrast_mode']
        self.base_temperature = config['base_temperature']
        self.device = device

    def forward(self,features, labels = None, mask = None):
        # feature shape: (N, 2, D/2)
        # 表示N个锚节点和他们的positive 节点
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)        # 

        batch_size = features.shape[0]  # 一个batch有N个锚节点 每个锚节点采样一个positive node 从它的类中，所以一个batch一共有2N个节点 batch_size = N
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is not None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)  # NxN单位对角阵
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # (N, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask  = torch.eq(labels, labels.T).float().to(self.device) # shape (N, N), 每行mask出和它同类的节点
        else: 
            mask = mask.float().to(self.device)
        
        contrast_count = features.shape[1]  # 2 每个节点和2N个节点对比
        # feature shape : (N, 2, D/2) - > ((N,D/2), (N,D/2)) - >  (2N, D/2)  前N个节点为N个锚节点，后N个节点依次为N个锚节点的positive 节点
        contrast_feature = torch.cat(torch.unbind(features, dim = 1), dim = 0)
        if self.contrast_mode == 'one':  # N 个embs作为anchor 和2N个embs计算对比损失
            anchor_feature = features[:, 0]  # shape (N, D/2) N个节点的第一个features 
            anchor_count = 1
        elif self.contrast_mode == 'all':  # 2N个embs作为anchor 和2N个embs计算对比损失
            anchor_feature = contrast_feature
            anchor_count = contrast_count  # 2N个节点全部作为锚节点
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # compute logits 
        # batch 节点互相的相似度
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)  # 2N x 2N， 所有2N个embs之间的相似度

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim = 1, keepdim = True)
        logits        = anchor_dot_contrast - logits_max.detach()  # 每行减最大相似度

        # tile mask  
        # mask：NxN 表示N个batch节点，每个batch节点mask出batch中的同类节点
        mask          = mask.repeat(anchor_count, contrast_count)  
        # mask-out self-contrast cases
        # (2N, 2N)的全1矩阵
        # batch_size = N logits_mask: 2N个节点 每个节点mask出除自身外的所有其他所有节点
        logits_mask   = torch.scatter(torch.ones_like(mask),   # 2Nx2N的全一矩阵 target 张量中填充0 
                                      1,  # 按行 表示(2N, 2N)矩阵的
                                      torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),  # index，[0,1,2,... 32x2-1] 
                                      0)

        # 这里mask为2N个节点的每个节点  mask出除自身外的其他所有positive nodes
        mask          = mask * logits_mask  


        # compute log_prob
        exp_logits  = torch.exp(logits) * logits_mask # 每个节点和其他所有节点的exp相似度
        # 分子： 锚节点i 和 任意节点j相似度， 分母: i和其他所有节点的相似度之和
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # log(exp (logits) / exp_logits.sum(1))  

        # compute mean of log-likelihood over positive 
        # 表示最大化节点和除自身外batch中其他positive nodes的相似度，最小化自身和batch中其他所有nodes的相似度
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 每个节点和2N个节点中其他positive 节点的平均相似度

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss