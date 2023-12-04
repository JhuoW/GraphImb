import torch
import random
import numpy as np

def train_epoch(config, data, model, optimizer, loss_func, device, epoch, cls_avg_feats, dataHelper):
    model.train()
    epoch_loss = 0
    nb_data    = 0
    # if config['model'] == 'mlp':
    #     x      = data.ndata['feature']
    #     x      = x.to(device)
    #     labels = data.ndata['label'].to(device)
    #     for iter, batch_nodes in enumerate(data_loader):
    #         batch_nodes  = batch_nodes.to(device)
    #         batch_x      = x[batch_nodes]
    #         optimizer   .zero_grad()

    #         batch_label  = labels[batch_nodes]
    #         batch_logits = model(batch_x)
    #         loss         = model.loss(batch_logits, batch_label)
    #         loss        .backward()
    #         optimizer   .step()
    #         epoch_loss  += loss.detach().item()
    #         nb_data     += batch_label.size(0)
    # epoch_loss /= (iter + 1)
    if config['model'] == 'graphimb':
        data_train_mask = data.ndata['imb_train_mask'].to(device)
        feat = data.ndata['feat'].to(device)
        labels = data.ndata['label'].to(device)
        data = data.to(device)
        cls_avg_feats = cls_avg_feats.to(device)
        optimizer.zero_grad()
        out, layerwise_embs  = model(data, feat, cls_avg_feats)        
        loss_ce = model.loss(out[data_train_mask], labels[data_train_mask], loss_func)
        loss_contrast = model.contrast_loss(layerwise_embs)
        loss = loss_ce + config['contrast_reg'] * loss_contrast
        loss.backward()
        optimizer.step()
    elif config['model'] in ['gcn','sage']:
        data_train_mask = data.ndata['imb_train_mask'].to(device)
        feat = data.ndata['feat'].to(device)
        labels = data.ndata['label'].to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out  = model(data, feat)        
        loss = model.loss(out[data_train_mask], labels[data_train_mask], loss_func)

        loss.backward()
        optimizer.step()


    return loss, optimizer

def train_epoch_supcon(config, data, model, SupconLoss, optimizer, dataHelper, n_batch, device, scheduler = None):
    epoch_loss = 0
    nb_data    = 0
    train_idx = torch.clone(dataHelper.imb_train_idx).numpy()
    random.shuffle(train_idx)
    for n in range(n_batch):
        model.train()
        optimizer.zero_grad()
        start = n * config['batch_size']
        labels = data.ndata['label']
        data = data.to(device)
        feats = data.ndata['feat'].to(device)
        end = (n + 1) * config['batch_size']
        node_id = train_idx[start: end] # 当前batch 的training nodes
        node_id2 = []
        for node in node_id:
            node_cls = labels[node]
            no_self = list(dataHelper.imb_idx_info[node_cls].numpy())
            no_self.remove(node)
            node_id2.extend([random.choice(no_self)])  # 为每个node随机采样positive training node       
        output = model(feats, data)
        # concat embs of anchor nodes and their positives  [B, 2D]  B = batchsize
        tmp_output = torch.cat([output[node_id], output[node_id2]], dim=1)   
        tmp_feature = torch.reshape(torch.unsqueeze(tmp_output, 1), (-1, 2, config['hid_dim']))  # (B,2,D)
        tmp_label = torch.reshape(labels[node_id].to(device), (-1, 1)) # batch labels
        sc_loss   = SupconLoss(tmp_feature, tmp_label)

        epoch_loss  += sc_loss.detach().item()
        # optimize based on supcon loss
        sc_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return epoch_loss / n_batch
        
def train_ce(config, data, model, classifier, optimizer, device, scheduler = None):
    model.train()
    classifier.train()
    optimizer.zero_grad()
    feats = data.ndata['feat'].to(device)
    data = data.to(device)
    imb_train_mask = data.ndata['imb_train_mask'].to(device)
    labels = data.ndata['label'].to(device)
    output = classifier(model.encoder(feats, data))
    loss   = torch.nn.CrossEntropyLoss()(output[imb_train_mask], labels[imb_train_mask])
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss