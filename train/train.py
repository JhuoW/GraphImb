import torch


def train_epoch(config, data, model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    nb_data    = 0
    if config['model'] == 'mlp':
        x      = data.ndata['feature']
        x      = x.to(device)
        labels = data.ndata['label'].to(device)
        for iter, batch_nodes in enumerate(data_loader):
            batch_nodes  = batch_nodes.to(device)
            batch_x      = x[batch_nodes]
            optimizer   .zero_grad()

            batch_label  = labels[batch_nodes]
            batch_logits = model(batch_x)
            loss         = model.loss(batch_logits, batch_label)
            loss        .backward()
            optimizer   .step()
            epoch_loss  += loss.detach().item()
            nb_data     += batch_label.size(0)
    epoch_loss /= (iter + 1)

    return epoch_loss, optimizer
