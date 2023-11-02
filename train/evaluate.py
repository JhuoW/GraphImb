from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
import numpy as np
import torch
from collections import namedtuple

def calc_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    labels = labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def evaluate_network(config, model, data, device, val_loader, test_loader, epoch):
    model.eval()
    eval_logits  = []
    eval_labels  = []
    eval_probs   = []
    if config['model'] == 'mlp':
        x         = data.ndata['feature']
        x         = x.to(device)
        val_mask  = data.ndata['val_mask'] .to(device)
        test_mask = data.ndata['test_mask'].to(device)
        labels    = data.ndata['label'].to(device)
        logits    = model(x)
        eval_loss = model.loss(logits[val_mask], labels[val_mask])
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = np.zeros_like(labels.detach().cpu().numpy())
        probs = probs.detach().cpu().numpy()
        preds[probs[:, 1] > thres] = 1
        val_mask = val_mask.detach().cpu().numpy()
        test_mask = test_mask.detach().cpu().numpy()
        vauc = roc_auc_score(labels[val_mask].detach().cpu().numpy(), probs[val_mask][:, 1])

        trec = recall_score(labels[test_mask].detach().cpu().numpy(), preds[test_mask])
        tpre = precision_score(labels[test_mask].detach().cpu().numpy(), preds[test_mask])
        tmf1 = f1_score(labels[test_mask].detach().cpu().numpy(), preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask].detach().cpu().numpy(), probs[test_mask][:, 1])
        conf_gnn = confusion_matrix(labels[test_mask].detach().cpu().numpy(), preds[test_mask])
        gmean_gnn = calc_gmean(conf_gnn)

        # for iter, eval_bach_nodes in enumerate(data_loader):
        #     eval_batch_x      = x[eval_bach_nodes]
        #     eval_batch_labels = labels[eval_bach_nodes]
        #     eval_batch_logits = model(eval_batch_x)
        #     eval_batch_probs  = eval_batch_logits.softmax(1)
        #     eval_logits.append(eval_batch_logits)
        #     eval_probs.append(eval_batch_probs)
        #     eval_labels.append(eval_batch_labels)
        # eval_logits = torch.cat(eval_logits, dim = 0)
        # eval_labels = torch.cat(eval_labels, dim = 0)
        # eval_probs  = torch.cat(eval_probs , dim = 0)

        epoch_val_loss = (eval_loss.sum()).detach().item()
        # epoch_val_loss = epoch_val_loss / val_mask.size(0)
        DataType   = namedtuple('Metrics', ['vf1', 'vauc' ,'tauc', 'tmaf1', 'tgmean'])
        results    = DataType(vf1 = f1, vauc = vauc, tauc = tauc, tmaf1 = tmf1, tgmean = gmean_gnn)

    return results, epoch_val_loss