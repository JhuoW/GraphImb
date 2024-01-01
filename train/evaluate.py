from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
import numpy as np
import torch
from collections import namedtuple
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn import metrics

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


def evaluate_network(config, model, data, device, loss_func,epoch, cls_avg_feats):
    model.eval()
    eval_logits  = []
    eval_labels  = []
    eval_probs   = []
    accs, baccs, f1s = [], [], []
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
    elif config['model'] == 'graphimb':
        data = data.to(device)
        data_train_mask = data.ndata['imb_train_mask'].to(device)
        feat = data.ndata['feat'].to(device)
        labels = data.ndata['label'].to(device)
        val_mask = data.ndata['val_mask'].to(device)
        test_mask = data.ndata['test_mask'].to(device)
        cls_avg_feats = cls_avg_feats.to(device)
        out, layerwise_embs = model(data, feat, cls_avg_feats)
        val_loss = model.loss(out[val_mask], labels[val_mask], loss_func)

        for i, mask in enumerate([data_train_mask, val_mask, test_mask]):
            pred = out[mask].max(1)[1]
            y_pred = pred.cpu().numpy()
            y_true = data.ndata['label'][mask].cpu().numpy()
            acc = pred.eq(data.ndata['label'][mask]).sum().item() / mask.sum().item()
            bacc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
    elif config['model'] in ['gcn', 'sage']:
        data = data.to(device)
        data_train_mask = data.ndata['imb_train_mask'].to(device)
        feat = data.ndata['feat'].to(device)
        labels = data.ndata['label'].to(device)
        val_mask = data.ndata['val_mask'].to(device)
        test_mask = data.ndata['test_mask'].to(device)
        out = model(data, feat)
        val_loss = model.loss(out[val_mask], labels[val_mask], loss_func)

        for i, mask in enumerate([data_train_mask, val_mask, test_mask]):
            pred = out[mask].max(1)[1]
            y_pred = pred.cpu().numpy()
            y_true = data.ndata['label'][mask].cpu().numpy()
            acc = pred.eq(data.ndata['label'][mask]).sum().item() / mask.sum().item()
            bacc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
    DataType   = namedtuple('Metrics', ['train_acc', 'train_f1' , 'train_bacc', 
                                        'val_acc'  , 'val_f1'   , 'val_bacc',
                                        'test_acc' , 'test_f1'  , 'test_bacc'])
    results    = DataType(train_acc = accs[0], train_f1 = f1s[0], train_bacc = baccs[0],
                            val_acc   = accs[1], val_f1   = f1s[1], val_bacc   = baccs[1],
                            test_acc  = accs[2], test_f1  = f1s[2], test_bacc  = baccs[2])
    return val_loss, results

def evaluate_supcon(config, data, model, classifier, device):
    model.eval()
    classifier.eval()
    data = data.to(device)
    feats = data.ndata['feat'].to(device)
    labels = data.ndata['label'].to(device)
    val_mask = data.ndata['val_mask'].to(device)
    test_mask = data.ndata['test_mask'].to(device)
    accs, baccs, f1s = [], [], []
    with torch.no_grad():
        output = classifier(model.encoder(feats, data))
        for i, mask in enumerate([val_mask, test_mask]):
            pred = torch.nn.Softmax(dim=1)(output.detach().cpu())[mask]
            pred = np.argmax(pred.numpy(), axis=1)
            y_true = labels[mask].cpu().numpy()
            f1 = f1_score(y_true, pred, average='macro')
            acc = metrics.accuracy_score(y_true, pred)
            # print(y_true.shape)
            # print(pred.shape)
            bacc = balanced_accuracy_score(y_true, pred)
            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
    DataType   = namedtuple('Metrics', ['val_acc'  , 'val_f1'   , 'val_bacc',
                                        'test_acc' , 'test_f1'  , 'test_bacc'])
    results    = DataType(val_acc   = accs[0], val_f1   = f1s[0], val_bacc   = baccs[0],
                          test_acc  = accs[1], test_f1  = f1s[1], test_bacc  = baccs[1])
    return results