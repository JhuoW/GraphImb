import dgl
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import os.path as osp
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from utils.random_seeder import set_random_seed
from utils.logger import Logger
from data.dataHelper import DataHelper
import sys
from tqdm import tqdm
from nets.load_nets import get_model
from utils.utils import cls_spec_avg_feats, get_idx_info
from nets.SupCon import SupConLoss, Classifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
METRIC_NAME = [
    # 'train_acc', 'train_f1' , 'train_bacc', 
               'val_acc'  , 'val_f1'   , 'val_bacc',
               'test_acc' , 'test_f1'  , 'test_bacc']

def prepare(config, model):
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = config['lr'], 
                                 weight_decay = config.get('weight_decay', 0))
    scheduler = None
    if config.get('lr_scheduler', False):
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'],gamma=config['gamma'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               factor   = config['resi'], 
                                                               patience = config['lr_schedule_patience'], 
                                                               min_lr   = 1e-3)
    loss_func = nn.CrossEntropyLoss()
    return optimizer, scheduler, loss_func

def prepare_supcon(config, classifier, models, device):
    SupconLossFunc = SupConLoss(config, device, contrast_mode=config['contrast_mode']).to(device)
    model = models[0]
    classifier = models[1]
    optimizer   = torch.optim.Adam([{'params': model.parameters()},
                                 {'params': SupconLossFunc.parameters()}],
                                lr = config['lr'],
                                weight_decay= config.get('weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    optimizer2 = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': classifier.parameters()}],
                                 lr = config['lr'],
                                 weight_decay= config.get('weight_decay', 0))
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.5)
    optimizers = (optimizer, optimizer2)
    schedulers = (scheduler, scheduler2)
    return optimizers, schedulers, SupconLossFunc

def train_val_pipeline(config, logger, run_id, dataHelper:DataHelper):
    device = config['device']
    data = dataHelper.data
    cls_train_nodes, train_cls_masks = get_idx_info(data, dataHelper)
    if config['model'] == 'supcon':
        models = get_model(config['model'], config, train_cls_masks,device, dataHelper)
        model = models[0].to(device)
        classifier = models[1].to(device)
        optimizers, schedulers, SupconLossFunc = prepare_supcon(config, classifier, models, device)
        sc_optimizer = optimizers[0]
        sc_scheduler = schedulers[0]
        cls_optimizer = optimizers[1]
        cls_scheduler = schedulers[1]
    else:
        model = get_model(config['model'], config,train_cls_masks,device, dataHelper).to(device)
        optimizer, scheduler, loss_func = prepare(config, model)


    cls_avg_feats = cls_spec_avg_feats(data, dataHelper)
    # training_nodes_per_cls_imb = dataHelper.imb_idx_info
   
    from train.train import train_epoch, train_epoch_supcon, train_ce
    from train.evaluate import evaluate_network, evaluate_supcon

    best_val_metric  = 0

    best_metric_epoch 	= -1 # best number on dev set
    patience_cnt 		= 0
    best_model          = None
    best_results        = None
    best_classifier     = None
    monitor             = config['monitor']
    assert monitor in ['f1', 'acc', 'bacc']
    # training via SupCon at the first stage
    if len(dataHelper.imb_train_idx) % config['batch_size'] == 0:
        n_batch = dataHelper.imb_train_idx.shape[0] // config['batch_size']
    else:
        n_batch = dataHelper.imb_train_idx.shape[0] // config['batch_size'] + 1

    try: 
        with tqdm(range(config['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                if config['model'] == 'supcon':
                    sc_loss = train_epoch_supcon(config, data, model, SupconLossFunc, sc_optimizer, dataHelper, n_batch, device, sc_scheduler)
                    postfix_str = "<Epo %d> [SupCon Loss] %.4f " % (
                                    epoch, float(sc_loss))
                    t.set_postfix_str(postfix_str)                    
                    continue
                else:
                    epoch_train_loss, optimizer = train_epoch(config, data, model, optimizer, loss_func, device, epoch, cls_avg_feats, dataHelper, n_batch = n_batch)
                
                
                val_loss, results  = evaluate_network(config, model, data, device, loss_func,epoch, cls_avg_feats)
                if config.get('lr_scheduler', False):
                    scheduler.step(val_loss)


                val_metric = getattr(results, 'val_{}'.format(monitor))

                if val_metric > best_val_metric:
                    best_metric_epoch 	= epoch
                    best_val_metric     = val_metric
                    best_results        = results
                    best_model          = model
                    patience_cnt        = 0
                else:
                    patience_cnt     +=1
                if config['patience'] > 0 and patience_cnt >= config['patience']:
                    break
                training_loss = epoch_train_loss.detach().cpu().numpy()
                postfix_str = "<Epo %d> [Train Loss] %.4f [Val Acc] %.4f <Best Dev:> [Epo] %d [Acc] %.4f [F1]: %.4f [bAcc]: %.4f <Test> [Acc] %.4f [F1]: %.4f [bAcc]: %.4f" % (
                                epoch, training_loss, results.val_acc, best_metric_epoch,  best_results.val_acc, best_results.val_f1, best_results.val_bacc, best_results.test_acc, best_results.test_f1, best_results.test_bacc)
                
                t.set_postfix_str(postfix_str)
                # Saving checkpoint
                
                if not os.path.exists(config['ckpt_dir']):
                    os.makedirs(config['ckpt_dir'])
                torch.save(best_model.state_dict(), '{}.pkl'.format(config['ckpt_dir'] + "/best_model"))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    if config['model'] == 'supcon':
        logger.log("Start Running Classifier...")
        classifier_epochs = config['classifier_epochs']
        t2 = tqdm(range(classifier_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for epoch in t2:
            epoch_train_loss = train_ce(config, data, model, classifier, cls_optimizer, device, scheduler = cls_scheduler)
            with torch.no_grad():
                results = evaluate_supcon(config, data, model, classifier, device)
            val_metric = getattr(results, 'val_{}'.format(monitor))
            if val_metric > best_val_metric:
                best_metric_epoch 	= epoch
                best_val_metric     = val_metric
                best_results        = results
                best_model          = model
                best_classifier     = classifier
                patience_cnt        = 0   
            else:
                patience_cnt       += 1
            if config['patience'] > 0 and patience_cnt >= config['patience']:
                break
            postfix_str = "<Epo %d> [Train Loss] %.4f [Val Acc] %.4f <Best Dev:> [Epo] %d [Acc] %.4f [F1]: %.4f [bAcc]: %.4f <Test> [Acc] %.4f [F1]: %.4f [bAcc]: %.4f" % (
                            epoch, float(epoch_train_loss), results.val_acc, best_metric_epoch, best_results.val_acc, best_results.val_f1, best_results.val_bacc, best_results.test_acc, best_results.test_f1, best_results.test_bacc)
            t2.set_postfix_str(postfix_str)
            if not os.path.exists(config['ckpt_dir']):
                os.makedirs(config['ckpt_dir'])
            torch.save(best_model.state_dict(), '{}.pkl'.format(config['ckpt_dir'] + "/best_model"))  
            torch.save(best_classifier.state_dict(), '{}.pkl'.format(config['ckpt_dir'] + "/best_classifier"))       


    # save config
    config_resave_file = osp.join(config['ckpt_dir'], '{}.yml'.format(config['dataset']))
    with open(config_resave_file, 'w+') as f:
        yaml.dump(config, f, sort_keys=True, indent = 2)


    logger.log("best epoch is %d" % best_metric_epoch)
    logger.log("Best Epoch Valid ACC is %.4f" % (best_results.val_acc))
    return best_results, epoch_train_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',     type = str,   default = 'configs/',                   help = "Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id',          type = int,   default = 2,                            help = "Please give a value for gpu id")
    parser.add_argument('--model',           type = str,   default = "supcon" ,                       help = "Please give a value for model name")
    parser.add_argument('--dataset',         type = str,   default = "cora",                       help = "Please give a value for dataset name")
    parser.add_argument('--data_dir',        type = str,   default = "/data2/weizhuo/datasets/",   help = "Please give a value for dataset folder path")
    parser.add_argument('--out_dir',         type = str,   default = "out/",                       help = "Please give a value for out_dir")
    parser.add_argument('--train_size',      type = float, default = 0.4)
    parser.add_argument('--val_size',        type = float, default = 0.2)
    args = parser.parse_args()

    logger          = Logger(mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()
    root_ckpt_dir = args.out_dir + 'checkpoints/' + args.model + "_" + args.dataset
    start_wall_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config_path = osp.join(args.config_file, args.dataset + '.yml')
    config      = get_config(config_path)
    config      = config[args.model]

    if args.gpu_id is not None:
        config['gpu']['id']  = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    if args.model is not None:
        MODEL_NAME   = args.model
    else:
        MODEL_NAME   = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    
    config['model']          = args.model
    config['device']         = device
    config['gpu_id']         = config['gpu']['id']
    config['dataset']        = args.dataset
    config['root_ckpt_dir']  = root_ckpt_dir
    config['datetime']       = start_wall_time
    config['ckpt_dir']       = os.path.join(config['root_ckpt_dir'], config['datetime'])
    datasetHelper            = load_data(args, config=config)
    
    if args.dataset in ['yelp', 'amazon', 'tfinance', 'tsocial']:
        datasetHelper.load_FD()
    elif args.dataset in ['cora', 'citeseer', 'pubmed']:
        datasetHelper.load_common()
    print_config(config)
    config['num_cls']        = datasetHelper.num_cls
    if config.get('seed',-1) > 0:
        set_random_seed(config['seed'])
        logger.log ("Seed set. %d" % (config['seed']))
    ress = []
    for run_id in range(config['multirun']):
        logger.add_line()
        logger.log ("\t\t%d th Run" % run_id)
        logger.add_line()
        final_results, epoch_train_loss = train_val_pipeline(config, logger, run_id, datasetHelper)
                
        logger.log("%d th Run ended. Final Train Loss is %s" % (run_id , str(epoch_train_loss.detach().cpu().numpy())))
        logger.log("%d th Run ended. Best Epoch is %s" % (run_id , str(final_results._asdict())))        
        ress.append(final_results)
    logger.add_line()
    test_result_file = osp.join(config['ckpt_dir'], 'results.txt')
    results_file = open(test_result_file,'w')
    
    for tv, metric in enumerate(METRIC_NAME):
        if tv == 3:
            logger.log ("#" * 30)
        metric_list = np.around([getattr(result, metric) for result in ress], decimals=5)
        avg         = np.mean(metric_list, axis = 0)
        std         = np.std(metric_list, axis=0, ddof=1)            
        logger.log("%s: %s" % (metric  , str([round(x,4) for x in metric_list])))
        logger.log("%s: avg / std = %.4f / %.4f" % (metric , avg , std))
        results_file.write("%s: avg / std = %.4f / %.4f" % (metric , avg , std))
        results_file.write("\n")
    results_file.close()

    # for metric in METRIC_NAME:
    #     metric_value = getattr(final_results, metric)
    #     avg         = np.mean(metric_list, axis = 0)
    #     std         = np.std(metric_list, axis=0, ddof=1)
    #     logger.log("%s : = %.4f" % (metric , metric_value))

if __name__ == "__main__":
    main()