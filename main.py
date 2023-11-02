import dgl
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import os.path as osp
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

METRIC_NAME = ['tauc', 'tmaf1', 'tgmean']

def config_model(config, dataHelper:DataHelper):
    model_params = {}

    if config["model"] == "mlp":
        
        model_params['in_dim']   = dataHelper.feat_dim
        model_params['hid_dim']  = config['hid_dim']
        model_params['out_dim']  = dataHelper.num_classes
        model_params['n_layer']  = config['n_layer']
        model_params['tail_act'] = config['tail_act']
        model_params['dropout']  = config['dropout']
        model_params['gn']       = config['gn']
        model_params['bias']     = config['bias']
        model_params['act']      = config['act']
    return model_params

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

    return optimizer, scheduler



def train_val_pipeline(config, logger, run_id, dataHelper:DataHelper, loaders):
    train_loader, val_loader, test_loader = loaders
    device = config['device']
    model_params = config_model(config, dataHelper)
    model = get_model(config['model'], model_params).to(device)
    optimizer, scheduler = prepare(config, model)
    data = dataHelper.data
    from train.train import train_epoch
    from train.evaluate import evaluate_network
    best_auc = 0. 
    best_metric_epoch 	= -1 # best number on dev set
    report_tst_result   = None
    patience_cnt 		= 0
    best_model          = None
    try: 
        with tqdm(range(config['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                epoch_train_loss, optimizer = train_epoch(config, data, model, optimizer, device, train_loader, epoch)
                results, epoch_val_loss     = evaluate_network(config, model, data, device, val_loader, test_loader, epoch)
                vauc                        = getattr(results, 'vauc')
                if best_auc < vauc:
                    best_metric_epoch = epoch
                    report_test_res   = results
                    patience_cnt      = 0
                    best_model        = model
                else:
                    patience_cnt     +=1
                if config['patience'] > 0 and patience_cnt >= config['patience']:
                    break
                t.set_postfix(epoch          = epoch, 
                              best_val_epoch = best_metric_epoch,
                              train_loss     = epoch_train_loss, 
                              val_auc        = results.vauc,
                              best_val_auc   = report_test_res.vauc,
                              test_auc       = report_test_res.tauc,
                              tmaf1          = report_test_res.tmaf1,
                              tgmean         = report_test_res.tgmean)
                # Saving checkpoint
                ckpt_dir = os.path.join(config['root_ckpt_dir'], "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(best_model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(best_metric_epoch)))
                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)
                if config['lr_scheduler']:
                    scheduler.step(epoch_val_loss)
                    # if optimizer.param_groups[0]['lr'] < config['min_lr']:
                    #     print("\n!! LR EQUAL TO MIN LR SET.")
                    #     break


    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    logger.log("best epoch is %d" % best_metric_epoch)
    logger.log("Best Epoch Valid AUC is %.4f" % (report_test_res.vauc))
    return report_test_res, epoch_train_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type = str,   default = 'configs/',                   help = "Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id',     type = int,   default = 1,                            help = "Please give a value for gpu id")
    parser.add_argument('--model',      type = str,   default = "mlp" ,                       help = "Please give a value for model name")
    parser.add_argument('--dataset',    type = str,   default = "yelp",                       help = "Please give a value for dataset name")
    parser.add_argument('--data_dir',   type = str,   default = "/ssd1/weizhuo/FDdatasets/",  help = "Please give a value for dataset folder path")
    parser.add_argument('--out_dir',    type = str,   default = "out/",                       help="Please give a value for out_dir")
    parser.add_argument('--train_size', type = float, default = 0.4)
    parser.add_argument('--val_size',   type = float, default = 0.2)
    args = parser.parse_args()
    logger          = Logger(mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()
    root_ckpt_dir = args.out_dir + 'checkpoints/' + args.model + "_" + args.dataset + "_" + str(args.train_size)
    
    config_path = osp.join(args.config, args.model + '_' + args.dataset+'_'+str(args.train_size) + '_' + str(args.val_size) + '.json')
    with open(config_path) as f:
        config = json.load(f)
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
    params = config['params']
    params['device']         = device
    params['gpu_id']         = config['gpu']['id']
    params['dataset_folder'] = config['dataset_folder']
    params['dataset_name']   = config['dataset']
    params['model']          = args.model
    params['root_ckpt_dir']  = root_ckpt_dir
    datasetHelper            = load_data(args, config=params)
    datasetHelper.load()
    print_config(params)
    if config.get('seed',-1) > 0:
        set_random_seed(config['seed'])
        logger.log ("Seed set. %d" % (config['seed']))
    loaders = datasetHelper.get_featLoader()  # loaders = (train_loader, val_loader, test_loader)
    ress = []
    for run_id in range(params['multirun']):
        logger.add_line()
        logger.log ("\t\t%d th Run" % run_id)
        logger.add_line()
        test_results, epoch_train_loss = train_val_pipeline(params, logger, run_id, datasetHelper, loaders)
        ress.append(test_results)
    logger.add_line()
    for metric in METRIC_NAME:
        metric_value = getattr(test_results, metric)
        logger.log("%s : = %.4f" % (metric , metric_value))

if __name__ == "__main__":
    main()