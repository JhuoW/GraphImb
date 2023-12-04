from nets.MLP_net import MLPNet
from nets.GImb_net import GImbNet
from data.dataHelper import DataHelper
from nets.private_contrast import PrivateSpace
from nets.SupCon import SupConGraph, Classifier
from nets.GRACE import Grace

def get_model(MODEL_NAME, config, train_cls_masks, device, dataHelper: DataHelper):
    if MODEL_NAME == 'graphimb':
        ps = PrivateSpace(config, dataHelper.data)
        model = GImbNet(config=config, in_dim = dataHelper.feat_dim, train_cls_masks=train_cls_masks, private_space = ps, out_dim = dataHelper.num_cls)
    elif MODEL_NAME in ['gcn','sage']:
        model = GImbNet(config=config, in_dim = dataHelper.feat_dim, out_dim = dataHelper.num_cls, module_name = MODEL_NAME)
    elif MODEL_NAME == 'supcon':
        supcon = SupConGraph(config, in_dim = dataHelper.feat_dim)
        classifier = Classifier(in_dim=config['hid_dim'], out_dim=dataHelper.num_cls)
        model      = (supcon, classifier)
    elif MODEL_NAME == 'grace':
        model = Grace(in_dim = dataHelper.feat_dim, hid_dim= config['hid_dim'], out_dim= config['out_dim'], num_layers=config['n_layer'], act_fn=config['act'], temp = config['temperature'])
    return model