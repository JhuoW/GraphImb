from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
import abc
from dgl.data.fraud import FraudDataset, FraudAmazonDataset
from sklearn.model_selection import train_test_split
from dgl import backend as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy import sparse as sp

class dataset:
    dataset_name               = None
    dataset_source_folder_path = None
    dataset_source_file_name   = None

    def __init__(self, dName=None) -> None:
        self.dataset_name = dName

    def print_dataset_information(self):
        print('Dataset Name: '        + self.dataset_name)
        print('Dataset Description: ' + self.dataset_descrition)

    @abc.abstractmethod
    def load(self):
        return


class DataHelper(dataset):
    def __init__(self, config, dName) -> None:
        super().__init__(dName)
        self.config = config
        self.dataset_source_folder_path = config['dataset_folder']
    def load(self):
        if self.dataset_name in ['amazon', 'yelp']:
            dataset = FraudDataset(name       = self.dataset_name,
                                   raw_dir    = self.dataset_source_folder_path,
                                   train_size = self.config['train_size'],
                                   val_size   = self.config['val_size'])   
        elif self.dataset_name in ['tfinance', 'tsocial']:
            dataset, label_dict = load_graphs('/ssd1/weizhuo/FDdatasets/{}'.format(self.dataset_name))
        norm = self.config['norm_feat']
        data = dataset[0]
        data.ndata['feature'] = torch.from_numpy(self.row_normalize(data.ndata['feature'], dtype=np.float32)) if norm else data.ndata['feature'].float()
        homo = self.config.get('homo', False)
        if self.dataset_name in ['tfinance', 'tsocial', 'grab']:
            if self.dataset_name == 'tfinance':
                data.ndata['label'] = data.ndata['label'].argmax(1)
        if homo:
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        relations =list(data.etypes)
        if self.config['add_self_loop']:
            for etype in relations:
                data = dgl.remove_self_loop(data, etype=etype)
                data = dgl.add_self_loop(data, etype=etype)
            print('add self loops')
        labels = data.ndata['label']
        index  = list(range(len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index,  # train_data 
                                                                labels[index],   # train_target
                                                                stratify     = labels[index],
                                                                train_size   = self.config['train_size'],
                                                                random_state = 2, 
                                                                shuffle      = True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, 
                                                                y_rest, 
                                                                stratify     = y_rest,  
                                                                test_size    = 0.67 if self.config['train_size'] == 0.4 else 0.1,
                                                                random_state = 2, 
                                                                shuffle      = True)
        train_mask = torch.zeros([len(labels)]).bool()
        val_mask   = torch.zeros([len(labels)]).bool()
        test_mask  = torch.zeros([len(labels)]).bool()

        train_mask[idx_train]    = 1
        val_mask[idx_valid]      = 1
        test_mask[idx_test]      = 1
        data.ndata["train_mask"] = F.tensor(train_mask)
        data.ndata["val_mask"]   = F.tensor(val_mask)
        data.ndata["test_mask"]  = F.tensor(test_mask)
        self.config_data(data, dataset)
        # print(self.val_nid.shape[0])
        if self.config['model']  == 'mlp':
            full_relations       = data.canonical_etypes


    def row_normalize(self, mx, dtype=np.float32):
        r"""Row-normalize sparse matrix.
        Reference: <https://github.com/williamleif/graphsage-simple>
        
        Parameters
        ----------
        mx : np.ndarray
            Feature matrix of all nodes.
        dtype : np.dtype
            Data type for normalized features. Default=np.float32

        Return : np.ndarray
            Normalized features.
        """
        rowsum    = np.array(mx.sum(1)) + 0.01
        r_inv     = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx        = r_mat_inv.dot(mx)
        
        return mx.astype(dtype)

    
    def config_data(self, data, dataset):
        self.data            = data
        self.dataset         = dataset
        self.train_mask      = data.ndata['train_mask']
        self.val_mask        = data.ndata['val_mask']
        self.test_mask       = data.ndata['test_mask']
        self.train_nid       = torch.LongTensor(torch.nonzero(self.train_mask, as_tuple=True)[0])
        self.val_nid         = torch.LongTensor(torch.nonzero(self.val_mask, as_tuple=True)[0])
        self.test_nid        = torch.LongTensor(torch.nonzero(self.test_mask, as_tuple=True)[0])        
        self.num_classes     = dataset.num_classes if not self.dataset_name in ['tfinance', 'tsocial', 'grab'] else len(torch.unique(data.ndata['label']))
        self.relations       = list(data.etypes)
        self.num_relations   = len(data.etypes)
        self.labels          = data.ndata['label'].squeeze().long()
        self.num_nodes       = self.labels.shape[0]
        self.feat            = data.ndata['feature']
        self.feat_dim        = self.feat.shape[1]  
        print(f"[Global] Dataset <{self.dataset_name}> Overview\n"
        #   f"\tAverage in-degree {sum(self.data.in_degrees):>6}, Average out-degree {sum(self.data.out_degrees):>6} \n"
          f"\t Num Edges {data.number_of_edges():>6}\n"
          f"\t Num Features {self.feat_dim:>6}\n"
          f"\tEntire (fraud/total) {torch.sum(self.labels):>6} / {self.labels.shape[0]:<6}\n"
          f"\tTrain  (fraud/total) {torch.sum(self.labels[self.train_nid]):>6} / {self.labels[self.train_nid].shape[0]:<6}\n"
          f"\tValid  (fraud/total) {torch.sum(self.labels[self.val_nid]):>6} / {self.labels[self.val_nid].shape[0]:<6}\n"
          f"\tTest   (fraud/total) {torch.sum(self.labels[self.test_nid]):>6} / {self.labels[self.test_nid].shape[0]:<6}\n")
    
    def get_featLoader(self):

        train_loader = DataLoader(self.train_nid, batch_size = self.config['batch_size'], shuffle=True)
        val_loader   = DataLoader(self.val_nid,   batch_size = self.val_nid.shape[0],     shuffle=False)
        test_loader  = DataLoader(self.test_nid,  batch_size = self.test_nid.shape[0],    shuffle=False)

        return train_loader, val_loader, test_loader
