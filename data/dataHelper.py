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
from dgl.data.utils import save_graphs
from scipy import sparse as sp
from dgl.data.citation_graph import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import dgl.transforms as T
import os.path as osp
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
    def load_FD(self):
        return

    @abc.abstractmethod
    def load_common(self):
        return 


class DataHelper(dataset):
    def __init__(self, config, dName) -> None:
        super().__init__(dName)
        self.config = config

    def load_FD(self):
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

    def load_common(self):
        trans_list = []
        if self.config['add_self_loop']:
            trans_list.append(T.AddSelfLoop())
        if self.config['norm_feat']:
            trans_list.append(T.RowFeatNormalizer(subtract_min = True, node_feat_names=['feat']))
        tfms = T.Compose(trans_list)
        if self.dataset_name == 'cora': # n_cls: 7
            dataset = CoraGraphDataset(raw_dir= '/data2/weizhuo/datasets/', transform=tfms)
        elif self.dataset_name == 'citeseer':# n_cls: 6
            dataset = CiteseerGraphDataset(raw_dir= '/data2/weizhuo/datasets/', transform=tfms)
        elif self.dataset_name == 'pubmed':# n_cls: 3
            dataset = PubmedGraphDataset(raw_dir= '/data2/weizhuo/datasets/', transform=tfms)
        else: 
            raise ValueError("Unknown dataset: {}".format(self.dataset_name))

        data            = dataset[0]
        self.num_nodes  = data.num_nodes()
        self.num_cls    = dataset.num_classes
        data_train_mask, data_val_mask, data_test_mask = data.ndata['train_mask'], data.ndata['val_mask'], data.ndata['test_mask']
        labels          = data.ndata['label']
        train_labels    = labels[data_train_mask]
        n_data_per_cls  = []
        for i in range(self.num_cls):
            num_in_cls = (train_labels == i).sum()
            n_data_per_cls.append(int(num_in_cls.item())) 

        idx_info = self.get_idx_info(labels, self.num_cls, data_train_mask) # [[训练集中label为1的节点]， [训练集中label为2的节点], .. ]
        self.training_nodes_per_cls = idx_info
        #for artificial imbalanced setting: only the last imb_class_num classes are imbalanced
        imb_cls_num = self.num_cls // 2  # 后一半的类为minor classes
        cls_num_list = n_data_per_cls  # 每个类的训练节点数
        new_cls_num_list = []
        imb_ratio = self.config['imb_ratio']
        max_num = np.max(cls_num_list[:self.num_cls-imb_cls_num])  # 前num_cls-imb_class_num个类中节点数最多的类
        
        imb_graph_path = 'imb_datasets/{}_{}_{}'.format(self.config['dataset'], imb_ratio, imb_cls_num)
        if osp.exists(imb_graph_path):
            data_tuple, _ = dgl.load_graphs(imb_graph_path)
            data = data_tuple[0]
            self.imb_idx_info = self.get_idx_info(labels, self.num_cls, data.ndata['imb_train_mask'])
        else:
            for i in range(self.num_cls):
                if imb_ratio > 1 and i > self.num_cls - 1 - imb_cls_num:
                    new_cls_num_list.append(min(int(max_num*(1./imb_ratio)), cls_num_list[i]))  # 对于小类，保留大类最多节点的1/10， 不满的则保留全部
                else:
                    new_cls_num_list.append(cls_num_list[i])  # 对于大类 保留所有节点
            cls_num_list = new_cls_num_list  # 每个类重新划分的节点数，大类保持全部，小类保持对多大类的1/10节点
            if imb_ratio >1:
                # idx_info: 每个类的节点
                # data_train_mask: 新的train_mask
                data_train_mask, idx_info = self.split_semi_dataset(self.num_nodes, n_data_per_cls, self.num_cls, cls_num_list, idx_info, fix = self.config['fix_dataset'])
                self.imb_idx_info = idx_info
            data.ndata['imb_train_mask'] = data_train_mask
            dgl.save_graphs('imb_datasets/{}_{}_{}'.format(self.config['dataset'], imb_ratio, imb_cls_num), data)
        self.config_data(data, dataset)


    def get_idx_info(self, label, n_cls, train_mask):
        index_list = torch.arange(len(label))
        idx_info = []
        for i in range(n_cls):
            cls_indices = index_list[((label == i) & train_mask)]  # 训练集中所有label 为i的节点
            idx_info.append(cls_indices)
        return idx_info

    def split_semi_dataset(self, total_node, n_data, n_cls, class_num_list, idx_info, fix = True):
        new_idx_info = []
        _train_mask = idx_info[0].new_zeros(total_node, dtype=torch.bool)
        for i in range(n_cls):
            if n_data[i] > class_num_list[i]:
                # 打乱类i的训练集节点
                if not fix:
                    cls_idx = torch.randperm(len(idx_info[i])) 
                else:
                    cls_idx = torch.arange(len(idx_info[i]))
                cls_idx = idx_info[i][cls_idx]
                # 大类全取全部， 小类全取大类的1/10的节点
                cls_idx = cls_idx[:class_num_list[i]]
                new_idx_info.append(cls_idx)
            else: # 若训练节点数小于预定义好的
                new_idx_info.append(idx_info[i])
            _train_mask[new_idx_info[i]] = True

        assert _train_mask.sum().long() == sum(class_num_list)
        assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

        return _train_mask, new_idx_info

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

    def train_pairs_numpy(self):
        labels = self.data.ndata['label'].numpy()
        node_idx = np.arange(self.num_nodes)
        pairs = np.reshape(np.concatenate((node_idx, labels)), (2, -1)).T
        return pairs
    
    def config_data(self, data, dataset):
        self.data            = data
        self.dataset         = dataset
        self.train_mask      = data.ndata['train_mask']
        self.val_mask        = data.ndata['val_mask']
        self.test_mask       = data.ndata['test_mask']
        self.imb_train_mask  = data.ndata['imb_train_mask']
        self.train_nid       = torch.LongTensor(torch.nonzero(self.train_mask, as_tuple=True)[0])
        self.val_nid         = torch.LongTensor(torch.nonzero(self.val_mask, as_tuple=True)[0])
        self.test_nid        = torch.LongTensor(torch.nonzero(self.test_mask, as_tuple=True)[0])
        self.imb_train_idx   = torch.LongTensor(torch.nonzero(self.imb_train_mask, as_tuple=True)[0])
        self.node_label_pairs = self.train_pairs_numpy()

        self.feat_dim        = data.ndata['feat'].shape[1] 
        self.labels          = data.ndata['label'] 
        print(f"[Global] Dataset <{self.dataset_name}> Overview\n"
        #   f"\tAverage in-degree {sum(self.data.in_degrees):>6}, Average out-degree {sum(self.data.out_degrees):>6} \n"
          f"\t Num Edges {data.number_of_edges():>6}\n"
          f"\t Num Features {self.feat_dim:>6}\n"
          f"\t Num Nodes {self.num_nodes:>6}\n")
