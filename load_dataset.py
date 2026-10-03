from typing import Optional, Callable, List
import os.path as osp
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import os
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
# from icecream import ic


def one_hot(idx, length):
    if type(idx) is int:
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    return x


def construct_node_feature(x, num_gate_types=3):
    gate_list = x[:, 1]
    gate_list = np.float32(gate_list)
    x_torch = one_hot(gate_list, num_gate_types)
    return x_torch


def calculate_forward_level(edge_index, num_nodes):

    node_ids = np.arange(num_nodes, dtype=int)
    node_order = np.zeros(num_nodes, dtype=int)
    unevaluated_nodes = np.ones(num_nodes, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        unready_children = child_nodes[unevaluated_mask]
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False
        n += 1

    forward_level = torch.from_numpy(node_order).long()
    forward_index = torch.LongTensor([i for i in range(num_nodes)])

    return forward_level, forward_index


class OrderedData(Data):
    def __init__(self, **kwargs):
    # def __init__(self, name=None, edge_index=None, edge_sign=None, x=None, y=None,
    #              pi_edge_index=None, pi_edge_sign=None, tt_pair_index=None, tt_dis=None,
    #              forward_level=None, forward_index=None):
        super().__init__()
        # self.name = name
        # self.edge_index = edge_index
        # self.edge_sign = edge_sign
        # self.pi_edge_index = pi_edge_index
        # self.pi_edge_sign = pi_edge_sign

        # self.x = x
        # self.y = y

        # self.tt_pair_index = tt_pair_index
        # self.tt_dis = tt_dis
        # self.forward_level = forward_level
        # self.forward_index = forward_index
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key or key == 'ID_seq':
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index':
            return 0
        elif key == "edge_index" or key == 'pi_edge_index' or key == 'tt_pair_index' or key == 'ID_seq' or key == 'ID_seq_mask':
            return 1
        else:
            return 0


def parse_signed_data(name, x, edge_index, edge_sign, pi_edges, tt_pair_index, tt_dis, y):

    node_features_tensor_d = torch.from_numpy(x).float()
    x_torch = construct_node_feature(node_features_tensor_d)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index_torch = edge_index.t().contiguous()
    assert edge_index_torch.shape[0] == 2  # edge_index_torch: [2, num_edges]
    edge_sign_torch = torch.LongTensor(edge_sign)

    pi_edges_signed_tensor = torch.from_numpy(pi_edges).long()
    pi_edge_index_torch = pi_edges_signed_tensor[:, :2].t().contiguous()
    assert pi_edge_index_torch.shape[0] == 2
    pi_edge_sign_torch = pi_edges_signed_tensor[:, 2]

    tt_pair_index_torch = torch.tensor(tt_pair_index, dtype=torch.long).t().contiguous()
    assert tt_pair_index_torch.shape[0] == 2
    tt_dis_torch = torch.tensor(tt_dis, dtype=torch.float32)

    forward_level, forward_index = calculate_forward_level(edge_index_torch, x_torch.shape[0])

    graph = OrderedData(name=name, edge_index=edge_index_torch, edge_sign=edge_sign_torch, x=x_torch,
                        pi_edge_index=pi_edge_index_torch, pi_edge_sign=pi_edge_sign_torch,
                        tt_pair_index=tt_pair_index_torch, tt_dis=tt_dis_torch,
                        forward_level=forward_level, forward_index=forward_index)

    graph.prob = torch.tensor(y).reshape(len(x))

    return graph

def reorder_graph(graph):
    "按照拓扑排序重新排列，******注意：只重新映射了要用到的属性，其他属性不变******"
    reorder_id = []     # new_id2old_id
    reorder_dist = {}   # old_id2new_id
    queue = []

    edge_index = graph.edge_index
    indegree = degree(edge_index[1], num_nodes=graph.num_nodes)
    src_dst_dist = {} # 空间换时间
    for i in range(edge_index.shape[1]):
        src = edge_index[0][i].item()
        dst = edge_index[1][i].item()
        if src not in src_dst_dist:
            src_dst_dist[src] = []
        src_dst_dist[src].append(dst)
    queue.extend(torch.where(indegree == 0)[0].tolist())

    with tqdm(total=graph.num_nodes, desc="Reordering graph", leave=False) as pbar:
        while queue:
            node = queue.pop(0)
            reorder_id.append(node)
            reorder_dist[node] = len(reorder_id) - 1

            target = src_dst_dist.get(node, [])
            for t in target:
                indegree[t] -= 1
                if indegree[t] == 0:
                    queue.append(t)
            pbar.update(1)
    assert len(reorder_id) == graph.num_nodes
    reorder_id = torch.tensor(reorder_id, dtype=torch.long)

    newgraph = OrderedData()
    newgraph.name = graph.name
    newgraph.x = graph.x[reorder_id]
    newgraph.prob = graph.prob[reorder_id]
    newgraph.edge_index = torch.tensor([[reorder_dist[edge_index[0][i].item()], reorder_dist[edge_index[1][i].item()]] for i in range(edge_index.shape[1])], dtype=torch.long).t()
    newgraph.edge_sign = graph.edge_sign
    newgraph.tt_pair_index = torch.tensor([[reorder_dist[graph.tt_pair_index[0][i].item()], reorder_dist[graph.tt_pair_index[1][i].item()]] for i in range(graph.tt_pair_index.shape[1])], dtype=torch.long).t()
    newgraph.tt_dis = graph.tt_dis

    # graph.x = graph.x[reorder_id]
    # graph.prob = graph.prob[reorder_id]
    # graph.edge_index = torch.tensor([[reorder_dist[edge_index[0][i].item()], reorder_dist[edge_index[1][i].item()]] for i in range(edge_index.shape[1])], dtype=torch.long).t()

    return newgraph

class Dataset(InMemoryDataset):
    def __init__(self, args, file_path_list, dataset_type, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'PolarGate'
        self.args = args

        os.makedirs('inmemory_dataset', exist_ok=True)
        self.root = 'inmemory_dataset'
     
        self.raw_files = file_path_list
        # assert dataset_type in ['train', 'valid', 'test']
        self.dataset_type = dataset_type

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), \
            "Cannot accept the transform, pre_transfrom and pre_filter args now."

        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.args.data_root_path

    @property
    def processed_dir(self):
        return osp.join(self.root, self.args.split_file, self.dataset_type)

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(path) for path in self.raw_files]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def load_edges(self, edge_file_path):
        edges = []
        edge_sign = []
        with open(edge_file_path, 'r') as f:
            for line in f:
                x = line.strip().split(',')
                assert len(x) == 3
                edges.append([int(x[0]), int(x[1])])
                edge_sign.append(float(x[2]))

        return edges, edge_sign

    def process(self):
        data_list = []

        pi_edges_dict = np.load(self.args.pi_edges_path, allow_pickle=True)['pi_edges'].item()
        tt_pair_dict = np.load(self.args.tt_pair_path, allow_pickle=True)['labels'].item()

        for data_dir in tqdm(self.raw_files):
            graph_name = os.path.basename(data_dir)
            edges, edge_sign = self.load_edges(os.path.join(data_dir, 'raw/signed_edge.csv'))
            node_features = np.genfromtxt(os.path.join(data_dir, 'raw/node-feat.csv'), delimiter=',')

            pi_edges_list = list(pi_edges_dict[graph_name].values())
            pi_edges_signed = np.concatenate(pi_edges_list, axis=0)
            tt_pair_index = tt_pair_dict[graph_name]['tt_pair_index']
            tt_dis = tt_pair_dict[graph_name]['tt_dis']

            node_labels = np.genfromtxt(os.path.join(data_dir, 'raw/prob.csv'), delimiter=None)

            graph = parse_signed_data(graph_name, node_features, edges, edge_sign, pi_edges_signed, tt_pair_index,
                                      tt_dis, node_labels)

            graph = reorder_graph(graph)

            # get left/right neibor ID seq
            ID_seq = torch.zeros((2, graph.num_nodes), dtype=torch.long)
            ID_seq_mask = torch.zeros((2, graph.num_nodes), dtype=torch.bool)
            for i in tqdm(range(graph.edge_index.shape[1]), desc="Get ID_seq", leave=False):
                target = graph.edge_index[1][i]
                source = graph.edge_index[0][i]

                if ID_seq_mask[0][target] == False:
                    ID_seq[0][target] = source
                    ID_seq_mask[0][target] = True
                elif ID_seq_mask[1][target] == False:
                    ID_seq[1][target] = source
                    ID_seq_mask[1][target] = True
                else:
                    assert False, f"Node {target} has more than 2 inputs!"
            graph.ID_seq = ID_seq
            graph.ID_seq_mask = ID_seq_mask

            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
        print('Total Circuits: {:}'.format(len(data_list)))

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PolarGate_processed')
    parser.add_argument('--split_file', type=str, default='0.05-0.05-0.9')
    args = parser.parse_args()

    args.data_root_path = Path.home().joinpath('AIGDataset', args.dataset)
    args.pi_edges_path = args.data_root_path.joinpath('npz', 'pi_edges.npz')
    args.tt_pair_path = args.data_root_path.joinpath('npz', 'labels.npz')

    file_path_list = ['/home/ljw/AIGDataset/PolarGate_processed/ac97_ctrl_syn_000',
                      '/home/ljw/AIGDataset/PolarGate_processed/logicProbe_syn_950',
                      '/home/ljw/AIGDataset/PolarGate_processed/mem_ctrl_syn_273']

    test_dataset = Dataset(args, file_path_list, 'test_code')
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=False)

    # ic(test_dataset[0].num_nodes)
    # ic(test_dataset[1].num_nodes)
    #
    for batch in test_dataloader:
        ic(batch)
        ic("Batch name:", batch.name)
        ic("Batch edge_index:", batch.edge_index.shape)
        ic("Batch edge_sign:", batch.edge_sign.shape)
        ic("Batch prob:", batch.prob)
        # ic(test_dataset[0].edge_index)
        # ic(test_dataset[1].edge_index)
        # ic("Batch x:", batch.x)
        # ic("Batch y:", batch.y)
        # ic("Batch pi_edge_index:", batch.pi_edge_index.t())
        # ic(test_dataset[0].pi_edge_index.t())
        # ic(test_dataset[1].pi_edge_index.t())
        # ic("Batch pi_edge_sign:", batch.pi_edge_sign)
        # ic("Batch tt_pair_index:", batch.tt_pair_index)
        # ic("Batch tt_dis:", batch.tt_dis)
        # ic("Batch forward_level:", batch.forward_level)
        # ic("Batch forward_index:", batch.forward_index)
        break