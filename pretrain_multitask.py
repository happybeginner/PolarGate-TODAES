# coding=utf-8
import argparse
import os
import random
import sys
from datetime import datetime
from collections import defaultdict
parent_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir_path)
from pathlib import Path
import time
import logging
import copy
# from icecream import ic
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from layers import MLP
from model import PolarGate
from load_dataset import Dataset
import warnings
from metala import MetaLAConfig
# warnings.filterwarnings("ignore")


def get_logger(name, logfile=None):
    logger = logging.getLogger(name)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PolarGate_processed')
    parser.add_argument('--task_type', type=str, default='prob', choices=['prob', 'tt', "multi"])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='PolarGate')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split_file', type=str, default='0.05-0.05-0.9')
    parser.add_argument('--loss_type', type=str, default='mae', choices=['mae', 'mse'])
    parser.add_argument('--name_others', type=str, default='')

    parser.add_argument('--asynchronous', action='store_true', default=False)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating weights')

    args = parser.parse_args()

    args.data_root_path = Path.home().joinpath('AIGDataset', args.dataset)
    args.pi_edges_path = args.data_root_path.joinpath('npz', 'pi_edges.npz')
    args.tt_pair_path = args.data_root_path.joinpath('npz', 'labels.npz')

    os.makedirs(Path.cwd().joinpath('ft_saved'), exist_ok=True)
    os.makedirs(Path.cwd().joinpath('results'), exist_ok=True)

    result = []
    if args.asynchronous:
        result.append('asynchronous')
    result_string = '_'.join(result)
    args.name_others += result_string

    if args.name_others != '':
        args.ft_model_path = Path.cwd().joinpath('ft_saved',
                f'{args.task_type}_{args.split_file}_{args.model}_{str(args.layer_num)}_{args.name_others}_state_dict.pth')
    else:
        args.ft_model_path = Path.cwd().joinpath('ft_saved',
                f'{args.task_type}_{args.split_file}_{args.model}_{str(args.layer_num)}_state_dict.pth')
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    return args


def split_data(args):
    if args.split_file == '1.0-0-0':
        with open(os.path.join(args.data_root_path, 'split', args.split_file, 'full.txt')) as file:
            lines = file.readlines()
        full_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]
        random.shuffle(full_file)
        return full_file, [], []
    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'train.txt')) as file:
        lines = file.readlines()
    train_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]

    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'valid.txt')) as file:
        lines = file.readlines()
    valid_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]
    random.shuffle(valid_file)

    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'test.txt')) as file:
        lines = file.readlines()
    test_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]

    return train_file, valid_file, test_file


def load_model(args):
    if args.asynchronous:
        model = PolarGate(args=args, device=args.device, in_dim=args.in_dim, out_dim=args.out_dim,
                           layer_num=args.layer_num).to(args.device)
    else:
        model = PolarGate(args=args, device=args.device, in_dim=args.in_dim, out_dim=args.out_dim,
                          layer_num=args.layer_num).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def AIG_Not_Edge_Loss(edge_index_s, out, typ='mae'):
    not_edges = edge_index_s[edge_index_s[:, 2] == -1]
    input_out = out[not_edges[:, 0]]
    output_out = out[not_edges[:, 1]]
    edges_sum = input_out + output_out
    if typ == 'mae':
        mae = F.l1_loss(edges_sum, torch.ones_like(edges_sum))
    else:
        mae = F.mse_loss(edges_sum, torch.ones_like(edges_sum))
    return mae


def AIG_AND_Edge_Loss(edge_index_s, out, typ='mae'):
    sorted_indices = torch.argsort(edge_index_s[:, 1])
    edge_index_s = edge_index_s[sorted_indices]
    and_edges = edge_index_s[edge_index_s[:, 2] == 1]
    inputs = out[and_edges[:, 0]]
    outputs = out[and_edges[:, 1]]
    inputs = inputs.view(-1, 2)
    outputs = outputs.view(-1, 2)
    min_inputs = torch.min(inputs, dim=1)[0]
    if typ == 'mae':
        loss = F.l1_loss(outputs[:, 0], min_inputs)
    else:
        loss = F.mse_loss(outputs[:, 0], min_inputs)
    return loss


def AIG_PI_AND_Edge_Loss(edge_index_s, out, typ='mae'):
    edge_index = edge_index_s[:, :2]
    edge_sign = edge_index_s[:, 2]
    neg_nodes = edge_index[edge_sign == -1][:, 1]

    all_nodes = torch.arange(out.size(0), device=out.device)
    pos_nodes = torch.tensor(list(set(all_nodes.tolist()) - set(neg_nodes.tolist())), device=out.device)
    loss_nodes = out[pos_nodes]

    if typ == 'mae':
        loss = F.l1_loss(loss_nodes, torch.zeros_like(loss_nodes))
    else:
        loss = F.mse_loss(loss_nodes, torch.zeros_like(loss_nodes))

    return loss

def zero_normalization(x):
    if x.shape[0] == 1: return x
    mean_x = torch.mean(x)
    std_x = torch.std(x) + 1e-8
    z_x = (x - mean_x) / std_x
    return z_x

def test(args, model, test_dataloader):
    model.eval()
    results = {'prob': 0.0, 'not': 0.0, 'and': 0.0, 'tt': 0.0, 'pi_and': 0.0,
               'level': defaultdict(lambda: {'value': 0.0, 'cnt': 0.0})}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, leave=False, desc="Testing")):
            batch = batch.to(args.device)
            edge_index_s = torch.cat((batch.edge_index.t(), batch.edge_sign.unsqueeze(1)), dim=1).long()
            node_labels_tensor = batch.prob.unsqueeze(1)
            out_emb, prob = model(batch.x, edge_index_s, batch)

            node_a = out_emb[batch.tt_pair_index[0]]
            node_b = out_emb[batch.tt_pair_index[1]]
            emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
            emb_dis_z = zero_normalization(emb_dis)
            tt_dis_z = zero_normalization(batch.tt_dis)

            # edge_batch = torch.zeros_like(emb_dis)
            # emb_dis_z = torch.zeros_like(emb_dis)
            # tt_dis_z = torch.zeros_like(batch.tt_dis)
            # for i in range(torch.max(batch.batch)):
            #     mask = (batch.batch[batch.tt_pair_index[0]]==i) & (batch.batch[batch.tt_pair_index[1]]==i)
            #     edge_batch[mask] = i
                
            # for i in range(torch.max(batch.batch)):
            #     emb_dis_z[edge_batch == i] = zero_normalization(emb_dis[edge_batch == i])
            #     tt_dis_z[edge_batch == i] = zero_normalization(batch.tt_dis[edge_batch == i])


            results['not'] += AIG_Not_Edge_Loss(edge_index_s, prob, typ=args.loss_type)
            results['and'] += AIG_AND_Edge_Loss(edge_index_s, prob, typ=args.loss_type)
            results['pi_and'] += AIG_PI_AND_Edge_Loss(edge_index_s, prob, typ=args.loss_type)

            if args.loss_type == 'mae':
                prob_loss = F.l1_loss(prob, node_labels_tensor)
                func_loss = F.l1_loss(emb_dis_z, tt_dis_z)
            else:
                prob_loss = F.mse_loss(prob, node_labels_tensor)
                func_loss = F.mse_loss(emb_dis_z, tt_dis_z)

            results['prob'] += prob_loss.item()
            results['tt'] += func_loss.item()


    results['prob'] /= len(test_dataloader)
    results['not'] /= len(test_dataloader)
    results['and'] /= len(test_dataloader)
    results['tt'] /= len(test_dataloader)
    results['pi_and'] /= len(test_dataloader)

    return results


def train(args, model, optimizer, train_dataloader, valid_dataloader, logger):
    logger.info('*********** Start End-to-End Train ***********')
    best_loss = 999999999999
    patience = args.patience
    total_time = 0.0
    cnt_epoch = 0
    epoch = -1

    while cnt_epoch < args.epochs:
        epoch += 1
        t = time.time()
        model.train()
        total_prob_loss = 0
        total_tt_loss = 0

        params_epoch_init = copy.deepcopy(model.state_dict())
        skip_flag = False
        accumulation_counter = 0

        for i, batch in enumerate(tqdm(train_dataloader, leave=False, desc=f'Epoch:{epoch}')):
            try:
                max_nodes = 8840 if args.task_type == 'tt' else 9000

                if batch.prob.shape[0]>max_nodes:
                    skip_flag = True
                    break

                batch = batch.to(args.device)
                node_labels_tensor = batch.prob.unsqueeze(1)
                edge_index_s = torch.cat((batch.edge_index.t(), batch.edge_sign.unsqueeze(1)), dim=1).long()
                out_emb, prob = model(batch.x, edge_index_s, batch)

                node_a = out_emb[batch.tt_pair_index[0]]
                node_b = out_emb[batch.tt_pair_index[1]]
                emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
                emb_dis_z = zero_normalization(emb_dis)
                tt_dis_z = zero_normalization(batch.tt_dis)

                if args.loss_type == 'mae':
                    prob_loss = F.l1_loss(prob, node_labels_tensor)
                    func_loss = F.l1_loss(emb_dis_z, tt_dis_z)
                else:
                    prob_loss = F.mse_loss(prob, node_labels_tensor)
                    func_loss = F.mse_loss(emb_dis_z, tt_dis_z)
            

                total_prob_loss += prob_loss.item()
                total_tt_loss += func_loss.item()

                # Scale loss for gradient accumulation
                if args.task_type == 'prob':
                    loss = prob_loss / args.gradient_accumulation_steps
                elif args.task_type == 'tt':
                    loss = func_loss / args.gradient_accumulation_steps
                elif args.task_type == 'multi':
                    loss = (prob_loss + func_loss) / args.gradient_accumulation_steps

                loss.backward()
                accumulation_counter += 1

                # Update weights only after accumulating gradients
                if accumulation_counter % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                del batch, node_labels_tensor, edge_index_s, out_emb, prob, node_a, node_b, emb_dis, emb_dis_z, tt_dis_z
                torch.cuda.empty_cache()                
            except Exception as e:
                logger.info(f"CUDA out of memory. Deprecate this batch to continue. max memory:{torch.cuda.max_memory_allocated(args.device)/1024 ** 2}MB, {batch.prob.shape[0]}, {batch.tt_dis.shape[0]}")
                optimizer.zero_grad()
                accumulation_counter = 0
                # Release all GPU tensor references to free memory
                batch = node_labels_tensor = edge_index_s = None
                out_emb = prob = node_a = node_b = None
                emb_dis = emb_dis_z = tt_dis_z = None
                prob_loss = func_loss = loss = None
                import gc; gc.collect()
                torch.cuda.empty_cache()

        # Update remaining accumulated gradients at the end of epoch
        if accumulation_counter % args.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if skip_flag:
            model.load_state_dict(params_epoch_init)
            logger.info(f"Skip epoch {epoch} because this epoch has a batch with too many nodes.")
            continue

        total_time += time.time() - t

        cnt_epoch += 1
        valid_results = test(args, model, valid_dataloader)

        total_prob_loss /= len(train_dataloader)
        total_tt_loss /= len(train_dataloader)

        logger.info('Epoch: {:02d} ({:02d}) | [Train] Prob: {:.4f} Func: {:.4f} Multi(Total): {:.4f} |'
                    '[Valid] Prob: {:.4f} Func: {:.4f} Multi(Total): {:.4f} | PI_AND: {:.4f} AND: {:.4f} NOT: {:.4f}'.format(
                    epoch, cnt_epoch,total_prob_loss, total_tt_loss,total_prob_loss+total_tt_loss, valid_results['prob'], valid_results['tt'],valid_results['prob']+valid_results['tt'],
                    valid_results['pi_and'], valid_results['and'], valid_results['not']))

        if args.task_type == 'multi':
            valid_results[args.task_type] = valid_results['prob'] + valid_results['tt']
        if valid_results[args.task_type] < best_loss:
            best_loss = valid_results[args.task_type]
            best_model_info = {
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())
            }
            patience = args.patience

        patience -= 1
        if patience <= 0:
            break

    torch.save(best_model_info, args.ft_model_path)

    return total_time / cnt_epoch, best_model_info

if __name__ == '__main__':
    args = parameter_parser()
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.name_others == '':
        logname = f'{args.task_type}_{args.dataset}_{args.model}_{args.layer_num}__{args.split_file}_{timestamp}.log'
    else:
        logname = f'{args.task_type}_{args.dataset}_{args.model}_{args.layer_num}__{args.split_file}_{args.name_others}_{timestamp}.log'
    logfile = os.path.join('results', logname)
    logger = get_logger(__name__, logfile=logfile)

    logger.info(dict(args._get_kwargs()))
    train_files, valid_files, test_files = split_data(args)
    

    if args.split_file == '1.0-0-0':
        train_files = train_files + valid_files + test_files

        full_dataset = Dataset(args, train_files, 'full')
        full_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        model, optimizer = load_model(args)
        logger.info(MetaLAConfig().__dict__)

        avg_train_time, best_model_info = train(args, model, optimizer, full_dataloader, full_dataloader, logger)
        del model, optimizer
    else:
        train_dataset = Dataset(args, train_files, 'train')
        valid_dataset = Dataset(args, valid_files, 'valid')
        test_dataset = Dataset(args, test_files, 'test')

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        model, optimizer = load_model(args)
        logger.info(MetaLAConfig().__dict__)

        avg_train_time, best_model_info = train(args, model, optimizer, train_dataloader, valid_dataloader, logger)
        del model, optimizer
        model, optimizer = load_model(args)
        # best_model_info = torch.load(args.ft_model_path)
        model.load_state_dict(best_model_info['model_state_dict'])
        test_results = test(args, model, test_dataloader)
        logger.info('*********** Test Result ***********')
        logger.info(
            '[Test] Prob: {:.4f} Func: {:.4f} | PI_AND: {:.4f} AND: {:.4f} NOT: {:.4f} | Avg.Train_time: {:.4f}'.format(
                test_results['prob'], test_results['tt'], test_results['pi_and'], test_results['and'],
                test_results['not'], avg_train_time))








