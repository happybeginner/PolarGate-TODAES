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
    parser.add_argument('--task_type', type=str, default='multi', choices=['prob', 'tt', "multi"])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='PolarGate', choices=['PolarGate', 'PolarGate_MetaLA'])
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split_file', type=str, default='0.05-0.05-0.9')
    parser.add_argument('--loss_type', type=str, default='mae', choices=['mae', 'mse'])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating weights')
    parser.add_argument('--asynchronous', action='store_true', default=False)

    args = parser.parse_args()

    args.data_root_path = Path.home().joinpath('AIGDataset', args.dataset)
    args.pi_edges_path = args.data_root_path.joinpath('npz', 'pi_edges.npz')
    args.tt_pair_path = args.data_root_path.joinpath('npz', 'labels.npz')

    os.makedirs(Path.cwd().joinpath('results'), exist_ok=True)

    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    return args


def split_data(args):
    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'train.txt')) as file:
        lines = file.readlines()
    train_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]

    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'valid.txt')) as file:
        lines = file.readlines()
    valid_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]

    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'test.txt')) as file:
        lines = file.readlines()
    test_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]

    return train_file, valid_file, test_file


def load_model(args):
    if args.model == 'PolarGate':
        from model import PolarGate
        model = PolarGate(args=args, device=args.device, in_dim=args.in_dim, out_dim=args.out_dim,
                          layer_num=args.layer_num).to(args.device)
    elif args.model == 'PolarGate_MetaLA':
        from model import PolarGate
        model = PolarGate(args=args, device=args.device, in_dim=args.in_dim, out_dim=args.out_dim,
                           layer_num=args.layer_num).to(args.device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def zero_normalization(x):
    if x.shape[0] == 1: return x
    mean_x = torch.mean(x)
    std_x = torch.std(x) + 1e-8
    z_x = (x - mean_x) / std_x
    return z_x

def test_inference_time(args, model, test_dataloader, logger):
    model.eval()
    inference_times = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, leave=False, desc="Testing Inference Time")):
            batch = batch.to(args.device)
            edge_index_s = torch.cat((batch.edge_index.t(), batch.edge_sign.unsqueeze(1)), dim=1).long()
            node_labels_tensor = batch.prob.unsqueeze(1)

            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

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

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            inference_times.append(end_time - start_time)

    avg_inference_time = sum(inference_times) / len(inference_times)
    total_inference_time = sum(inference_times)

    logger.info(f'Total Inference Time: {total_inference_time:.4f}s')
    logger.info(f'Average Inference Time per Batch: {avg_inference_time:.4f}s')

    return total_inference_time, avg_inference_time


def train_one_epoch(args, model, optimizer, train_dataloader, logger):
    logger.info('*********** Start Training for 1 Epoch ***********')

    
    model.train()
    total_prob_loss = 0
    total_tt_loss = 0
    accumulation_counter = 0
    batch_count = 0
    total_time = 0

    for i, batch in enumerate(tqdm(train_dataloader, leave=False, desc='Training Epoch')):
        try:
            max_nodes = 8840 if args.task_type == 'tt' else 9000

            if batch.prob.shape[0] > max_nodes:
                logger.info(f"Skipping batch {i} with {batch.prob.shape[0]} nodes (exceeds {max_nodes})")
                continue

            batch = batch.to(args.device)
            node_labels_tensor = batch.prob.unsqueeze(1)
            edge_index_s = torch.cat((batch.edge_index.t(), batch.edge_sign.unsqueeze(1)), dim=1).long()

            start_time = time.time()
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
            batch_count += 1

            # Update weights only after accumulating gradients
            if accumulation_counter % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_time += time.time() - start_time
            del batch, node_labels_tensor, edge_index_s, out_emb, prob, node_a, node_b, emb_dis, emb_dis_z, tt_dis_z
            torch.cuda.empty_cache()

        except Exception as e:
            logger.info(f"Error in batch {i}: {str(e)}")
            if torch.cuda.is_available():
                logger.info(f"CUDA memory: {torch.cuda.max_memory_allocated(args.device)/1024 ** 2}MB")
            optimizer.zero_grad()
            accumulation_counter = 0
            torch.cuda.empty_cache()

    # Update remaining accumulated gradients at the end of epoch
    if accumulation_counter % args.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    
    train_time = total_time

    if batch_count > 0:
        total_prob_loss /= batch_count
        total_tt_loss /= batch_count
        logger.info(f'Training Time for 1 Epoch: {train_time:.4f}s')
        logger.info(f'Train Loss - Prob: {total_prob_loss:.4f}, Func: {total_tt_loss:.4f}, Total: {total_prob_loss + total_tt_loss:.4f}')
    else:
        logger.info(f'No batches processed during training')

    return train_time


if __name__ == '__main__':
    args = parameter_parser()
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    logname = f'time_test_{args.model}_{args.task_type}_{args.split_file}_{timestamp}.log'
    logfile = os.path.join('results', logname)
    logger = get_logger(__name__, logfile=logfile)

    logger.info('='*60)
    logger.info(f'Testing Model: {args.model}')
    logger.info(f'Task Type: {args.task_type}')
    logger.info(f'Split: {args.split_file}')
    logger.info('='*60)
    logger.info(dict(args._get_kwargs()))

    # Load data
    train_files, valid_files, test_files = split_data(args)

    train_dataset = Dataset(args, train_files, 'train')
    test_dataset = Dataset(args, test_files, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Load model
    model, optimizer = load_model(args)
    logger.info(f'Model Parameters: {sum(p.numel() for p in model.parameters())}')

    # Test training time (1 epoch)
    train_time = train_one_epoch(args, model, optimizer, train_dataloader, logger)

    # Test inference time
    total_inference_time, avg_inference_time = test_inference_time(args, model, test_dataloader, logger)

    # Summary
    logger.info('='*60)
    logger.info('*********** Time Test Summary ***********')
    logger.info(f'Model: {args.model}')
    logger.info(f'Training Time (1 epoch): {train_time:.4f}s len:{len(train_dataset)}')
    logger.info(f'Total Inference Time: {total_inference_time:.4f}s len:{len(test_dataset)}')
    logger.info(f'Average Inference Time per Batch: {avg_inference_time:.4f}s')
    logger.info('='*60)
