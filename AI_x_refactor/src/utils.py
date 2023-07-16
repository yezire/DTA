import numpy as np
import torch
import os
import argparse
import random


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_cv_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=26,
                        help="global random seed")
    parser.add_argument("--d", default=1024, type=int,
                        help="embedding dimension d")
    parser.add_argument("--n", default=1.0, type=float,
                        help="global gradient norm to be clipped")
    parser.add_argument("--k", default=512, type=int,
                        help="dimension of project matrices k")
    parser.add_argument("--model", default="CreaTDA_og", type=str,
                        help="model type", choices=['CreaTDA_og', 'CreaTDA'])
    parser.add_argument("--l2-factor", default=1.0,
                        type=float, help="weight of l2 regularization")
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight-decay", default=0,
                        type=float, help='weight decay of optimizer')
    parser.add_argument("--num-steps", default=3000,
                        type=int, help='number of training steps')
    parser.add_argument("--device", choices=[-1, 0, 1, 2, 3],
                        default=0, type=int, help='device number (-1 for cpu)')
    parser.add_argument("--n-folds", default=5, type=int,
                        help="number of folds for cross validation")
    parser.add_argument("--round", default=10, type=int,
                        help="number of rounds of cross validation")
    parser.add_argument("--test-size", default=0.1, type=float,
                        help="portion of validation data w.r.t. trainval-set")
    parser.add_argument("--mask", default='random', type=str,
                        choices=['random', 'tda_disease'], help="masking scheme")
    args = parser.parse_args()
    return args


def get_re_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=26,
                        help="global random seed")
    parser.add_argument("--d", default=1024, type=int,
                        help="embedding dimension d")
    parser.add_argument("--n", default=1.0, type=float,
                        help="global gradient norm to be clipped")
    parser.add_argument("--k", default=512, type=int,
                        help="dimension of project matrices k")
    parser.add_argument("--model", default="CreaTDA", type=str,
                        help="model type", choices=['CreaTDA_og', 'CreaTDA'])
    parser.add_argument("--l2-factor", default=1.0,
                        type=float, help="weight of l2 regularization")
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight-decay", default=0,
                        type=float, help='weight decay of optimizer')
    parser.add_argument("--num-steps", default=3000,
                        type=int, help='number of training steps')
    parser.add_argument("--device", choices=[-1, 0, 1, 2, 3],
                        default=0, type=int, help='device number (-1 for cpu)')
    args = parser.parse_args()
    return args


def get_pre_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=26,
                        help="global random seed")
    parser.add_argument("--d", default=1024, type=int,
                        help="embedding dimension d")
    parser.add_argument("--k", default=512, type=int,
                        help="dimension of project matrices k")
    parser.add_argument("--l2-factor", default=1.0,
                        type=float, help="weight of l2 regularization")
    parser.add_argument("--model", default="CreaTDA", type=str, help="model type",
                        choices=['CreaTDA_og', 'CreaTDA', 'GTN', 'DTINet', 'RGCN', 'HGT'])
    parser.add_argument("--device", choices=[-1, 0, 1, 2, 3],
                        default=0, type=int, help='device number (-1 for cpu)')
    parser.add_argument("--top", default=200, type=int,
                        help="number of top indices")
    args = parser.parse_args()
    return args


def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return torch.Tensor(new_matrix)
