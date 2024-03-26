import json
import os
import torch
import random
import numpy as np
import torch_geometric

from GDN import GDN
from dataprocess import DataProcess
from evaluate import get_score
from train import train
from utils import get_device
from test import test

with open('config.json', 'r') as f:
    cfg = json.load(f)

DIM = cfg['train_param']['dim']
INPUT_DIM = cfg['train_param']['input_dim']
OUT_LAYER_NUM = cfg['train_param']['out_layer_num']
OUT_LAYER_INTER_DIM = cfg['train_param']['out_layer_inter_dim']
TOPK = cfg['train_param']['topk']

BATCH_SIZE = cfg['train_param']['batch_size']
VAL_RATIO = cfg['train_param']['val_ratio']
EPOCH = cfg['train_param']['epoch']


class Progeam:
    def __init__(self) -> None:
        self.dp = DataProcess(cfg, BATCH_SIZE, VAL_RATIO, 'attack')
        edge_index_sets = [self.dp.fc_edges_indexes]
        self.model = GDN(edge_index_sets, len(self.dp.chosen_features_lst),
                         dim=DIM,
                         input_dim=INPUT_DIM,
                         out_layer_num=OUT_LAYER_NUM,
                         out_layer_inter_dim=OUT_LAYER_INTER_DIM,
                         top_k=TOPK
                         ).to(get_device())

        self.train_log = None
        self.test_result = None
        self.val_result = None

    def run(self):
        self.train_log = train(self.model, self.dp.train_dataloader, self.dp.val_dataloader,
                               epoch=30, save_path='./model/gdn.pth')
        self.model.load_state_dict(torch.load('./model/gdn.pth'))
        best_model = self.model.to(get_device())
        _, self.test_result = test(best_model, self.dp.test_dataloader)
        _, self.val_result = test(best_model, self.dp.val_dataloader)

        get_score(self.test_result, self.val_result)


if __name__ == '__main__':
    p = Progeam()
    p.run()


