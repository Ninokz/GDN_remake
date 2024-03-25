import os
import torch
import random
import numpy as np

from GDN import GDN
from dataprocess import DataProcess
from evaluate import get_score
from train import train
from utils import get_device
from test import test

class Progeam:
    def __init__(self) -> None:
        self.dp = DataProcess('attack')
        edge_index_sets = []
        edge_index_sets.append(self.dp.fc_edges_indexs)
        self.model = GDN(edge_index_sets, len(self.dp.chosen_features_lst),
                         dim=64,
                         input_dim=5,
                         out_layer_num=1,
                         out_layer_inter_dim=128,
                         topk=5
                         ).to(get_device())
        
    def run(self):
        self.train_log = train(self.model, self.dp.train_dataloader, self.dp.val_dataloader, epoch=30, save_path='./model/gdn.pth')
        self.model.load_state_dict(torch.load('./model/gdn.pth'))
        best_model = self.model.to(get_device())
        _, self.test_result = test(best_model, self.dp.test_dataloader)
        _, self.val_result = test(best_model, self.dp.val_dataloader)

        get_score(self.test_result, self.val_result)


if __name__ == '__main__':
   p = Progeam()
   p.run()

