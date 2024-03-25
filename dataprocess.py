import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Subset

from timedataset import TimeDataset


class DataProcess:
    def __init__(self, slide_config: dict, batch_size: int, val_ratio: float, label_name='') -> None:
        self.data_folder = './data/'
        self.chonsen_features_data_csv = self.data_folder + "features.csv"
        self.train_data_csv = self.data_folder + "train.csv"
        self.test_data_csv = self.data_folder + "test.csv"

        self.chosen_features_lst = pd.read_csv(self.chonsen_features_data_csv, index_col=False)["feature"].tolist()
        self.sr_train_data = pd.read_csv(self.train_data_csv)
        self.sr_test_data = pd.read_csv(self.test_data_csv)

        if label_name in self.sr_train_data.columns:
            self.sr_train_data = self.sr_train_data.drop(columns=[label_name])

        self.fc_graph = self._create_features_fully_connected_graph(self.chosen_features_lst)
        self.fc_edges_indexes = self.create_fc_edges_indexs(self.fc_graph, self.chosen_features_lst,
                                                            self.chosen_features_lst)
        self.fc_edges_indexes = torch.tensor(self.fc_edges_indexes, dtype=torch.long)

        self.train_data_in = []
        self.test_data_in = []
        self.__process_step_1__()

        self.train_dataset = TimeDataset(self.train_data_in, self.fc_edges_indexes, 'train', config=slide_config)
        self.test_dataset = TimeDataset(self.test_data_in, self.fc_edges_indexes, 'test', config=slide_config)

        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.__process_step_2__(self.train_dataset,
                                                                                                   self.test_dataset,
                                                                                                   batch_size=batch_size,
                                                                                                   val_ratio=val_ratio)

    def __process_step_1__(self):
        for feature in self.chosen_features_lst:
            if feature in self.sr_train_data.columns:
                self.train_data_in.append(self.sr_train_data.loc[:, feature].values.tolist())
            else:
                print(feature, 'not exist in data')
        count_train = len(self.train_data_in[0])
        self.train_data_in.append([0] * count_train)

        for feature in self.chosen_features_lst:
            if feature in self.sr_test_data.columns:
                self.test_data_in.append(self.sr_test_data.loc[:, feature].values.tolist())
            else:
                print(feature, 'not exist in data')
        count_test = len(self.test_data_in[0])
        self.test_data_in.append(self.sr_test_data[self.sr_test_data.columns[-1]].tolist())

    def __process_step_2__(self, train_dataset: TimeDataset, test_dataset: TimeDataset, batch_size: int, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size,
                                      shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch_size,
                                    shuffle=False)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0)

        return train_dataloader, val_dataloader, test_dataloader

    def create_features_fully_connected_graph(self, features_list: list):
        fc_graph = {}
        for feature in features_list:
            if feature not in fc_graph:
                fc_graph[feature] = []
            for other_feature in features_list:
                if other_feature != feature:
                    fc_graph[feature].append(other_feature)
        return fc_graph

    def _create_features_fully_connected_graph(self, features_list: list):
        fc_graph = {}
        for feature in features_list:
            fc_graph[feature] = [other_feature for other_feature in features_list if other_feature != feature]
        return fc_graph

    def create_fc_edges_indexs(self, fc_graph: dict, all_features: list, chosen_features: list):
        ch_f = chosen_features
        edge_indexes = [
            [],
            []
        ]
        for node_name, node_list in fc_graph.items():
            if node_name not in all_features:
                continue
            if node_name not in ch_f:
                ch_f.append(node_name)

            p_index = ch_f.index(node_name)
            for child in node_list:
                if child not in all_features:
                    continue
                if child not in ch_f:
                    raise ValueError(f"Child {child} not in index_feature_map")

                c_index = ch_f.index(child)
                edge_indexes[0].append(c_index)
                edge_indexes[1].append(p_index)
        return edge_indexes

    @staticmethod
    def get_batch_edge_index(org_edge_index, batch_num, node_num):
        edge_index = org_edge_index.clone().detach()
        edge_num = org_edge_index.shape[1]
        batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
        for i in range(batch_num):
            batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num
        return batch_edge_index.long()
