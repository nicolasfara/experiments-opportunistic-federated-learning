import copy
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np


class CNNMnist(nn.Module):

    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def average_weights(models):
    """ Averages the weights

    Args:
        models (list): a list of state_dict

    Returns:
        state_dict: the average state_dict
    """
    w_avg = copy.deepcopy(models[0])
    for key in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[key] += models[i][key]
        w_avg[key] = torch.div(w_avg[key], len(models))
    return w_avg


def discrepancy(weightsA, weightsB):
    ''' Computes the discrepancy between two models

    Args:
        weightsA (state_dict): the dict containing the weights of the first model
        weightsB (state_dict): the dict containing the weights of the second model

    Returns:
        double: the discrepancy between the two models
    '''
    keys = weightsA.keys()
    S_t = len(keys)
    d = 0
    for key in keys:
        w_a = weightsA[key]
        w_b = weightsB[key]
        norm = torch.norm(torch.sub(w_a, w_b))
        d += norm
    return d / S_t


def get_dataset(args):
    data_dir = '../data/mnist/'
    apply_transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(data_dir,
                                   train=True,
                                   download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir,
                                  train=False,
                                  download=True,
                                  transform=apply_transform)

    user_groups = dataset_to_nodes_partitioning(train_dataset, args['num_users'])

    return train_dataset, test_dataset, user_groups


def dataset_to_nodes_partitioning(nodes_count: int, areas: int, random_seed: int, shuffling: bool = False):
    np.random.seed(random_seed)  # set seed from Alchemist to make the partitioning deterministic
    dataset_download_path = "../../build/dataset"
    apply_transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(dataset_download_path, train=True, download=True, transform=apply_transform)

    nodes_per_area = int(nodes_count / areas)
    dataset_labels_count = len(train_dataset.classes)
    split_nodes_per_area = np.array_split(np.arange(nodes_count), areas)
    split_classes_per_area = np.array_split(np.arange(dataset_labels_count), areas)
    nodes_and_classes = zip(split_nodes_per_area, split_classes_per_area)

    index_mapping = {}

    for index, (nodes, classes) in enumerate(nodes_and_classes):
        records_per_class = [index for index, (_, lab) in enumerate(train_dataset) if lab in classes]
        # intra-class shuffling
        if shuffling:
            np.random.shuffle(records_per_class)
        split_record_per_node = np.array_split(records_per_class, nodes_per_area)
        for node_index, node in enumerate(nodes):
            index_mapping[node_index] = split_record_per_node[node_index].tolist()

    print(len(index_mapping))
    return index_mapping
