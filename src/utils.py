import os
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import copy
import glob
from numpy.testing import assert_array_almost_equal

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)



def create_datasets(data_path, dataset_name, num_clients, num_shards, iid,noisy_type,noisy_rate,non_iid_labels_rate,batch_size):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    dataset_name = dataset_name.upper()
    # get dataset from torchvision.datasets if exists


    if dataset_name == "CINIC10":
        num_classes = 10
        cinic10_root = 'data/cinic-10'

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])])

        training_dataset = torchvision.datasets.ImageFolder(root=cinic10_root + '/train', transform=transform)
        test_dataset = torchvision.datasets.ImageFolder(root=cinic10_root + '/test', transform=transform)
        if noisy_type != "clean" :
            noisy_dataset_labels,actual_noise_rate = noisify(train_labels=training_dataset.targets,
                                        noise_type=noisy_type, noise_rate=noisy_rate,
                                        random_state=0,
                                        classes=num_classes)
            for index in range(len(noisy_dataset_labels)):
                training_dataset.targets[index] = int(noisy_dataset_labels[index][0])
        local_datasets = split_dataset(train_dataset=training_dataset,test_dataset=test_dataset,class_num=10,client_num=num_clients,non_iid_alpha=non_iid_labels_rate,iid=iid,batch_size=batch_size) 
    elif dataset_name == "TINY-IMAGENET":
        num_classes = 200
        root_dir = 'data/tiny-imagenet-200'  
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        training_dataset = torchvision.datasets.ImageFolder(root=root_dir + '/train', transform=transform)
        test_dataset = torchvision.datasets.ImageFolder(root=root_dir + '/val/images', transform=transform)  
        if noisy_type != "clean" :
            noisy_dataset_labels,actual_noise_rate = noisify(train_labels=training_dataset.targets,
                                        noise_type=noisy_type, noise_rate=noisy_rate,
                                        random_state=0,
                                        classes=num_classes)
            for index in range(len(noisy_dataset_labels)):
                training_dataset.targets[index] = int(noisy_dataset_labels[index][0])
        local_datasets = split_dataset(train_dataset=training_dataset,test_dataset=test_dataset,class_num=200,client_num=num_clients,non_iid_alpha=non_iid_labels_rate,iid=iid,batch_size=batch_size) 
    elif hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            # transform = torchvision.transforms.Compose(
            #     [
            #         torchvision.transforms.ToTensor(),
            #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #     ]
            # )
            num_classes = 10
            train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
            training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=train_transform
            )
            test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
                transform=test_transform)
            if noisy_type != "clean" :
                noisy_dataset_labels,actual_noise_rate = noisify(train_labels=training_dataset.targets,
                                        noise_type=noisy_type, noise_rate=noisy_rate,
                                        random_state=0,
                                        classes=num_classes)
                for index in range(len(noisy_dataset_labels)):
                    training_dataset.targets[index] = int(noisy_dataset_labels[index][0])
            local_datasets = split_dataset(train_dataset=training_dataset,test_dataset=test_dataset,class_num=10,client_num=num_clients,non_iid_alpha=non_iid_labels_rate,iid=iid,batch_size=batch_size)  #remember to change the class_num
            
        elif dataset_name in ["MNIST"]:
            transform = torchvision.transforms.ToTensor()
        # prepare raw training & test datasets
            training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=train_transform
            )
            test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
                transform=test_transform)
            local_datasets = split_dataset(train_dataset=training_dataset,test_dataset=test_dataset,class_num=10,client_num=num_clients,non_iid_alpha=non_iid_labels_rate,iid=iid,batch_size=batch_size)
        elif dataset_name in ["CIFAR100"]:
            num_classes = 100
            mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            # train_transform = transforms.Compose([
            #     transforms.RandomCrop(size=32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=mean, std=std)])
            # test_transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=mean, std=std)])
            transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(96, antialias=True),
            torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
            # torchvision.transforms.RandomCrop(32, padding=4),  # 随机裁剪
            torchvision.transforms.ToTensor(),  # 转换为张量
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 标准化
            ])
            # prepare raw training & test datasets
            # transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            # ])
            training_dataset = torchvision.datasets.__dict__[dataset_name](
                root=data_path,
                train=True,
                download=True,
                transform=transform)
            test_dataset = torchvision.datasets.__dict__[dataset_name](
                root=data_path,
                train=False,
                download=True,
                transform=transform)
            if noisy_type != "clean" :
                noisy_dataset_labels,actual_noise_rate = noisify(train_labels=training_dataset.targets,
                                        noise_type=noisy_type, noise_rate=noisy_rate,
                                        random_state=0,
                                        classes=num_classes)
                for index in range(len(noisy_dataset_labels)):
                    training_dataset.targets[index] = int(noisy_dataset_labels[index][0])
            local_datasets = split_dataset(train_dataset=training_dataset,test_dataset=test_dataset,class_num=100,client_num=num_clients,non_iid_alpha=non_iid_labels_rate,iid=iid,batch_size=batch_size)
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    # if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
    #     training_dataset.data.unsqueeze_(3)
    # num_categories = np.unique(training_dataset.targets).shape[0]
    
    # if "ndarray" not in str(type(training_dataset.data)):
    #     training_dataset.data = np.asarray(training_dataset.data)
    # if "list" not in str(type(training_dataset.targets)):
    #     training_dataset.targets = training_dataset.targets.tolist()
    # if noisy_type != "clean" :
    #     noisy_dataset_labels,actual_noise_rate = noisify(train_labels=training_dataset.targets,
    #                                     noise_type=noisy_type, noise_rate=noisy_rate,
    #                                     random_state=0,
    #                                     classes=num_classes)
    # for index in range(len(noisy_dataset_labels)):
    #     training_dataset.targets[index] = int(noisy_dataset_labels[index][0])

    # split dataset according to iid flag
    train_dataset_labels = []
    for client_idx in range(len(local_datasets)):
        train_dataset_labels.append([])
        train_num = len(local_datasets[client_idx])
        adjust_num = int(train_num % 64)
        if adjust_num ==1:
            ad_num = int(train_num / 64)
            adjust_num = 64*adjust_num
            local_datasets[client_idx] = local_datasets[client_idx][int(0):adjust_num]
    for index in range(len(local_datasets[client_idx])):
        train_dataset_labels[client_idx].append(local_datasets[client_idx][index][1])

    return local_datasets, test_dataset

def noisify(train_labels,noise_type,noise_rate,random_state,classes):
    train_labels_cath = []
    for labels in train_labels:
        train_labels_cath.append([labels])
    train_labels = copy.deepcopy(train_labels_cath)
    train_labels = np.array(train_labels)
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=classes)
    if noise_type == 'None':
        train_noisy_labels = train_labels
        actual_noise_rate = 0
    return train_noisy_labels, actual_noise_rate

def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise


def split_dataset(train_dataset, test_dataset, class_num, client_num, non_iid_alpha, iid, batch_size):
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    if iid:
        partition_proportions = np.full(shape=(class_num, client_num), fill_value=1 / client_num)
    elif not iid:
        partition_proportions = np.random.dirichlet(alpha=np.full(shape=client_num, fill_value=non_iid_alpha),
                                                    size=class_num)

    client_train_datasets = split_dataset_by_proportion(train_dataset, partition_proportions, class_num, client_num,
                                                        iid, batch_size)
    return client_train_datasets



# def split_dataset(train_dataset, test_dataset, class_num, client_num, non_iid_alpha, iid, batch_size):
#     seed = 12345
#     random.seed(seed)
#     np.random.seed(seed)
#     if iid:
#         partition_proportions = np.full(shape=(class_num, client_num), fill_value=1 / client_num)
#     elif not iid:
#         partition_proportions = np.random.dirichlet(alpha=np.full(shape=client_num, fill_value=non_iid_alpha),
#                                                     size=class_num)

#     client_train_datasets = split_dataset_by_proportion(train_dataset, partition_proportions, class_num, client_num,
#                                                         iid, batch_size)
#     return client_train_datasets

def split_common_dataset(train_dataset, test_dataset, class_num, client_num, non_iid_alpha, iid, batch_size,common):
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    if iid:
        partition_proportions = np.full(shape=(class_num, client_num), fill_value=1 / client_num)
    elif not iid:
        partition_proportions = np.random.dirichlet(alpha=np.full(shape=client_num, fill_value=non_iid_alpha),
                                                    size=class_num)

    client_train_datasets,client_idcs = split_dataset_by_proportion_common(train_dataset, partition_proportions, class_num, client_num,
                                                        iid, batch_size,common)
    return client_train_datasets,client_idcs


# def split_dataset_by_proportion(dataset, partition_proportions, class_num, client_num, iid, batch_size):
#     seed = 12345
#     random.seed(seed)
#     np.random.seed(seed)
#     data_labels = dataset.targets
#     class_idcs = [list(np.argwhere(np.array(data_labels) == y).flatten())
#                   for y in range(class_num)]

#     client_idcs = [[] for _ in range(client_num)]
#     if iid == False:
#         for _ in range(batch_size):  # 每个客户端至少一个样本,至少让样本数和batch_size一样大，这样dataloader中的drop_last就不会让有的客户端没有数据
#             for client_id in range(client_num):
#                 class_id = np.random.randint(low=0, high=class_num)
#                 client_idcs[client_id].append(class_idcs[class_id].pop())

#     for c, fracs in zip(class_idcs, partition_proportions):
#         np.random.shuffle(c)
#         for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
#             client_idcs[i].extend(list(idcs))
#     client_data_list = [[] for _ in range(client_num)]
#     for client_id, client_data in zip(client_idcs, client_data_list):
#         for id in client_id:
#             client_data.append(dataset[id])

#     client_datasets = []
#     for client_data in client_data_list:
#         np.random.shuffle(client_data)
#         client_datasets.append(client_data)
#     return client_datasets

def split_dataset_by_proportion(dataset, partition_proportions, class_num, client_num, iid, batch_size):
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    data_labels = dataset.targets
    class_idcs = [list(np.argwhere(np.array(data_labels) == y).flatten())
                  for y in range(class_num)]
    client_idcs = [[] for _ in range(client_num)]
    if iid == False:
        for _ in range(batch_size):  # 每个客户端至少一个样本,至少让样本数和batch_size一样大，这样dataloader中的drop_last就不会让有的客户端没有数据
            for client_id in range(client_num):
                class_id = np.random.randint(low=0, high=class_num)
                client_idcs[client_id].append(class_idcs[class_id].pop())

    for c, fracs in zip(class_idcs, partition_proportions):
        np.random.shuffle(c)
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i].extend(list(idcs))
    client_data_list = [[] for _ in range(client_num)]
    for client_id, client_data in zip(client_idcs, client_data_list):
        for id in client_id:
            client_data.append(dataset[id])

    client_datasets = []
    for client_data in client_data_list:
        np.random.shuffle(client_data)
        client_datasets.append(client_data)
    return client_datasets

def split_dataset_by_proportion_common(dataset, partition_proportions, class_num, client_num, iid, batch_size,common):
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    data_labels = dataset.targets
    class_idcs = [list(np.argwhere(np.array(data_labels) == y).flatten())
                  for y in range(class_num)]

    client_idcs = [[] for _ in range(client_num)]
    if iid == False:
        for _ in range(batch_size):  # 每个客户端至少一个样本,至少让样本数和batch_size一样大，这样dataloader中的drop_last就不会让有的客户端没有数据
            for client_id in range(client_num):
                class_id = np.random.randint(low=0, high=class_num)
                client_idcs[client_id].append(class_idcs[class_id].pop())

    for c, fracs in zip(class_idcs, partition_proportions):
        np.random.shuffle(c)
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            if common:
                idcs = idcs[0:100]
            client_idcs[i].extend(list(idcs))
    client_data_list = [[] for _ in range(client_num)]
    for client_id, client_data in zip(client_idcs, client_data_list):
        for id in client_id:
            client_data.append(dataset[id])

    client_datasets = []
    for client_data in client_data_list:
        np.random.shuffle(client_data)
        client_datasets.append(client_data)
        client_commondatasets = copy.deepcopy(client_datasets)
    return client_commondatasets,client_idcs

