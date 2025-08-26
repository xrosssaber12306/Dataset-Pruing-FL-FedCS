import gc
import pickle
import logging
import copy
from sklearn.cluster import KMeans
import torchvision.models._utils
from torchvision.models import feature_extraction
from scipy import stats
import numpy as np
import random
import torchvision
import cv2 as cv
from matplotlib import pyplot as plt
from collections import OrderedDict
from torch.distributed import ProcessGroup
from sklearn.decomposition import PCA

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device,non_iid_labels_rate):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None
        self.feature = OrderedDict()
        self.pruning_flag = True
        self.num = 0
        self.first_pruning_flag_first = True
        self.labels_num_collection = []
        self.class_center = OrderedDict()
        self.class_center_cath = OrderedDict()
        self.pruned_dataloader = None
        self.local_score = []
        self.acc = []
        self.quantity = None
        self.dyn_k =[]

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=False)
        self.batch_size = client_config["batch_size"]
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        self.non_iid_labels_rate = client_config["non_iid_labels_rate"]
        self.class_num = client_config["class_num"]
        self.pruning_data_location = []
        self.labels_need_pruning = []
        self.first_pruning_flag = True
        self.score = []
        self.chanels_selected = dict()
        self.uncerten_k=[]


    def client_update(self,optim_config):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)
        self.num = 0
        self.optim_config = optim_config
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            # next_pruning_data = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,'others':0}
            for data, labels in self.pruned_dataloader:
                data_index =0
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.num = self.num + len(labels)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step() 
                data_index = data_index+1

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        if self.first_pruning_flag:
            self.first_pruning_flag = False

            
    def distance_different_pre_train_client(self,epoach_flag,feature_keys,optim_config):
        self.model.train()
        self.model.to(self.device)
        self.optim_config = optim_config
        # self.pruning_label_select(beta)

        index_num = 0
        data_num = 0
        feature_dict = OrderedDict()
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)

        for data,labels in self.dataloader:
            data_num = data_num+len(labels)
            # new_model = torchvision.models._utils.IntermediateLayerGetter(self.model,{'layer4.0.conv1':'1'})
            data, labels = data.float().to(self.device), labels.long().to(self.device)
            optimizer.zero_grad()
            outputs = self.model(data)

            loss = eval(self.criterion)()(outputs, labels)
            loss.backward()
            optimizer.step()
            index_num = index_num + 1
            #if self.device == "cuda":
            torch.cuda.empty_cache()
        self.model.to("cpu")
        # del feature_extractor
        return self.score
    

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy

        
    def get_pruning_location(self,pruning_rate_f,pruning_rate_l):
        before_pruning_record = dict() #Save the position before pruning.
        will_pruning_score = []
        #Extract the pruning scores of the target dataset into a tensor and save the corresponding images with their scores.
        for label in self.labels_need_pruning:
            before_pruning_record = dict()
            out_batch_index = 0
            will_pruning_score = []
            pruning_data_location = []
            for data,labels in self.dataloader:
                for in_batch_index in range(len(labels)):
                    if labels[in_batch_index] == label:
                        will_pruning_score.append(self.score[out_batch_index][in_batch_index])
                        before_pruning_record[self.score[out_batch_index][in_batch_index]] = [out_batch_index,in_batch_index]
                out_batch_index = out_batch_index + 1
            after_pruning_scores1 = self.topk(will_pruning_score,pruning_rate_f)#aum:same to EL2N
            for index in range(len(will_pruning_score)):
                # if will_pruning_score[index] != after_pruning_scores1[index] and will_pruning_score[index]==after_pruning_scores2[index]:
                if will_pruning_score[index] == after_pruning_scores1[index]:
                    self.pruning_data_location.append(before_pruning_record[will_pruning_score[int(index)]])
                    self.score[before_pruning_record[will_pruning_score[int(index)]][0]][before_pruning_record[will_pruning_score[int(index)]][1]] = torch.tensor(0)
        before_pruning_record = dict()
        will_pruning_score = []
        pruning_data_location = []
        out_batch_index = 0
        for data,labels in self.dataloader:
            for in_batch_index in range(len(labels)):
                if type(self.score[out_batch_index][in_batch_index]) != torch.Tensor: 
                    will_pruning_score.append(self.score[out_batch_index][in_batch_index])
                    before_pruning_record[self.score[out_batch_index][in_batch_index]] = [out_batch_index,in_batch_index]
            out_batch_index = out_batch_index + 1
        after_pruning_scores = self.topk(will_pruning_score,pruning_rate_l)
        for index in range(len(will_pruning_score)):
            if will_pruning_score[index] == after_pruning_scores[index]:
                self.pruning_data_location.append(before_pruning_record[will_pruning_score[int(index)]])


    def pruning_label_select(self,beta):       
        labels_num_list = [0 for _ in range(self.class_num)]
        labels_need_pruning = []
        labels_num_dict = dict()
        labels_num_cath = []
        for data, labels in self.dataloader:
            for labels_index in range(len(labels)):
                labels_num_list[labels[labels_index]] = labels_num_list[labels[labels_index]] + 1
        self.labels_num_collection = copy.deepcopy(labels_num_list)
        labels_num_cath = copy.deepcopy(labels_num_list)
        for index in range(len(labels_num_cath)):
            labels_num_dict[labels_num_cath[index]] = index
        labels_num_cath.sort(reverse=True)
        for index in range(int(1),int(len(labels_num_cath))):
            labels_difference = abs(labels_num_cath[0] - labels_num_list[index])
            label_flag = labels_difference/labels_num_cath[0]
            if label_flag>beta:
                labels_num_list[index] = int(0)
        for index in range(len(labels_num_cath)):
            if labels_num_list[index] != 0:
                labels_need_pruning.append(index)
        self.labels_need_pruning = copy.deepcopy(labels_need_pruning)
        del labels_num_cath,labels_need_pruning,labels_num_dict,labels_num_list

    def topk(self,list,rate):
        list_tensor = torch.tensor(list)
        index = torch.argsort(torch.abs(list_tensor), descending=False)[0: math.floor(len(list) *(1 -rate))]
        com_list = list_tensor.index_fill(0, index, 0)
        return com_list
  
        
    def sortk(self,list,rate):
        list_cath = copy.deepcopy(list)
        list_sort = copy.deepcopy(list)
        list_sort.sort()
        judgement = list_sort[int((1-rate)*len(list_sort))]
        for index in range(len(list_sort)):
            if list_cath[index] < judgement:
                list_cath[index] = 0
        com_list = torch.tensor(list_cath)
        del list_cath,list_sort
        return com_list



    def pruning(self,pruning_method,pruning_rate_f,pruning_rate_l,beta,batch_size):
        self.pruning_label_select(beta)
        self.get_pruning_location(pruning_rate_f,pruning_rate_l)
        self.num = 0
        data_pruned = []
        outside_index = 0
        data_index = 0
        for data,labels in self.dataloader:
            for in_index in range(len(labels)):
                if [data_index,in_index] not in self.pruning_data_location:
                    data_pruned.append((data[in_index],labels[in_index]))
            outside_index = outside_index + 1
            data_index = data_index + 1
        train_num = len(data_pruned)
        if train_num % batch_size == 1:
            data_pruned = data_pruned[0:train_num-1]
        self.pruned_dataloader = DataLoader(data_pruned, batch_size=self.batch_size, shuffle=True)
        del data_pruned,self.score



    def distance_center_without_center(self,beta):
        self.model.train()
        self.model.to(self.device)
        self.pruning_label_select(beta)
        index_num = 0
        feature_class_dict = dict()
        p = []
        feature_dict = OrderedDict()
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        nodes,_ = feature_extraction.get_graph_node_names(self.model)
        feature_keys = nodes[68:69]
        feature_extractor = feature_extraction.create_feature_extractor(self.model,return_nodes=feature_keys)
        for keys in feature_keys:
            feature_dict[keys] = []
            self.feature[keys] = []
        for data,labels in self.dataloader:
            data, labels = data.float().to(self.device), labels.long().to(self.device)
            optimizer.zero_grad()
            feature_information = feature_extractor(data)
        #avg_center
            for keys in feature_information.keys():
                if index_num == 0:
                    self.class_center[keys] = []
                    for index in range(len(self.labels_num_collection)):
                        self.class_center[keys].append(int(0)) 
                self.feature[keys].append([])
                feature_dict[keys].append([])
                feature_dict[keys][index_num] = feature_information[keys].cpu().detach().numpy()
                self.feature[keys][index_num] = feature_dict[keys][index_num]
                for inside_index in range(len(labels)):
                    for keys in self.feature.keys():
                        #for chanel_index in range(len(self.chanels_selected[keys])):
                        if type(self.class_center[keys][int(labels[inside_index])]) != 'int' :
                            self.class_center[keys][int(labels[inside_index])] = 1/self.labels_num_collection[int(labels[inside_index])]*self.feature[keys][index_num][inside_index] + self.class_center[keys][int(labels[inside_index])]
                        else:
                            self.class_center[keys][int(labels[inside_index])] = 1/self.labels_num_collection[int(labels[inside_index])]*self.feature[keys][index_num][inside_index]
                index_num = index_num + 1
        for keys in self.feature.keys():
            del self.feature[keys],feature_dict
            self.feature[keys] = []

    def distance_different_score_without_center(self,class_center,beta):
        self.model.train()
        self.model.to(self.device)
        self.pruning_label_select(beta)
        index_num = 0
        feature_class_dict = dict()
        p = []
        feature_dict = OrderedDict()
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        nodes,_ = feature_extraction.get_graph_node_names(self.model)
        feature_keys = nodes[68:69]
        feature_extractor = feature_extraction.create_feature_extractor(self.model,return_nodes=feature_keys)
        for keys in feature_keys:
            feature_dict[keys] = []
            self.feature[keys] = []
        for data,labels in self.dataloader:
            data, labels = data.float().to(self.device), labels.long().to(self.device)
            optimizer.zero_grad()
            feature_information = feature_extractor(data)
            for keys in feature_information.keys():
                self.feature[keys].append([])
                feature_dict[keys].append([])
                feature_dict[keys][index_num] = feature_information[keys].cpu().detach().numpy()
                self.feature[keys][index_num] = feature_dict[keys][index_num]
            index_num = index_num + 1
        for keys in self.feature.keys():
                # detemine_mean = self.feature[keys][0][0][0].shape[0]
                # scores_weight = 1/detemine_mean
            out_side_index = 0
            scores_weight = 1
            if keys == 'avgpool' or keys == 'maxpool':
                scores_weight = 4*scores_weight
            for data,labels in self.dataloader:
                if keys == next(iter(self.feature)):
                    self.score.append([])
                    self.local_score.append([])
                for in_side_index in range(len(labels)):
                    dists_list = []
                    local_dists_list = []
                    dist_dict = dict()
                    local_dist_dict = dict()
                    for index in range(len(class_center[keys])):
                        if type(self.class_center[keys][index]) != int:
                            local_dists_cath = np.linalg.norm(self.feature[keys][out_side_index][in_side_index]-self.class_center[keys][index],2)
                        else:
                            local_dists_cath = 99999999999
                        dists = np.linalg.norm(self.feature[keys][out_side_index][in_side_index]-class_center[keys][index],2)
                        local_dists_list.append(local_dists_cath)
                        dists_list.append(dists)
                        dist_dict[dists] = index
                        local_dist_dict[local_dists_cath] = index
                    dist_cath = copy.deepcopy(dists_list)
                    local_dists_list_cath = copy.deepcopy(local_dists_list)
                    dist_cath.pop(labels[in_side_index])
                    local_dists_list_cath.pop(labels[in_side_index])
                    sorted_dist = sorted(dist_cath)
                    local_sorted_dist = sorted(local_dists_list_cath)
                    scores_cath = sorted_dist[0] - dists_list[labels[in_side_index]]
                    local_score_cath = local_sorted_dist[0] - local_dists_list[labels[in_side_index]]
                    scores = scores_weight*scores_cath
                    local_score_cath = local_score_cath*scores_weight
                    if keys == next(iter(self.feature)):
                        self.score[out_side_index].append(scores)
                        self.local_score[out_side_index].append(local_score_cath)
                    else:
                        self.score[out_side_index][in_side_index] = self.score[out_side_index][in_side_index] + scores
                        self.local_score[out_side_index][in_side_index] = self.local_score[out_side_index][in_side_index] + local_score_cath
                out_side_index = out_side_index + 1
        del feature_keys,feature_extractor,feature_dict


