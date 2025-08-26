import copy
import gc
import logging

import numpy as np
import torch
import timm
import torch.nn as nn
import cv2 as cv
import copy,math
from transformers import AutoImageProcessor, AutoModelForImageClassification

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict
from torchvision.models import resnet,mobilenet,vgg11,vgg13,resnet50,vit_b_16,ViT_B_16_Weights
from .feature_extraction import get_graph_node_names,create_feature_extractor
from torchvision import models
from .models import *
from .utils import *
from .client import Client

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        #self.model = eval(model_config["name"])(**model_config)
        self.scores = []
        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]
        self.pre_flag = True
        self.normalized_quantity =  None

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]
        self.non_iid_labels_rate = data_config["non_iid_labels_rate"]
        self.pruning_score_rule = data_config["pruning_score_rule"]
        self.feature_keys = data_config["feature_keys"]
        self.class_num = data_config["num_classes"]
        self.quantity = []
        self.remain_clients = []

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]
        self.pre_rounds = fed_config["pre_rounds"]
        self.pre_interval_round = fed_config["pre_interval_round"]
        self.noisy_type = fed_config["noisy_type"]
        self.noisy_rate = fed_config["noisy_rate"]
        self.score_round = fed_config["score_round"]

        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config
        self.model_name = model_config["name"]
        self.pruning_rate_f = data_config["pruning_rate_f"]
        self.pruning_rate_l = data_config["pruning_rate_l"]
        self.beta = data_config["beta"]
        if self.model_name == "resnet18":
            self.model = resnet.resnet18()
            # self.model.load_state_dict(torch.load('weight/resnet18-f37072fd.pth'))
            # model_avqpool = torch.nn.AdaptiveAvgPool2d(1)
            # num_ftrs = self.model.fc.in_features
            # self.model.fc = nn.Linear(num_ftrs,self.class_num)
        if self.model_name == "mobilenetv2":
            self.model = mobilenet.mobilenet_v2(num_classes=self.class_num)
        if self.model_name =="VGG11":
            self.model = vgg11(num_classes=self.class_num)
        if self.model_name =="VGG13":
            self.model = vgg13(num_classes=self.class_num) 
        if self.model_name == "vit_b_16":
            processor = AutoImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
            self.model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        if self.model_name == "vit_tiny":
            pre_path = 'weight/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'
            self.model = timm.create_model('vit_tiny_patch16_224', num_classes=self.class_num, img_size=96,
                              pretrained=True,pretrained_cfg_overlay=dict(file=pre_path))
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        seed = self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if self.model_name == "resnet18":
            self.model = resnet.resnet18(num_classes=self.class_num)
            # self.model.load_state_dict(torch.load('weight/resnet18-f37072fd.pth'))
            # model_avqpool = torch.nn.AdaptiveAvgPool2d(1)
            # num_ftrs = self.model.fc.in_features
            # self.model.fc = nn.Linear(num_ftrs,self.class_num)
        if self.model_name == "mobilenetv2":
            self.model = mobilenet.mobilenet_v2(num_classes=self.class_num)
        if self.model_name =="VGG11":
            self.model = vgg11(num_classes=self.class_num)
        if self.model_name =="VGG13":
            self.model = vgg13(num_classes=self.class_num) 
        if self.model_name =="resnet50":
            self.model = resnet50() 
            self.model.load_state_dict(torch.load('weight/resnet50-19c8e357.pth'))
            model_avqpool = torch.nn.AdaptiveAvgPool2d(1)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.class_num)
        # init_net(self.model, **self.init_config)
        if self.model_name == "vit_b_16":
            self.model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")
            # self.model.load_state_dict(torch.load('weight/vit_b_16-c867db91.pth'))
            # self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.class_num)
        if self.model_name == "vit_tiny":
            pre_path = 'weight/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'
            self.model = timm.create_model('vit_tiny_patch16_224', num_classes=self.class_num, img_size=96,
                              pretrained=True,pretrained_cfg_overlay=dict(file=pre_path))
        self.pre_model = resnet.resnet18()
        # self.pre_model.load_state_dict(torch.load('weight/resnet50-19c8e357.pth'))
        pre_model_avqpool = torch.nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.pre_model.fc.in_features
        self.pre_model.fc = nn.Linear(num_ftrs,self.class_num)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid,self.noisy_type,self.noisy_rate,self.non_iid_labels_rate,self.batch_size)
        
        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config,non_iid_labels_rate = self.non_iid_labels_rate,class_num = self.class_num
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
        test_loss, test_accuracy = self.evaluate_global_model()
        
    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device,non_iid_labels_rate=self.non_iid_labels_rate)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):

        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            #assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            #assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def pre_transmit_model(self, sampled_client_indices=None):
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            #assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.pre_model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            #assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.pre_model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())
        # sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, p=self.normalized_quantity, replace=False).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update(optim_config=self.optim_config)
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update(optim_config=self.optim_config)
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()

    def pre_average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.pre_model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.pre_model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        self.pre_flag = False
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        if self.mp_flag:
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
            print(message); logging.info(message)
            del message; gc.collect()

            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        else:
            self.evaluate_selected_models(sampled_client_indices)
        selected_total_size = 0
        for index in sampled_client_indices:
            selected_total_size = selected_total_size + self.clients[index].num

         # calculate averaging coefficient of weights
        mixing_coefficients = [self.clients[idx].num / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
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
        return test_loss, test_accuracy

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        # model_avqpool = torch.nn.AdaptiveAvgPool2d(1)
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs,self.class_num)
        self.transmit_model()
        test_loss, test_accuracy = self.evaluate_global_model()   
        self.results['loss'].append(test_loss)
        self.results['accuracy'].append(test_accuracy)
        self._round = 1
        self.writer.add_scalars(
            'Loss',
            {f"[{self.dataset_name}]_{self.model_name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid},noniidrate_{self.non_iid_labels_rate},method_{self.pruning_score_rule}": test_loss},
            self._round
            )
        self.writer.add_scalars(
            'Accuracy', 
            {f"[{self.dataset_name}]_{self.model_name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid},noniidrate_{self.non_iid_labels_rate},method_{self.pruning_score_rule}": test_accuracy},
            self._round
            )

        for r in range(self.num_rounds):
    # display and log experiment configuration
            # for index in range(self.num_clients):
            #     self.clients[index].optim_config['lr'] = self.cosine_decay_with_warmup(global_step=r,total_steps=self.num_rounds)
            if r%self.pre_interval_round == 0:
                self.distance_different_pruning_without_center()
                for index in range(self.num_clients):
                    self.clients[index].pruning(self.pruning_score_rule,self.pruning_rate_f,self.pruning_rate_l,self.beta,self.batch_size)
            self._round = r + 1 + int(self.pre_rounds/self.local_epochs)
            for index in range(self.num_clients):
                self.clients[index].optim_config['lr'] = self.cosine_decay_with_warmup(global_step=r,total_steps=self.num_rounds)
            self.train_federated_model()
            test_loss, test_accuracy = self.evaluate_global_model()
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)

            self.writer.add_scalars(
                'Loss',
                {f"[{self.dataset_name}]_{self.model_name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid},noniidrate_{self.non_iid_labels_rate},method_{self.pruning_score_rule}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Accuracy', 
                {f"[{self.dataset_name}]_{self.model_name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid},noniidrate_{self.non_iid_labels_rate},method_{self.pruning_score_rule}": test_accuracy},
                self._round
                )

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()


    
    def cosine_decay_with_warmup(self,global_step,
                                total_steps,
                                learning_rate_base=0.0125,
                                warmup_learning_rate=0.0025,
                                warmup_steps=20,
                                hold_base_rate_steps=0):
        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
        learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
            (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
        if hold_base_rate_steps > 0:
            learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                    learning_rate)
        return np.where(global_step > total_steps, 0.0, learning_rate)
                    

    def distance_different_pruning_without_center(self):
        self.pre_flag = True
        data_score_cath = []
        feature = dict()
        class_center = dict()
        feature_class_dict = dict()
        sample_client_num = [i for i in range(self.num_clients)]
        for pre_epoach in range(self.pre_rounds):
            for index in range(self.num_clients):
                self.clients[index].optim_config['lr'] = self.cosine_decay_with_warmup(global_step=pre_epoach,total_steps=self.num_rounds)
            selected_total_size = 0
            if pre_epoach ==0:
                data_score_list = []
            for c_idx in range(self.num_clients):
                if pre_epoach == 0:
                    self.clients[c_idx].first_pruning_flag = True
                    selected_total_size += len(self.clients[c_idx])
                    mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sample_client_num]
                message = f"[Round: {str(pre_epoach).zfill(4)}] Start updating selected client {str(self.clients[c_idx].id).zfill(4)}...!"
                print(message, flush=True); logging.info(message)
                del message; gc.collect()
                self.clients[c_idx].distance_different_pre_train_client(pre_epoach,self.feature_keys,self.optim_config)
            if int(pre_epoach+1)%self.local_epochs == 0 and pre_epoach>0 :
                self.average_model(sampled_client_indices=sample_client_num,coefficients=mixing_coefficients)
                self.transmit_model(sample_client_num)
                test_loss, test_accuracy = self.evaluate_global_model()
            
                self.results['loss'].append(test_loss)
                self.results['accuracy'].append(test_accuracy)
                self._round = int((pre_epoach+1)/self.local_epochs) + 1 
                self.writer.add_scalars(
                    'Loss',
                    {f"[{self.dataset_name}]_{self.model_name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid},noniidrate_{self.non_iid_labels_rate},method_{self.pruning_score_rule}": test_loss},
                    self._round
                    )
                self.writer.add_scalars(
                    'Accuracy', 
                    {f"[{self.dataset_name}]_{self.model_name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid},noniidrate_{self.non_iid_labels_rate},method_{self.pruning_score_rule}": test_accuracy},
                    self._round
                    )

                message = f"[pre_Round: {str(pre_epoach).zfill(4)}] Evaluate global model's performance...!\
                    \n\t[Server] ...finished evaluation!\
                    \n\t=> Loss: {test_loss:.4f}\
                    \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
                print(message); logging.info(message)
                del message; gc.collect()
        for c_idx in range(self.num_clients):
            self.clients[c_idx].distance_center_without_center(self.beta)
            for keys in self.clients[c_idx].class_center.keys():
                if c_idx == 0:
                    class_center[keys] = []
                    feature_class_dict[keys] = []
                for class_index in range(self.class_num):
                    if c_idx == 0:
                        class_center[keys].append([])
                        feature_class_dict[keys].append([])
                    if type(self.clients[c_idx].class_center[keys][class_index]) != int:
                        feature_class_dict[keys][class_index].append(self.clients[c_idx].class_center[keys][class_index])
        for keys in feature_class_dict.keys():
            for class_index in range(len(feature_class_dict[keys])):
                feature_class_dict[keys][class_index] = np.array(feature_class_dict[keys][class_index])
                shape = feature_class_dict[keys][class_index][0].shape
                feature_class_dict[keys][class_index] = feature_class_dict[keys][class_index].reshape(len(feature_class_dict[keys][class_index]),-1)
                class_center[keys][class_index] = np.median(feature_class_dict[keys][class_index],axis=0)
                class_center[keys][class_index] = class_center[keys][class_index].reshape(shape)
        for c_idx in range(self.num_clients):
            self.clients[c_idx].distance_different_score_without_center(class_center,self.beta)
        del class_center

    def normalize_list(self,values):
        total = sum(values)
        if total == 0:
            return [0 for _ in values]
        return [value / total for value in values]
