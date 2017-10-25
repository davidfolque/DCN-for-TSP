
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import nou_utils as utils
import os
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import argparse

from data_generator import Generator
from model import Split_GNN, GNN
from nou_Logger import Logger

parser = argparse.ArgumentParser()


###############################################################################
#                             General Settings                                #
###############################################################################

parser.add_argument('--load', action='store_true')
parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(20000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(1000))
parser.add_argument('--iterations', nargs='?', const=1, type=int,
                    default=int(10e6))
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--beam_size', nargs='?', const=1, type=int, default=2)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
parser.add_argument('--path_dataset', nargs='?', const=1, type=str, default='./dataset/')
parser.add_argument('--path_load_split', nargs='?', const=1, type=str, default='')
parser.add_argument('--path_load_tsp', nargs='?', const=1, type=str, default='')
parser.add_argument('--path_load_merge', nargs='?', const=1, type=str, default='')
parser.add_argument('--path_save_model', nargs='?', const=1, type=str, default='/saved_model/')
parser.add_argument('--path_logger', nargs='?', const=1, type=str, default='')
parser.add_argument('--path_tsp', nargs='?', const=1, type=str, default='')
parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=50)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
parser.add_argument('--N', nargs='?', const=1, type=int, default=20)

args = parser.parse_args()


CEL = nn.CrossEntropyLoss()
BCE = nn.BCELoss()
template_train1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} '
template_train2 = ('{:<10} {:<10} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.3f} \n')
template_test1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'
template_test2 = '{:<10} {:<10} {:<10.5f} {:<10.5f} {:<10.5f} {:<10} {:<10.3f} \n'
info_train = ['TRAIN', 'iteration', 'loss_split', 'loss_merge', 'cost', 'accuracy', 'elapsed']
info_test = ['TEST', 'iteration', 'loss_split', 'loss_merge', 'accuracy', 'beam_size', 'elapsed']

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

def extract(sample):
    input = sample[0]
    W = sample[0][0][:, :, :, 1]
    WTSP, labels = sample[1][0].type(dtype_l), sample[1][1].type(dtype_l)
    target = WTSP, labels
    cities = sample[2]
    perms = sample[3]
    costs = sample[4]
    return input, W, WTSP, labels, target, cities, perms, costs


class DCN():
    def __init__(self, batch_size, num_features, num_layers, J, dim_input, clip_grad_norm, logger):
        self.logger = logger
        self.clip_grad_norm = clip_grad_norm
        self.batch_size = batch_size
        self.J = J
        self.Split = Split_GNN(batch_size, num_features, num_layers, J+2, dim_input=dim_input)
        self.Tsp = GNN(num_features, num_layers, J+2, dim_input=dim_input)
        self.Merge = GNN(num_features, num_layers, J+2, dim_input=dim_input)
        self.optimizer_split = optim.RMSprop(self.Split.parameters())
        self.optimizer_tsp = optim.Adamax(self.Tsp.parameters(), lr=1e-3)
        self.optimizer_merge = optim.Adamax(self.Merge.parameters(), lr=1e-3)
    
    def load_split(self, path_load):
        self.Split = self.logger.load_model(path_load, 'split')
    
    def load_tsp(self, path_load):
        self.Split = self.logger.load_model(path_load, 'tsp')
    
    def load_merge(self, path_load):
        self.Split = self.logger.load_model(path_load, 'merge')
    
    def save_model(self, path_load):
        self.logger.save_model(path_load, self.Split, self.Tsp, self.Merge)
    
    def set_dataset(self, path_dataset, num_examples_train, num_examples_test, N_train, N_test):
        self.gen = Generator(path_dataset, args.path_tsp)
        self.gen.num_examples_train = num_examples_train
        self.gen.num_examples_test = num_examples_test
        self.gen.N_train = N_train
        self.gen.N_test = N_test
        self.gen.load_dataset()
    
    def sample_one(self, probs, mode='train'):
        probs = 1e-4 + probs*(1 - 2e-4) # to avoid log(0)
        if mode == 'train':
            rand = torch.zeros(*probs.size()).type(dtype)
            nn.init.uniform(rand)
        else:
            rand = torch.ones(*probs.size()).type(dtype) / 2
        sample = (probs > Variable(rand)).type(dtype)
        log_probs_samples = (sample*torch.log(probs) + (1-sample)*torch.log(1-probs)).sum(1)
        return sample.data, log_probs_samples
    
    def compute_operators(self, W, e, J):
        # operators: {Id, W, W^2, ..., W^{J-1}, D, U}++
        # e[batch_index,j] is the id of the subgraph where node j is
        bs = e.size(0)
        n = W.size(-1)
        e_rows = e.unsqueeze(1).expand(bs, n, n)
        e_cols = e.unsqueeze(2).expand(bs, n, n)
        Phi = ((e_rows - e_cols) == 0).type(dtype)
        W = W * Phi
        QQ = W.clone()
        WW = torch.zeros(bs, n, n, J + 2).type(dtype)
        eye = torch.eye(n).type(dtype).unsqueeze(0).expand(bs,n,n)
        WW[:, :, :, 0] = eye
        for j in range(J):
            WW[:, :, :, j + 1] = QQ.clone()
            QQ = torch.bmm(QQ, QQ)
            mx = QQ.max(1)[0].unsqueeze(1).expand_as(QQ) * Phi
            mx = mx.max(2)[0].unsqueeze(2).expand_as(QQ) * Phi + (1-Phi)
            QQ = QQ / torch.clamp(mx, min=1e-6)
            QQ *= np.sqrt(2)
        d = W.sum(1)
        D = d.unsqueeze(1).expand_as(eye) * eye
        WW[:, :, :, J] = D
        WW[:, :, :, J + 1] = Phi / Phi.sum(1).unsqueeze(1).expand_as(Phi)
        return WW, d, Phi
    
    def forward(self, input, W, cities):
        scores, probs = self.Split(input)
        #variance = compute_variance(probs)
        sample, log_probs_samples = self.sample_one(probs, mode='train')
        WW, x, Phi = self.compute_operators(W.data, sample, self.J)
        x = torch.cat((x.unsqueeze(2),cities),2)
        y = WW[:,:,:,1]
        WW = Variable(WW).type(dtype)
        x = Variable(x).type(dtype)
        y = Variable(y).type(dtype)
        partial_pred = self.Tsp((WW,x,y))
        partial_pred = partial_pred * Variable(Phi)
        pred = self.Merge((input[0], input[1], partial_pred))
        return probs, log_probs_samples, pred
    
    def compute_loss(self, pred, target, logprobs):
        loss_split = 0.0
        loss_merge = 0.0
        labels = target[1]
        for i in range(labels.size()[-1]):
            for j in range(labels.size()[0]):
                lab = labels[j, :, i].contiguous().view(-1)
                cel = CEL(pred[j], lab)
                loss_merge += cel
                loss_split += Variable(cel.data) * logprobs[j]
        return loss_merge/pred.size(0), loss_split/pred.size(0)
    
    def train(self, iterations, print_freq, test_freq, save_freq, path_model):
        for it in range(iterations):
            start = time.time()
            batch = self.gen.sample_batch(self.batch_size, cuda=torch.cuda.is_available())
            input, W, WTSP, labels, target, cities, perms, costs = extract(batch)
            probs, log_probs_samples, pred = self.forward(input, W, cities)
            loss_merge, loss_split = self.compute_loss(pred, target, log_probs_samples)
            #loss_split -= variance*rf
            self.Split.zero_grad()
            loss_split.backward()
            nn.utils.clip_grad_norm(self.Split.parameters(), self.clip_grad_norm)
            self.optimizer_split.step()
            self.Tsp.zero_grad()
            self.Merge.zero_grad()
            loss_merge.backward()
            nn.utils.clip_grad_norm(self.Tsp.parameters(), clip_grad)
            nn.utils.clip_grad_norm(self.Merge.parameters(), clip_grad)
            self.optimizer_tsp.step()
            self.optimizer_merge.step()
            
            self.logger.add_train_loss(loss_split, loss_merge)
            self.logger.add_train_accuracy(pred, labels, W)
            elapsed = time.time() - start
            
            if it%print_freq == 0 and it > 0:
                loss_split = loss_split.data.cpu().numpy()[0]
                loss_merge = loss_merge.data.cpu().numpy()[0]
                out = ['---', it, loss_split, loss_merge, self.logger.cost_train[-1],
                       self.logger.accuracy_train[-1], elapsed]
                print(template_train1.format(*info_train))
                print(template_train2.format(*out))
                #print(variance)
                #print(probs[0])
                #plot_clusters(it, probs[0], cities[0])
                #os.system('eog ./plots/clustering/clustering_it_{}.png'.format(it))
            if it%test_freq == 0 and it > 0:
                self.test()
            if it%save_freq == 0 and it > 0:
                self.save_model(path_model)
    
    def test(self):
        iterations_test = int(self.gen.num_examples_test / self.batch_size)
        for it in range(iterations_test):
            start = time.time()
            batch = gen.sample_batch(batch_size, is_training=False, it=it,
                                    cuda=torch.cuda.is_available())
            input, W, WTSP, labels, target, cities, perms, costs = extract(batch)
            probs, log_probs_samples, pred = self.forward(input, W, cities)
            loss_merge, loss_split = self.compute_loss(pred, target, log_probs_samples)
            #loss_split -= variance*rf
            last = (it == iterations_test-1)
            self.logger.add_test_accuracy(pred, labels, perms, W, cities, costs,
                                    last=last, beam_size=beam_size)
            self.logger.add_test_loss(loss_split, loss_merge, last=last)
            elapsed = time.time() - start
            '''if not last and it % 100 == 0:
                loss = loss.data.cpu().numpy()[0]
                out = ['---', it, loss, logger.accuracy_test_aux[-1], 
                    logger.cost_test_aux[-1], beam_size, elapsed]
                print(template_test1.format(*info_test))
                print(template_test2.format(*out))'''
        print('TEST COST: {} | TEST ACCURACY {}\n'
            .format(self.logger.cost_test[-1], self.logger.accuracy_test[-1]))




if __name__ == '__main__':
    
    N_train = 20
    N_test = 40
    
    num_examples_train = 20000
    num_examples_test = 1000
    clip_grad = 40.0
    iterations = 50000
    batch_size = 20
    num_features = 10
    num_layers = 5
    J = 4
    rf = 10.0 # regularization factor
    beam_size = 20
    
    logger = Logger('./logs')
    Dcn = DCN(batch_size, num_features, num_layers, J, 3, args.clip_grad_norm, logger)
    Dcn.set_dataset(args.path_dataset, num_examples_train, num_examples_test, N_train, N_test)
    Dcn.load_split(args.path_load_split)
    Dcn.load_tsp(args.path_load_tsp)
    Dcn.train(args.iterations, args.print_freq, args.test_freq, args.save_freq, args.path_save_model)
    















