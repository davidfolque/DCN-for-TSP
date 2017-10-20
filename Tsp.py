


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_generator import Generator
from Split_GNN import Split_GNN
import os

'''
class Tsp():
    def __init__(self, batch_size, num_features, num_layers, J):
        self.batch_size = batch_size
        self.num_features = num_features,
        self.num_layers = num_layers
        self.J = J
        
        self.split = Split_GNN(batch_size, num_features, num_layers, J)
        
'''

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

def sample_one(pred, mode='train'):
    if mode == 'train':
        rand = torch.zeros(*pred.size()).type(dtype)
        nn.init.uniform(rand)
    else:
        rand = torch.ones_like(pred).type(dtype) / 2
    sample = (pred > rand).type(dtype)
    return sample

def compute_operators(W, e, J):
    # operators: {Id, W, W^2, ..., W^{J-1}, D, U}++
    # e[batch_index,j] is the id of the subgraph where node j is
    bs = e.size(0)
    n = W.size(-1)
    e_rows = e.unsqueeze(1).expand(bs, n, n)
    e_cols = e.unsqueeze(2).expand(bs, n, n)
    Phi = ((e_rows - e_cols) == 0).type(dtype)
    W = W * Phi
    QQ = W.clone()
    WW = torch.zeros(bs, n, n, J + 2)
    eye = torch.eye(n).type(dtype).unsqueeze(0).expand(bs,n,n)
    WW[:, :, :, 0] = eye
    for j in range(J):
        WW[:, :, :, j + 1] = QQ.clone()
        QQ = torch.bmm(QQ, QQ)
        mx = QQ.max(1)[0].unsqueeze(1).expand_as(QQ) * Phi
        mx = mx.max(2)[0].unsqueeze(2).expand_as(QQ) * Phi + (1-Phi)
        QQ = QQ / mx
        QQ *= np.sqrt(2)
    d = W.sum(1)
    D = d.unsqueeze(1).expand_as(eye) * eye
    WW[:, :, :, J] = D
    WW[:, :, :, J + 1] = Phi / Phi.sum(1).unsqueeze(1).expand_as(Phi)
    return WW, d


def load_model_tsp(parameters_path):
    path = os.path.join(parameters_path, 'parameters/gnn.pt')
    if os.path.exists(path):
        print('GNN successfully loaded from {}'.format(path))
        return torch.load(path)
    else:
        raise ValueError('Parameter path {} does not exist.'
                            .format(path))


if __name__ == '__main__':
    path_dataset = './dataset/broma/'
    gen = Generator(path_dataset, './LKH/')
    N = 20
    gen.num_examples_train = 200
    gen.num_examples_test = 10
    gen.N = N
    gen.load_dataset()
    
    grad_clip_split = 40.0
    iterations = 5000
    batch_size = 20
    num_features = 10
    num_layers = 5
    J = 4
    Split = Split_GNN(batch_size, num_features, num_layers, J+2, dim_input=3)
    
    optimizer = optim.RMSprop(Split.parameters())
    
    path_tsp = 
    tsp_gnn = load_model()
    
    for it in range(iterations):
        sample = gen.sample_batch(batch_size, cuda=torch.cuda.is_available())
        input, W, WTSP, labels, target, cities, perms, costs = extract(sample)
        scores, probs = Split(input)
        sample = sample_one(probs.data, mode='train')
        WW, x = compute_operators(W.data, sample, J)
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    '''W = torch.FloatTensor([[[
        4, 3, 0, 0 ],[ # 0---1 
        1, 2, 0, 0 ],[ #   X |
        0, 0, 1, 0 ],[ # 2   3
        0, 0, 9, 0 ]],
        [[
        0, 1, 0, 0 ],[ # 0---1
        1, 0, 0, 1 ],[ #     |
        0, 0, 2, 0 ],[ # 2---3
        0, 1, 0, 0]]])
    
    W = torch.FloatTensor([[[
        0, 1, 0, 1 ],[ # 0---1 
        1, 0, 1, 1 ],[ #   X |
        0, 1, 0, 0 ],[ # 2   3
        1, 1, 0, 0 ]],
        [[
        0, 1, 0, 0 ],[ # 0---1
        1, 0, 0, 1 ],[ #     |
        0, 0, 0, 1 ],[ # 2---3
        0, 1, 1, 0]]]).type(dtype)
    e = torch.FloatTensor([
        [1, 1, 2, 1],
        [1, 1, 1, 1]]).type(dtype)
    
    WW, x = compute_operators(W,e,3)
    print(WW[0,:,:,0])
    print(WW[0,:,:,1])
    print(WW[0,:,:,2])
    print(WW[0,:,:,3])
    print(WW[0,:,:,4])
    
    
    print(WW[1,:,:,0])
    print(WW[1,:,:,1])
    print(WW[1,:,:,2])
    print(WW[1,:,:,3])
    print(WW[1,:,:,4])
    
    exit()'''
    
    
    