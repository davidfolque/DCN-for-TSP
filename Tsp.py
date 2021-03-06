


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from data_generator import Generator
from model import Split_GNN, GNN
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


CEL = nn.CrossEntropyLoss()
BCE = nn.BCELoss()
template = '{:<10} {:<10.5f} {:<10.5f} {:<10.5f} '



def extract(sample):
    input = sample[0]
    W = sample[0][0][:, :, :, 1]
    WTSP, labels = sample[1][0].type(dtype_l), sample[1][1].type(dtype_l)
    target = WTSP, labels
    cities = sample[2]
    perms = sample[3]
    costs = sample[4]
    return input, W, WTSP, labels, target, cities, perms, costs

def sample_one(probs, mode='train'):
    probs = 1e-4 + probs*(1 - 2e-4) # to avoid log(0)
    if mode == 'train':
        rand = torch.zeros(*probs.size()).type(dtype)
        nn.init.uniform(rand)
    else:
        rand = torch.ones_like(probs).type(dtype) / 2
    sample = (probs > Variable(rand)).type(dtype)
    log_probs_samples = (sample*torch.log(probs) + (1-sample)*torch.log(1-probs)).sum(1)
    return sample.data, log_probs_samples

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
    WW = torch.zeros(bs, n, n, J + 2).type(dtype)
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
    return WW, d, Phi


def load_model_tsp(parameters_path):
    path = os.path.join(parameters_path, 'parameters/gnn.pt')
    if os.path.exists(path):
        print('GNN successfully loaded from {}'.format(path))
        return torch.load(path)
    else:
        raise ValueError('Parameter path {} does not exist.'
                            .format(path))

def compute_loss(pred, target, logprobs):
    loss_supervised = 0.0
    loss_reinforce = 0.0
    cross_entropy = True
    if cross_entropy:
        #pred = pred.view(-1, pred.size()[-1])
        labels = target[1]
        for i in range(labels.size()[-1]):
            for j in range(labels.size()[0]):
                lab = labels[j, :, i].contiguous().view(-1)
                cel = CEL(pred[j], lab)
                loss_supervised += cel
                loss_reinforce += Variable(cel.data) * logprobs[j]
    else:
        labels = target[0]
        loss_supervised = BCE(F.sigmoid(pred), labels.type(dtype)).mean()
        # raise ValueError('Only cross entropy implemented.')
    return loss_supervised/pred.size(0), loss_reinforce/pred.size(0)

def plot_clusters(num, pred, cities):
    plt.figure(0)
    plt.clf()
    colors = cm.rainbow([0.0,0.5])
    bin_pred = (pred >= 0.5).data
    pts1x = torch.masked_select(cities[:,0], bin_pred)
    pts1y = torch.masked_select(cities[:,1], bin_pred)
    pts2x = torch.masked_select(cities[:,0], 1-bin_pred)
    pts2y = torch.masked_select(cities[:,1], 1-bin_pred)
    plt.scatter(pts1x.cpu().numpy(), pts1y.cpu().numpy(), c=colors[0])
    plt.scatter(pts2x.cpu().numpy(), pts2y.cpu().numpy(), c=colors[1])
    plt.title('clustering')
    plt.savefig('./plots/clustering/clustering_it_{}.png'.format(num))

def compute_accuracy(pred, labels):
    pred = torch.topk(pred, 2, dim=2)[1]
    p = torch.sort(pred, 2)[0]
    l = torch.sort(labels, 2)[0]
    # print('pred', p)
    # print('labels', l)
    # print(torch.eq(p, l).min(2)[0].type(dtype).size())
    error = 1 - torch.eq(p, l).min(2)[0].type(dtype)
    frob_norm = error.mean(1)
    accuracy = 1 - frob_norm
    accuracy = accuracy.mean(0).squeeze()
    return accuracy.data.cpu().numpy()[0]

def compute_variance(probs):
    N = probs.size(1)
    mean = probs.sum(1) / N
    dif = probs - mean.unsqueeze(1).expand_as(probs)
    variance = ((dif*dif).sum(1).sum(0)) / (N*probs.size(0))
    return variance


if __name__ == '__main__':
    path_dataset = './dataset/broma/'
    gen = Generator(path_dataset, './LKH/')
    N = 20
    gen.num_examples_train = 200
    gen.num_examples_test = 10
    gen.N = N
    gen.load_dataset()
    
    clip_grad = 40.0
    iterations = 5000
    batch_size = 20
    num_features = 10
    num_layers = 5
    J = 4
    rf = 10.0 # regularization factor
    
    Split = Split_GNN(batch_size, num_features, num_layers, J+2, dim_input=3)
    Tsp = GNN(num_features, num_layers, J+2, dim_input=3)
    Merge = GNN(num_features, num_layers, J+2, dim_input=3)
    
    optimizer_split = optim.RMSprop(Split.parameters())
    optimizer_tsp = optim.Adamax(Tsp.parameters(), lr=1e-3)
    optimizer_merge = optim.Adamax(Merge.parameters(), lr=1e-3)
    
    for it in range(iterations):
        sample = gen.sample_batch(batch_size, cuda=torch.cuda.is_available())
        input, W, WTSP, labels, target, cities, perms, costs = extract(sample)
        scores, probs = Split(input)
        variance = compute_variance(probs)
        sample, log_probs_samples = sample_one(probs, mode='train')
        WW, x, Phi = compute_operators(W.data, sample, J)
        x = torch.cat((x.unsqueeze(2),cities),2)
        y = WW[:,:,:,1]
        WW = Variable(WW).type(dtype)
        x = Variable(x).type(dtype)
        y = Variable(y).type(dtype)
        #print(WW, x, y)
        partial_pred = Tsp((WW,x,y))
        partial_pred = partial_pred * Variable(Phi)
        #print(input[0], input[1], partial_pred)
        pred = Merge((input[0], input[1], partial_pred))
        loss_supervised, loss_reinforce = compute_loss(pred, target, log_probs_samples)
        loss_reinforce -= variance*rf
        Split.zero_grad()
        loss_reinforce.backward()
        nn.utils.clip_grad_norm(Split.parameters(), clip_grad)
        optimizer_split.step()
        Tsp.zero_grad()
        Merge.zero_grad()
        loss_supervised.backward()
        nn.utils.clip_grad_norm(Tsp.parameters(), clip_grad)
        nn.utils.clip_grad_norm(Merge.parameters(), clip_grad)
        optimizer_tsp.step()
        optimizer_merge.step()
        
        if it%50 == 0:
            acc = compute_accuracy(pred, labels)
            out = [it, loss_reinforce.data[0], loss_supervised.data[0], acc]
            print(template.format(*out))
            print(variance)
            #print(probs[0])
            #print('iteracio {}:\nloss_r={}\nloss_s={}'.format(it,loss_reinforce.data[0],loss_supervised.data[0]))
            plot_clusters(it, probs[0], cities[0])
            #os.system('eog ./plots/clustering/clustering_it_{}.png'.format(it))
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    
    