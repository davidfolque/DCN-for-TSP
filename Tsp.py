


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from data_generator import Generator
from model import Split_GNN, GNN
from Logger import Logger
import utils
import os
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

'''
class Tsp():
    def __init__(self, batch_size, num_features, num_layers, J):
        self.batch_size = batch_size
        self.num_features = num_features,
        self.num_layers = num_layers
        self.J = J
        
        self.split = Split_GNN(batch_size, num_features, num_layers, J)
        
'''

sortid = open('sortida.txt', 'w')

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

template_train1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} '
template_train2 = ('{:<10} {:<10} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}'
                   '{:<10.3f} \n')
template_test1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'
template_test2 = '{:<10} {:<10} {:<10.5f} {:<10.5f} {:<10.5f} {:<10} {:<10.5f}'
info_train = ['TRAIN', 'iteration', 'loss', 'accuracy', 'cost', 'loss_re',
              'elapsed']
info_test = ['TEST', 'iteration', 'loss', 'accuracy', 'cost',
             'beam_size', 'elapsed']


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
        rand = torch.ones(*probs.size()).type(dtype) / 2
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

def compute_test_cost(pred, W, beam_size):
    utils.beamsearch_hamcycle(pred.data, W.data, beam_size=beam_size)

def test(split, logger, gen):
    iterations_test = int(gen.num_examples_test / batch_size)
    # siamese_gnn.eval()
    for it in range(iterations_test):
        start = time.time()
        batch = gen.sample_batch(batch_size, is_training=False, it=it,
                                cuda=torch.cuda.is_available())
        input, W, WTSP, labels, target, cities, perms, costs = extract(batch)
        
        pred = Tsp(input)
        loss = compute_loss(pred, target, Variable(torch.zeros(batch_size).type(dtype)))[0]
            
        last = (it == iterations_test-1)
        logger.add_test_accuracy(pred, labels, perms, W, cities, costs, last=last)
        logger.add_test_loss(loss, last=last)
        elapsed = time.time() - start
        if not last and it % 100 == 0:
            loss = loss.data.cpu().numpy()[0]
            out = ['---', it, loss, logger.accuracy_test_aux[-1], 
                   0, 0, elapsed]
            print(template_test1.format(*info_test))
            print(template_test1.format(*info_test), file=sortid)
            print(template_test2.format(*out))
            print(template_test2.format(*out), file=sortid)
    print('TEST COST: {} | TEST ACCURACY {}\n'
          .format(logger.cost_test[-1], logger.accuracy_test[-1]))
    print('TEST COST: {} | TEST ACCURACY {}\n'
          .format(logger.cost_test[-1], logger.accuracy_test[-1]), file=sortid)

def execute(Split, Tsp, Merge, batch):
    input, W, WTSP, labels, target, cities, perms, costs = extract(batch)
    #print(target[0])
    #scores, probs = Split(input)
    #variance = compute_variance(probs)
    #sample, log_probs_samples = sample_one(probs, mode='train')
    #WW, x, Phi = compute_operators(W.data, sample, J)
    #x = torch.cat((x.unsqueeze(2),cities),2)
    #y = WW[:,:,:,1]
    #WW = Variable(WW).type(dtype)
    #x = Variable(x).type(dtype)
    #y = Variable(y).type(dtype)
    #print(WW, x, y)
    #partial_pred = Tsp((WW,x,y))
    #partial_pred = partial_pred * Variable(Phi)
    #print(input[0], input[1], partial_pred)
    entradaY = oracle_split(W.data, target[0], perms)[0]
    pred = Merge((input[0], input[1], entradaY.float()))
    return 0, 0, Variable(torch.zeros(batch_size,N)).type(dtype), pred

def fake_split_b(W, mat_perm, perms, i):
    N = mat_perm.size(-1)
    def index(k):
        return (N+k+i)%N
    n2 = N // 2
    pre0 = math.floor(perms[index(-1)]+0.5)
    exc0 = math.floor(perms[index(0)]+0.5)
    pre10 = math.floor(perms[index(n2-1)]+0.5)
    exc10 = math.floor(perms[index(n2)]+0.5)
    return pre0,exc0,pre10,exc10

def fake_split(W, mat_perm, perms, i):    
    N = mat_perm.size(-1)
    def index(k):
        return (N+k+i)%N
    
    bs = mat_perm.size(0)
    n2 = N // 2
    out = mat_perm.clone()
    for b in range(bs):
        pre0 = math.floor(perms[b,index(-1)]+0.5)
        exc0 = math.floor(perms[b,index(0)]+0.5)
        pre10 = math.floor(perms[b,index(n2-1)]+0.5)
        exc10 = math.floor(perms[b,index(n2)]+0.5)
        out[b,pre0,exc0] = 0
        out[b,exc0,pre0] = 0
        out[b,pre0,exc10] = 1
        out[b,exc10,pre0] = 1
        out[b,pre10,exc0] = 1
        out[b,exc0,pre10] = 1
        out[b,pre10,exc10] = 0
        out[b,exc10,pre10] = 0
    return out

def oracle_split(W, mat_perm, perms):
    N = mat_perm.size(-1)
    bs = mat_perm.size(0)
    out = mat_perm.clone()
    sep = torch.LongTensor(bs).fill_(0).type(dtype_l)
    gt = torch.zeros(bs,N).type(dtype)
    for b in range(bs):
        costos = torch.zeros(N//2).type(dtype)
        for i in range(N // 2):
            pre0,exc0,pre10,exc10 = fake_split_b(W[b],mat_perm[b],perms[b],i)
            costos[i] = -(W[b,pre0,exc0] + W[b,pre10,exc10] - W[b,pre0,exc10] - W[b,pre10,exc0])
        imax = costos.max(0)[1]
        for i in range(N // 2):
            ind = math.floor(perms[b,(N + imax[0] + i)%N] + 0.5)
            gt[b,ind] = 1.0
        
    return gt

def compute_split_loss(probs, gt):
    bs = probs.size(0)
    N = probs.size(1)
    gt_inv = Variable(1 - gt)
    gt = Variable(gt)
    probs_inv = 1 - probs
    A = torch.bmm(gt.unsqueeze(1),torch.log(probs.unsqueeze(2))) + torch.bmm(gt_inv.unsqueeze(1),torch.log(probs_inv.unsqueeze(2)))
    B = torch.bmm(gt_inv.unsqueeze(1),torch.log(probs.unsqueeze(2))) + torch.bmm(gt.unsqueeze(1),torch.log(probs_inv.unsqueeze(2)))
    m = torch.min(-A,-B).squeeze()
    loss = m.mean(0)
    return loss

def execute_split_train(Split, batch):
    input, W, WTSP, labels, target, cities, perms, costs = extract(batch)
    scores, probs = Split(input)
    #variance = compute_variance(probs)
    #sample, log_probs_samples = sample_one(probs, mode='train')
    ground_truth = oracle_split(W.data, target[0], perms)
    loss = compute_split_loss(probs, ground_truth)
    return probs, ground_truth, loss


if __name__ == '__main__':
    path_dataset = '/data/folque/dataset/'
    gen = Generator(path_dataset, './LKH/')
    #N = 20
    gen.num_examples_train = 20000
    gen.num_examples_test = 1000
    gen.N_train = 10
    gen.N_test = 10
    gen.load_dataset()
    
    clip_grad = 40.0
    iterations = 100000
    batch_size = 20
    num_features = 20
    num_layers = 20
    J = 4
    beam_size = 40
    
    logger = Logger('./logs')
    
    Split = Split_GNN(batch_size, num_features, num_layers, J+2, dim_input=3)
    Tsp = GNN(num_features, num_layers, J+2, dim_input=3)
    Merge = GNN(num_features, num_layers, J+2, dim_input=3)
    
    optimizer_split = optim.RMSprop(Split.parameters())
    optimizer_tsp = optim.Adamax(Tsp.parameters(), lr=1e-3)
    optimizer_merge = optim.Adamax(Merge.parameters(), lr=1e-3)
    
    mode = 'train'
    #Split, Tsp, Merge = logger.load_model('./logs')
    if mode == 'train':
        for it in range(iterations):
            start = time.time()
            batch = gen.sample_batch(batch_size, cuda=torch.cuda.is_available())
            input, W, WTSP, labels, target, cities, perms, costs = extract(batch)
            #probs, ground_truth, loss_split = execute_split_train(Split, batch)
            #Split.zero_grad()
            #loss_split.backward()
            #nn.utils.clip_grad_norm(Split.parameters(), clip_grad)
            #optimizer_split.step()
            pred = Tsp(input)
            loss = compute_loss(pred, target, Variable(torch.zeros(batch_size).type(dtype)))[0]
            Tsp.zero_grad()
            #Merge.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(Tsp.parameters(), clip_grad)
            #nn.utils.clip_grad_norm(Merge.parameters(), clip_grad)
            optimizer_tsp.step()
            #optimizer_merge.step()
            logger.add_train_loss(loss)
            logger.add_train_accuracy(pred, labels)
            elapsed = time.time() - start
            
            if it%100 == 0:
                loss = loss.data.cpu().numpy()[0]
                out = ['---', it, loss, logger.accuracy_train[-1],
                    0, 0, elapsed]
                print(template_train1.format(*info_train))
                print(template_train1.format(*info_train), file=sortid)
                print(template_train2.format(*out))
                print(template_train2.format(*out), file=sortid)
                #print(variance)
                #print(probs[0])
                #plot_clusters(it, probs[0], cities[0])
                #os.system('eog ./plots/clustering/clustering_it_{}.png'.format(it))
            if it%2000 == 0 and it > 0:
                test(Split, logger, gen)
            if it%1000 == 0 and it > 0:
                logger.save_model('./saved_model/',Split,Tsp,Merge)
    else:
        Split, Tsp, Merge = logger.load_model('./logs')
        test(Split, logger, gen)
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    
    
