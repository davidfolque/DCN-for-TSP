import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from data_generator import Generator
import torch.nn.functional as F
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def gmul(input):
    W, x, y = input
    # x is a tensor of size (bs, N, num_features)
    # y is a tensor of size (bs, N, N)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    J = W_size[-1]
    W = W.split(1, 3)
    W = W + (y.unsqueeze(3),)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    #print(W, x)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


class Gconv(nn.Module):
    def __init__(self, feature_maps, J, last=False):
        super(Gconv, self).__init__()
        self.num_inputs = (J+1)*feature_maps[0]
        self.num_outputs = feature_maps[1]
        self.last = last
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2).type(dtype)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs // 2).type(dtype)
        self.beta = nn.Linear(self.num_outputs, 1).type(dtype)
        self.sigma = nn.Parameter(torch.FloatTensor([random.uniform(0.0,1.0)]).type(dtype))
        self.bn_instance = nn.InstanceNorm1d(self.num_outputs).type(dtype)

    def forward(self, input):
        W, x, y = input
        N = y.size(-1)
        bs = y.size(0)
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        if self.last:
            x1 = self.fc1(x)
        else:
            x1 = F.relu(self.fc1(x)) # has size (bs*N, num_outputs // 2)
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        x = x.view(*x_size[:-1], self.num_outputs)
        x = self.bn_instance(x.permute(0, 2, 1)).permute(0, 2, 1)
        bx = self.beta(x) # has size (bs,N,1)
        Bx = bx.expand(bs,N,N) # has size (bs,N,N) by repeating columns
        y = F.sigmoid(Bx + Bx.permute(0,2,1) + self.sigma*y)
        #y = (y + y.permute(0,2,1))/2
        y = y * (1-Variable(torch.eye(N).type(dtype)).unsqueeze(0).expand(bs,N,N))
        return W, x, y


class Split_GNN(nn.Module):
    def __init__(self, batch_size, num_features, num_layers, J, dim_input=1):
        super(Split_GNN, self).__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [dim_input, num_features]
        self.featuremap_mi = [num_features, num_features]
        self.featuremap_end = [num_features, num_features]
        self.layer0 = Gconv(self.featuremap_in, J)
        for i in range(num_layers):
            module = Gconv(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv(self.featuremap_end, J, last=True)
        self.linear_last = nn.Linear(self.featuremap_end[1], 1, bias=True).type(dtype)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        x = out[1]
        x = x.view(-1,self.featuremap_end[1])
        scores = self.linear_last(out[1]).view(self.batch_size, -1)
        probs = F.sigmoid(scores)
        return scores, probs

def compute_cluster_loss(bin_pred, cities):
    bin_pred2 = bin_pred.unsqueeze(2).expand_as(cities)
    cities_masked = bin_pred2*cities
    n_cities = bin_pred2.sum(1)
    mean = cities_masked.sum(1)/n_cities.clamp(min=1.0)
    mean_masked = bin_pred2 * mean.unsqueeze(1).expand_as(cities)
    distances = mean_masked - cities_masked
    loss = (distances*distances).sum(2).sum(1)
    return loss

def compute_loss(pred, cities):
    pred = 1e-6 + pred*(1-2e-6)
    bin_pred = (pred >= 0.5).data.float().type(dtype)
    c1 = compute_cluster_loss(bin_pred, cities)
    c2 = compute_cluster_loss(1-bin_pred, cities)
    rndm = torch.rand(*pred.size())
    bin_rndm = (rndm >= 0.5).float().type(dtype)
    baseline1 = compute_cluster_loss(bin_rndm, cities)
    baseline2 = compute_cluster_loss(1-bin_rndm, cities)
    c = c1 + c2 - baseline1 - baseline2
    c = Variable(c)
    bin_pred = Variable(bin_pred)
    p = torch.sum(torch.log(pred)*bin_pred,1) + torch.sum(torch.log(1-pred)*(1-bin_pred),1)
    return torch.dot(p, c), (c1+c2).mean(), (baseline1+baseline2).mean()

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
    
def normalize_embeddings(emb):
    norm = torch.mul(emb, emb).sum(2).unsqueeze(2).sqrt().expand_as(emb)
    return emb.div(norm)

class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J, dim_input=1):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [dim_input, num_features]
        self.featuremap_mi = [num_features, num_features]
        self.featuremap_end = [num_features, num_features]
        self.layer0 = Gconv(self.featuremap_in, J)
        for i in range(num_layers):
            module = Gconv(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv(self.featuremap_end, J, last=True)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        emb = self.layerlast(cur)[1]
        # emb are tensors of size (bs, N, num_features)
        N = emb.size(1)
        
        # l2normalize the embeddings
        emb = normalize_embeddings(emb)
        out = torch.bmm(emb, emb.permute(0, 2, 1))
        diag = (-1000 * Variable(torch.eye(N).unsqueeze(0)
                .expand_as(out)).type(dtype))
        out = out + diag
        return out # out has size (bs, N, N)


if __name__ == '__main__':
    path_dataset = './dataset/'
    gen = Generator(path_dataset, './LKH/')
    N = 20
    gen.num_examples_train = 20000
    gen.num_examples_test = 1000
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
    
    for it in range(iterations):
        sample = gen.sample_batch(batch_size, cuda=torch.cuda.is_available())
        input, W, WTSP, labels, target, cities, perms, costs = extract(sample)
        #print(input)
        #exit()
        pred = Split(input)
        loss, distances, baselines = compute_loss(pred[1], cities)
        Split.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(Split.parameters(), grad_clip_split)
        optimizer.step()
        if it%100 == 0:
            print('iteration {}, loss={},\n distances={}, baselines={}'.format(it,loss.data[0],distances, baselines))
        if it%50 == 0:
            plot_clusters(it, pred[1][0], cities[0])
        
    
    
    
    
    
    
    
    
    
    