import collections
import pickle
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # avoid display backend on headless GPU nodes
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from numpy import linalg as LA
try:
    from tqdm.notebook import tqdm
except ImportError:  # headless / non-notebook env
    from tqdm import tqdm

use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas


def centerDatas(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
   
    return datas

def scaleEachUnitaryDatas(datas):
  
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms


def QRreduction(datas):
    
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self):
        self.mus = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:n_shot,].mean(1)                           

    def updateFromEstimate(self, estimate, alpha):   
        
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
                                         
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        
        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * n_queries
       
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test
        
        p_xj[:,:n_lsamples].fill_(0)
        p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        
        return p_xj

    def estimateFromMask(self, mask):

        emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus

          
# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
    
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, epochInfo=None):
     
        p_xj = model.getProbas()
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))
        
        m_estimates = model.estimateFromMask(self.probas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, n_epochs=20):
        
        self.probas = model.getProbas()
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}  lr_m: {:0.3f}".format(epoch, self.alpha))
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            if (self.progressBar): pb.update()
        
        # get final accuracy and return it
        op_xj = model.getProbas()
        acc = self.getAccuracy(op_xj)
        return acc
    

if __name__ == '__main__':
    # ---- parameters (override via CLI) ----
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--n_ways', type=int, default=5)
    parser.add_argument('--n_queries', type=int, default=15)
    parser.add_argument('--n_runs', type=int, default=10000)
    parser.add_argument('--lam', type=float, default=10)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--dsname', type=str, default='miniimagenet',
                        help='Logical dataset name (miniimagenet/cub/cifar/cross); '
                             'for medical subsets use --dataset medical --medical_subset <name>')
    parser.add_argument('--n_lsamples', type=int, default=0,
                        help='computed if 0')
    parser.add_argument('--output_json', type=str, default='',
                        help='Optional path to save the result as JSON')
    # Medical-specific flags (needed so test_standard knows where to find features)
    parser.add_argument('--dataset', type=str, default='',
                        help='Set to "medical" for medical subsets')
    parser.add_argument('--medical_subset', type=str, default='',
                        help='Medical subset name (e.g. 胃_1query)')
    parser.add_argument('--model', type=str, default='WideResNet28_10',
                        help='Backbone model used to extract features (must match save_plk.py run)')
    parser.add_argument('--method', type=str, default='S2M2_R',
                        help='Training method used to extract features (must match save_plk.py run)')
    args_cli = parser.parse_args()

    n_shot = args_cli.n_shot
    n_ways = args_cli.n_ways
    n_queries = args_cli.n_queries
    n_runs = args_cli.n_runs
    n_lsamples = args_cli.n_lsamples or n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask
    # Tell FSLTask where to find the medical features (if applicable)
    if args_cli.dataset == 'medical' and args_cli.medical_subset:
        FSLTask.configure_medical(
            subset=args_cli.medical_subset,
            model=args_cli.model,
            method=args_cli.method)
        print(f'[test_standard] configured medical: subset={args_cli.medical_subset} '
              f'model={args_cli.model} method={args_cli.method}')

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(args_cli.dsname)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs, n_samples)

    # Power transform
    beta = args_cli.beta
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)

    ndatas = QRreduction(ndatas)
    n_nfeat = ndatas.size(2)

    ndatas = scaleEachUnitaryDatas(ndatas)
    ndatas = centerDatas(ndatas)

    print("size of the datas...", ndatas.size())

    if use_gpu:
        ndatas = ndatas.cuda()
        labels = labels.cuda()

    # MAP
    lam = args_cli.lam
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    alpha = args_cli.alpha
    optim = MAP(alpha)

    optim.verbose = False
    optim.progressBar = True

    acc_test = optim.loop(model, n_epochs=args_cli.n_iter)

    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100 * x for x in acc_test)))

    if args_cli.output_json:
        import json
        with open(args_cli.output_json, 'w') as f:
            json.dump({
                'dataset': args_cli.dsname,
                'n_ways': n_ways, 'n_shot': n_shot, 'n_queries': n_queries,
                'n_runs': n_runs,
                'test_acc': float(acc_test[0] * 100),
                'pm': float(acc_test[1] * 100),
                'lam': lam, 'alpha': alpha, 'beta': beta,
            }, f, indent=2)
        print(f'Saved result to {args_cli.output_json}')
    
    

