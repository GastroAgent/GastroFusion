from collections import OrderedDict
import torch

def gradient_update_parameters(model, meta_params=None, loss=None, params=None, inner_lr=None, first_order=False, GAP=True, approx=False):
    if params is None:
        params=OrderedDict(model.named_parameters())
    grads=torch.autograd.grad(loss, params.values(), create_graph=not first_order)
    out=OrderedDict()
    for (n,par),g in zip(params.items(),grads):
        if GAP and g.ndim==4 and meta_params is not None and n in meta_params:
            shape=g.shape; gm=g.reshape(g.size(0),-1)
            if approx:
                m=torch.diag(torch.nn.functional.softplus(meta_params[n], beta=2))
                gm=m@gm if gm.size(0)<=gm.size(1) else gm@m
            else:
                u,_,_=torch.linalg.svd(gm.detach(), full_matrices=False)
                d=torch.nn.functional.softplus(meta_params[n], beta=2)
                k=min(u.shape[1], d.numel())
                m=u[:,:k]@torch.diag(d[:k])@u[:,:k].T
                gm=m@gm
            g=gm.reshape(shape)
        out[n]=par-inner_lr*g
    return out
