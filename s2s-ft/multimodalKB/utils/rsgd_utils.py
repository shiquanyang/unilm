import torch as th
import pdb


def expm(p, d_p, normalize=False, lr=None, out=None):
    if lr is not None:
        d_p.mul_(-lr)
    if out is None:
        out = p
    out.add_(d_p)
    return out

def rgrad(p, d_p):
    if d_p.is_sparse:
        p_sqnorm = th.sum(
            p[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        n_vals.renorm_(2, 0, 5)
        d_p = th.sparse.DoubleTensor(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = th.sum(p ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p

def logm(p, d_p, out=None):
    return p - d_p

def ptransp(p, x, y, v):
    ix, v_ = v._indices().squeeze(), v._values()
    return p.index_copy_(0, ix, v_)