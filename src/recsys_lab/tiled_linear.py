"""Optional first-order Linear backward with bounded reduction blocks on MPS.

Forward and parameters are unchanged. This is a numeric implementation variant,
not a new architecture. Enable only after a CPU/MPS gradient identity check.
"""
import torch
from torch import nn
from torch.nn import functional as F

class _TiledLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,chunk):
        ctx.save_for_backward(x,weight);ctx.has_bias=bias is not None;ctx.chunk=chunk
        return F.linear(x,weight,bias)

    @staticmethod
    def backward(ctx,grad_output):
        x,weight=ctx.saved_tensors
        gx=grad_output.matmul(weight) if ctx.needs_input_grad[0] else None
        xf=x.reshape(-1,x.shape[-1]);gf=grad_output.reshape(-1,weight.shape[0])
        gw=torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        gb=torch.zeros(weight.shape[0],device=weight.device,dtype=weight.dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None
        for lo in range(0,len(xf),ctx.chunk):
            g=gf[lo:lo+ctx.chunk];v=xf[lo:lo+ctx.chunk]
            if gw is not None:gw.add_(g.T.contiguous()@v.contiguous())
            if gb is not None:gb.add_(g.sum(0))
        return gx,gw,gb,None

class TiledLinear(nn.Module):
    def __init__(self,linear,chunk=16384):
        super().__init__();self.in_features=linear.in_features;self.out_features=linear.out_features
        self.weight=linear.weight;self.bias=linear.bias;self.chunk=chunk

    def forward(self,x):
        return _TiledLinear.apply(x,self.weight,self.bias,self.chunk)

def patch_projection(model,chunk=16384):
    old=model.item_mlp[3]
    assert isinstance(old,nn.Linear)
    model.item_mlp[3]=TiledLinear(old,chunk)
    return model

def patch_tower_linears(model,chunk=16384):
    """All4 Linear layers of the current mean-pooling ResidualTower only."""
    for block,indices in [(model.item_mlp,[0,3]),(model.user_mlp,[0,2])]:
        for index in indices:
            if isinstance(block[index],TiledLinear):continue
            assert isinstance(block[index],nn.Linear)
            block[index]=TiledLinear(block[index],chunk)
    return model
