"""Observe actual clipped gradients and AdamW updates without changing the loss.

CPU arithmetic below is a reference for ONE tensor update, never a CPU model fit.
Incident files contain the exact critical-tensor pre-state and full post-state;
they do not claim to contain the whole model's pre-update state.
"""
from collections import deque
import hashlib,json
from pathlib import Path
import numpy as np
import torch


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


class UpdateMonitor:
    protocol=dict(critical_parameter='item_mlp.3.weight',every_step=True,all_layers_every=50,
        gradient_clipping=1.,
        warning=dict(critical_relative_update=.05,critical_max_abs_update=.05,adam_max_abs_error=.0001,
                     global_gradient_norm=1000.),
        stop=dict(adam_max_abs_error=.001,postclip_norm=1.01,
                  catastrophic_update='weight norm more than doubles and relative update exceeds0.5',
                  sustained_loss='100 consecutive rolling100 windows above best rolling100 by both2.0 and25%; only after1000updates'),
        incident_limit=2,incident_states='critical tensor+Adam pre-update; full model+optimizer post-update; exact batch',
        note='Thresholds are diagnostic project rules, not evidence of a bug by themselves; no final test')

    def __init__(self):
        self.index=0;self.stop_reason=None;self.window=deque(maxlen=100)
        self.best_loss=float('inf');self.bad_windows=0;self.incidents=0;self.warning_types=set()

    def attach(self,dest):
        self.dest=Path(dest);self.log=(self.dest/'update_diagnostics.jsonl').open('w',buffering=1)
        self.layers=(self.dest/'layer_diagnostics.jsonl').open('w',buffering=1)
        (self.dest/'monitor_snapshot.py').write_bytes(Path(__file__).read_bytes())

    def before_update(self,model,optimizer,batch,loss,grad_norm):
        self.index+=1;self.loss=loss;self.parameter=dict(model.named_parameters())['item_mlp.3.weight']
        self.before=self.parameter.detach().cpu().clone()
        self.gradient=self.parameter.grad.detach().cpu().clone()
        self.pre_state=cpu_tree(optimizer.state[self.parameter])
        self.group=next(g for g in optimizer.param_groups if any(p is self.parameter for p in g['params']))
        grads=[p.grad.detach() for p in model.parameters() if p.grad is not None]
        self.grad_pre=float(grad_norm.detach().cpu())
        self.grad_post=float(torch.stack([g.norm() for g in grads]).norm().detach().cpu())
        if not np.isfinite(self.grad_pre) or not np.isfinite(self.grad_post):
            self.save_failure(model,optimizer,batch,'Non-finite gradient norm before optimizer update')
            raise FloatingPointError('Non-finite gradient')
        if self.index<=3:
            h=hashlib.sha256()
            for x in batch:h.update(np.asarray(x).tobytes())
            with (self.dest/'first_batches.jsonl').open('a') as f:
                f.write(json.dumps(dict(update=self.index,sha256=h.hexdigest()))+'\n')

    def save_failure(self,model,optimizer,batch,reason):
        torch.save(dict(reason=reason,model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
            batch={name:torch.from_numpy(np.asarray(value).copy()) for name,value in zip(['history','actions','candidates','qdraw','mask'],batch)}),
            self.dest/'numeric_failure.pt')

    def adam_reference(self):
        p=self.before;g=self.gradient;s=self.pre_state;cfg=self.group
        if not s:
            first=torch.zeros_like(p);second=torch.zeros_like(p);step=1
        else:
            first=s['exp_avg'];second=s['exp_avg_sq'];step=int(s['step'].item())+1
        b1,b2=cfg['betas'];m=b1*first+(1-b1)*g;v=b2*second+(1-b2)*g.square()
        # Current recipe has no AMSGrad, maximize or differentiable optimizer.
        assert not cfg.get('amsgrad',False) and not cfg.get('maximize',False)
        return p*(1-cfg['lr']*cfg['weight_decay'])-cfg['lr']*(m/(1-b1**step))/(torch.sqrt(v/(1-b2**step))+cfg['eps'])

    def after_update(self,model,optimizer,batch,loss):
        after=self.parameter.detach().cpu().clone();delta=after-self.before
        reference=self.adam_reference();error=float((after-reference).abs().max())
        pre_norm=float(self.before.norm());post_norm=float(after.norm());dn=float(delta.norm())
        relative=dn/max(pre_norm,1e-12);maxabs=float(delta.abs().max())
        state=optimizer.state[self.parameter]
        avg=state['exp_avg'].detach().cpu();sq=state['exp_avg_sq'].detach().cpu()
        finite=bool(torch.isfinite(after).all() and torch.isfinite(avg).all() and torch.isfinite(sq).all())
        self.window.append(loss);rolling=float(np.mean(self.window));warnings=[]
        if len(self.window)==100:
            self.best_loss=min(self.best_loss,rolling)
            bad=self.index>=1000 and rolling>self.best_loss+2. and rolling>self.best_loss*1.25
            self.bad_windows=self.bad_windows+1 if bad else 0
        if relative>.05:warnings.append('large_relative_update')
        if maxabs>.05:warnings.append('large_coordinate_update')
        if error>.0001:warnings.append('adam_reference_mismatch')
        if self.grad_pre>1000:warnings.append('large_preclip_gradient')
        if not finite:self.stop_reason='Non-finite critical parameter or Adam moments'
        if error>.001:self.stop_reason='Adam reference mismatch above0.001'
        if self.grad_post>1.01:self.stop_reason='Postclip gradient norm above1.01'
        if post_norm>2*pre_norm and relative>.5:self.stop_reason='Catastrophic critical-layer update'
        if self.bad_windows>=100:self.stop_reason='Sustained rolling-loss increase'
        row=dict(update=self.index,loss=loss,rolling100=rolling,best_rolling100=self.best_loss if len(self.window)==100 else None,
            lr=self.group['lr'],gradient_norm_before_clip=self.grad_pre,gradient_norm_after_clip=self.grad_post,
            critical_weight_norm_before=pre_norm,critical_weight_norm_after=post_norm,
            critical_clipped_gradient_norm=float(self.gradient.norm()),critical_update_norm=dn,
            critical_relative_update=relative,critical_max_abs_update=maxabs,
            adam_reference_max_abs_error=error,adam_m_min=float(avg.min()),adam_m_max=float(avg.max()),
            adam_v_min=float(sq.min()),adam_v_max=float(sq.max()),finite=finite,
            warnings=warnings,stop_reason=self.stop_reason)
        row={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
        self.log.write(json.dumps(row,allow_nan=False)+'\n')
        parts=getattr(self,'loss_parts',None)
        if parts is not None and parts['update']==self.index:
            with (self.dest/'ccl_loss_parts.jsonl').open('a') as f:f.write(json.dumps(parts)+'\n')
        if self.index%50==0:
            layers=[]
            for name,p in model.named_parameters():
                state=optimizer.state.get(p,{})
                layers.append(dict(name=name,weight_norm=float(p.detach().norm().cpu()),
                    gradient_norm=None if p.grad is None else float(p.grad.detach().norm().cpu()),
                    adam_m_norm=None if 'exp_avg' not in state else float(state['exp_avg'].detach().norm().cpu()),
                    adam_v_min=None if 'exp_avg_sq' not in state else float(state['exp_avg_sq'].detach().min().cpu())))
            self.layers.write(json.dumps(dict(update=self.index,layers=layers))+'\n')
        new_warning=set(warnings)-self.warning_types;self.warning_types.update(warnings)
        if (new_warning or self.stop_reason) and self.incidents<2:
            package=dict(update=self.index,diagnostic=row,
                critical_parameter_before=self.before,critical_gradient_after_clip=self.gradient,
                critical_optimizer_before=self.pre_state,
                batch={name:torch.from_numpy(np.asarray(value).copy()) for name,value in zip(['history','actions','candidates','qdraw','mask'],batch)},
                model_after=cpu_tree(model.state_dict()),optimizer_after=cpu_tree(optimizer.state_dict()),
                scope='Full post-update state; only critical tensor has exact pre-update state')
            path=self.dest/f'incident_{self.index}.pt';torch.save(package,path);self.incidents+=1
            print('UPDATE_INCIDENT',self.index,warnings,self.stop_reason,'norm',pre_norm,post_norm,'Adam_error',error,flush=True)
        if self.index%100==0:
            print('UPDATE_MONITOR',self.index,'grad',round(self.grad_pre,5),'postclip',round(self.grad_post,5),
                'weight',round(post_norm,5),'relative_update',round(relative,7),'Adam_error',error,flush=True)
