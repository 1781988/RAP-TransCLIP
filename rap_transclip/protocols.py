from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class ProtocolOutput:
    indices: torch.Tensor
    metadata: dict[str,object]

def full_protocol(labels:torch.Tensor,seed:int=1,**_:object)->ProtocolOutput:
    return ProtocolOutput(torch.arange(len(labels)),{"protocol":"full","seed":seed})

def partial_class_protocol(labels:torch.Tensor,seed:int=1,class_fraction:float=0.5,**_:object)->ProtocolOutput:
    rng=np.random.default_rng(seed); classes=torch.unique(labels).cpu().numpy(); count=max(1,int(round(len(classes)*class_fraction))); active=np.sort(rng.choice(classes,size=count,replace=False)); mask=torch.zeros_like(labels,dtype=torch.bool)
    for class_id in active.tolist(): mask|=labels==int(class_id)
    return ProtocolOutput(mask.nonzero(as_tuple=False).squeeze(1),{"protocol":"partial_class","seed":seed,"class_fraction":class_fraction,"active_classes":active.tolist()})

def dirichlet_long_tail_protocol(labels:torch.Tensor,seed:int=1,alpha:float=0.5,maximum_samples:int|None=None,**_:object)->ProtocolOutput:
    rng=np.random.default_rng(seed); classes=torch.unique(labels).cpu().numpy(); target_total=maximum_samples or len(labels); proportions=rng.dirichlet(np.full(len(classes),alpha)); selected=[]
    for class_id,proportion in zip(classes.tolist(),proportions.tolist()):
        candidates=(labels==int(class_id)).nonzero(as_tuple=False).squeeze(1).cpu().numpy(); count=min(len(candidates),max(1,int(round(target_total*proportion)))); selected.extend(rng.choice(candidates,size=count,replace=False).tolist())
    rng.shuffle(selected); return ProtocolOutput(torch.tensor(selected,dtype=torch.long),{"protocol":"dirichlet_long_tail","seed":seed,"alpha":alpha})

def create_protocol(name:str,labels:torch.Tensor,seed:int,**kwargs:object)->ProtocolOutput:
    if name=="full": return full_protocol(labels,seed=seed,**kwargs)
    if name=="partial_class": return partial_class_protocol(labels,seed=seed,**kwargs)
    if name in {"long_tail","dirichlet_long_tail"}: return dirichlet_long_tail_protocol(labels,seed=seed,**kwargs)
    raise ValueError(f"Unsupported protocol: {name}")
