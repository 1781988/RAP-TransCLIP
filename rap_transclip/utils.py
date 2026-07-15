from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
import numpy as np
import torch

def seed_everything(seed:int,deterministic:bool=True)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic: torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def l2_normalize(x:torch.Tensor,dim:int=-1,eps:float=1e-12)->torch.Tensor:
    return x/x.norm(dim=dim,keepdim=True).clamp_min(eps)

def read_json(path:str|Path)->Any:
    with Path(path).open("r",encoding="utf-8") as handle: return json.load(handle)

def write_json(path:str|Path,obj:Any)->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as handle: json.dump(obj,handle,indent=2,ensure_ascii=False)
