from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

@dataclass
class LoadedVLM:
    model: torch.nn.Module
    preprocess: Any
    tokenizer: Any

def _extract_state_dict(checkpoint:Any)->dict[str,torch.Tensor]:
    if isinstance(checkpoint,dict):
        for key in ("state_dict","model","module","model_state_dict"):
            value=checkpoint.get(key)
            if isinstance(value,dict): checkpoint=value; break
    if not isinstance(checkpoint,dict): raise TypeError("Checkpoint does not contain a state dictionary")
    cleaned={}
    for key,value in checkpoint.items():
        if not isinstance(value,torch.Tensor): continue
        for prefix in ("module.","model."):
            if key.startswith(prefix): key=key[len(prefix):]
        cleaned[key]=value
    return cleaned

def load_vlm(backend:str,architecture:str,pretrained:str|None,checkpoint:str|None,device:torch.device)->LoadedVLM:
    try: import open_clip
    except ImportError as exc: raise ImportError("Install open-clip-torch before extracting features") from exc
    if backend not in {"open_clip","open_clip_checkpoint"}: raise ValueError(f"Unsupported backend: {backend}")
    if backend=="open_clip": model,_,preprocess=open_clip.create_model_and_transforms(architecture,pretrained=pretrained)
    else:
        model,_,preprocess=open_clip.create_model_and_transforms(architecture,pretrained=None)
        if checkpoint is None: raise ValueError("A checkpoint path is required for open_clip_checkpoint")
        checkpoint_path=Path(checkpoint)
        if not checkpoint_path.exists(): raise FileNotFoundError(checkpoint_path)
        state=_extract_state_dict(torch.load(checkpoint_path,map_location="cpu")); missing,unexpected=model.load_state_dict(state,strict=False)
        print(f"Checkpoint loaded: {checkpoint_path}"); print(f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")
    tokenizer=open_clip.get_tokenizer(architecture); return LoadedVLM(model.to(device).eval(),preprocess,tokenizer)
