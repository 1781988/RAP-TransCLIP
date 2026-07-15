from __future__ import annotations

import torch
from sklearn.metrics import f1_score


def top1_accuracy(probabilities:torch.Tensor,labels:torch.Tensor)->float:
    return float((probabilities.argmax(dim=1)==labels).float().mean().item()*100.0)

def macro_f1(probabilities:torch.Tensor,labels:torch.Tensor)->float:
    return float(f1_score(labels.detach().cpu().numpy(),probabilities.argmax(dim=1).detach().cpu().numpy(),average="macro",zero_division=0)*100.0)

def expected_calibration_error(probabilities:torch.Tensor,labels:torch.Tensor,num_bins:int=15)->float:
    confidence,predictions=probabilities.max(dim=1); correctness=(predictions==labels).float(); boundaries=torch.linspace(0,1,num_bins+1,device=probabilities.device); ece=torch.zeros((),device=probabilities.device)
    for lower,upper in zip(boundaries[:-1],boundaries[1:]):
        mask=(confidence>lower)&(confidence<=upper)
        if mask.any(): ece+=mask.float().mean()*(confidence[mask].mean()-correctness[mask].mean()).abs()
    return float(ece.item()*100.0)
