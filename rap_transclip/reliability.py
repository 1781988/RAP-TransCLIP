from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .utils import l2_normalize

@dataclass
class PromptReliabilityResult:
    weights: torch.Tensor
    prototypes: torch.Tensor
    probabilities: torch.Tensor
    scores: torch.Tensor

def normalized_entropy(probabilities:torch.Tensor,eps:float=1e-8)->torch.Tensor:
    k=probabilities.shape[-1]; p=probabilities.clamp_min(eps); entropy=-(p*p.log()).sum(dim=-1); return entropy/max(float(torch.log(torch.tensor(float(k),device=probabilities.device))),eps)

def estimate_prompt_reliability(image_features:torch.Tensor,text_features:torch.Tensor,cfg:dict,logit_scale:float=100.0,target_assignments:torch.Tensor|None=None,previous_weights:torch.Tensor|None=None)->PromptReliabilityResult:
    m,k,_=text_features.shape; n=image_features.shape[0]; logits=logit_scale*torch.einsum("nd,mkd->mnk",image_features,text_features); probabilities=F.softmax(logits,dim=-1)
    top_count=min(max(int(cfg.get("min_top_samples",8)),int(n*float(cfg.get("top_fraction",0.02)))),n); scores=torch.empty((m,k),device=image_features.device,dtype=image_features.dtype)
    for prompt_id in range(m):
        p=probabilities[prompt_id]; entropy=normalized_entropy(p)
        for class_id in range(k):
            class_prob=p[:,class_id]; values,indices=class_prob.topk(top_count); confidence=values.mean(); selected_entropy=entropy[indices].mean(); competitors=p[indices].clone(); competitors[:,class_id]=-1.0; margin=(values-competitors.max(dim=1).values).mean(); agreement=confidence if target_assignments is None else 1.0-(values-target_assignments[indices,class_id]).abs().mean()
            scores[prompt_id,class_id]=float(cfg.get("confidence_weight",1.0))*confidence-float(cfg.get("entropy_weight",0.6))*selected_entropy+float(cfg.get("margin_weight",0.7))*margin+float(cfg.get("agreement_weight",0.8))*agreement
    weights=F.softmax(scores/max(float(cfg.get("temperature",0.15)),1e-4),dim=0)
    if previous_weights is not None:
        ema=float(cfg.get("weight_ema",0.5)); weights=ema*previous_weights+(1.0-ema)*weights; weights=weights/weights.sum(dim=0,keepdim=True).clamp_min(1e-8)
    prototypes=l2_normalize(torch.einsum("mk,mkd->kd",weights,text_features)); return PromptReliabilityResult(weights,prototypes,probabilities,scores)

def estimate_sample_reliability(ensemble_probabilities:torch.Tensor,prompt_probabilities:torch.Tensor,cfg:dict)->torch.Tensor:
    entropy_quality=1.0-normalized_entropy(ensemble_probabilities); predicted_class=ensemble_probabilities.argmax(dim=1); gather_index=predicted_class.view(1,-1,1).expand(prompt_probabilities.shape[0],-1,1); selected=prompt_probabilities.gather(2,gather_index).squeeze(-1); disagreement=selected.var(dim=0,unbiased=False); disagreement=disagreement/disagreement.max().clamp_min(1e-8); agreement_quality=1.0-disagreement; quality=float(cfg.get("entropy_weight",0.6))*entropy_quality+float(cfg.get("disagreement_weight",0.4))*agreement_quality; return quality.clamp_min(float(cfg.get("minimum",0.05))).clamp_max(1.0)
