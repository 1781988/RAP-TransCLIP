from __future__ import annotations

from dataclasses import dataclass, field
import time
import torch
import torch.nn.functional as F
from .graph import build_graph
from .reliability import estimate_prompt_reliability, estimate_sample_reliability
from .utils import l2_normalize

@dataclass
class SolverOutput:
    assignments: torch.Tensor
    prototypes: torch.Tensor
    prompt_weights: torch.Tensor | None
    class_prior: torch.Tensor
    sample_reliability: torch.Tensor | None
    iterations: int
    elapsed_seconds: float
    diagnostics: dict[str,object]=field(default_factory=dict)

def _initialize_mu(features:torch.Tensor,assignments:torch.Tensor,topk:int)->torch.Tensor:
    n,d=features.shape; k=assignments.shape[1]; count=min(max(1,topk),n); mu=torch.empty((k,d),device=features.device,dtype=features.dtype)
    for class_id in range(k):
        values,indices=assignments[:,class_id].topk(count); weights=values/values.sum().clamp_min(1e-8); mu[class_id]=(weights[:,None]*features[indices]).sum(dim=0)
    return l2_normalize(mu)

def _update_mu(features:torch.Tensor,assignments:torch.Tensor)->torch.Tensor:
    return l2_normalize((assignments.T@features)/assignments.sum(dim=0)[:,None].clamp_min(1e-8))

def _update_shared_variance(features:torch.Tensor,assignments:torch.Tensor,mu:torch.Tensor,floor:float,chunk_size:int=2048)->torch.Tensor:
    variance=torch.zeros(features.shape[1],device=features.device,dtype=features.dtype); total=assignments.sum().clamp_min(1e-8)
    for start in range(0,features.shape[0],chunk_size):
        end=min(start+chunk_size,features.shape[0]); diff2=(features[start:end,None,:]-mu[None,:,:]).square(); variance+=torch.einsum("nk,nkd->d",assignments[start:end],diff2)
    return (variance/total).clamp_min(floor)

def _gaussian_log_likelihood(features:torch.Tensor,mu:torch.Tensor,variance:torch.Tensor,chunk_size:int=2048)->torch.Tensor:
    outputs=[]; inv=variance.reciprocal(); log_det=variance.log().sum()
    for start in range(0,features.shape[0],chunk_size):
        end=min(start+chunk_size,features.shape[0]); diff2=(features[start:end,None,:]-mu[None,:,:]).square(); outputs.append(-0.5*(torch.einsum("nkd,d->nk",diff2,inv)+log_det))
    return torch.cat(outputs)

def _active_gate(probabilities:torch.Tensor,cfg:dict)->torch.Tensor:
    mass=probabilities.mean(dim=0); peak=probabilities.max(dim=0).values; uniform_mass=1.0/probabilities.shape[1]; mass_threshold=float(cfg.get("mass_ratio_threshold",0.20))*uniform_mass; temperature=max(float(cfg.get("gate_temperature",0.05)),1e-4); return torch.sigmoid((peak-float(cfg.get("peak_threshold",0.30)))/temperature)*torch.sigmoid((mass-mass_threshold)/temperature)

def _update_prior(assignments:torch.Tensor,initial_prior:torch.Tensor,previous_prior:torch.Tensor,cfg:dict)->torch.Tensor:
    k=assignments.shape[1]; alpha=float(cfg.get("dirichlet_alpha",0.5)); counts=assignments.sum(dim=0); posterior=(counts+alpha)/(counts.sum()+alpha*k); posterior=posterior*_active_gate(assignments,cfg); posterior=posterior+float(cfg.get("minimum_probability",1e-4)); posterior=posterior/posterior.sum(); anchor=float(cfg.get("anchor_weight",0.20)); posterior=(1.0-anchor)*posterior+anchor*initial_prior; ema=float(cfg.get("ema",0.5)); posterior=ema*previous_prior+(1.0-ema)*posterior; return posterior/posterior.sum().clamp_min(1e-8)

def zero_shot_assignments(image_features:torch.Tensor,text_prototypes:torch.Tensor,logit_scale:float)->torch.Tensor:
    return F.softmax(logit_scale*image_features@text_prototypes.T,dim=1)

def solve_rs_transclip(image_features:torch.Tensor,uniform_text_prototypes:torch.Tensor,cfg:dict)->SolverOutput:
    started=time.perf_counter(); image_features=l2_normalize(image_features); prototypes=l2_normalize(uniform_text_prototypes); y_hat=zero_shot_assignments(image_features,prototypes,cfg["zero_shot"]["logit_scale"]); z=y_hat.clone(); k=z.shape[1]; prior=torch.full((k,),1.0/k,device=z.device,dtype=z.dtype); mu=_initialize_mu(image_features,z,int(cfg["solver"]["prototype_topk"])); variance=torch.full((image_features.shape[1],),1.0/image_features.shape[1],device=z.device,dtype=z.dtype)
    graph=build_graph(image_features,k=int(cfg["graph"]["k"]),backend=cfg["graph"]["backend"],mutual=False,kernel="cosine",reliability=None,chunk_size=int(cfg["graph"]["chunk_size"]),minimum_similarity=float(cfg["graph"].get("minimum_similarity",0.0)))
    completed=0
    for outer in range(int(cfg["solver"]["outer_iterations"])):
        likelihood=_gaussian_log_likelihood(image_features,mu,variance)
        for _ in range(int(cfg["solver"]["assignment_iterations"])):
            propagated=torch.sparse.mm(graph.matrix,z); logits=likelihood/float(cfg["solver"]["likelihood_temperature"])+float(cfg["solver"]["graph_strength"])*propagated+float(cfg["solver"]["text_anchor_strength"])*y_hat.clamp_min(1e-8).log(); z=F.softmax(logits,dim=1)
        mu=_update_mu(image_features,z); variance=_update_shared_variance(image_features,z,mu,float(cfg["solver"]["covariance_floor"])); completed=outer+1
    return SolverOutput(z,mu,None,prior,None,completed,time.perf_counter()-started,{"graph_backend":graph.backend,"graph_edges":graph.num_edges})

def solve_rap_transclip(image_features:torch.Tensor,all_text_prototypes:torch.Tensor,cfg:dict)->SolverOutput:
    started=time.perf_counter(); image_features=l2_normalize(image_features); all_text_prototypes=l2_normalize(all_text_prototypes); logit_scale=float(cfg["zero_shot"]["logit_scale"]); prompt_result=estimate_prompt_reliability(image_features,all_text_prototypes,cfg["prompt_reliability"],logit_scale=logit_scale); y_hat=zero_shot_assignments(image_features,prompt_result.prototypes,logit_scale); sample_reliability=estimate_sample_reliability(y_hat,prompt_result.probabilities,cfg["sample_reliability"])
    g=cfg["graph"]; graph=build_graph(image_features,k=int(g["k"]),backend=g["backend"],mutual=bool(g["mutual"]),kernel=g["kernel"],local_scale_rank=int(g.get("local_scale_rank",g["k"])),reliability=sample_reliability if g.get("reliability_weighting",True) else None,chunk_size=int(g["chunk_size"]),minimum_similarity=float(g.get("minimum_similarity",0.0)))
    z=y_hat.clone(); initial_prior=z.mean(dim=0); initial_prior=initial_prior/initial_prior.sum().clamp_min(1e-8); prior=initial_prior.clone(); mu=_initialize_mu(image_features,z,int(cfg["solver"]["prototype_topk"])); variance=torch.full((image_features.shape[1],),1.0/image_features.shape[1],device=z.device,dtype=z.dtype); completed=0; previous_z=z
    for outer in range(int(cfg["solver"]["outer_iterations"])):
        likelihood=_gaussian_log_likelihood(image_features,mu,variance)
        for _ in range(int(cfg["solver"]["assignment_iterations"])):
            propagated=torch.sparse.mm(graph.matrix,z); logits=likelihood/float(cfg["solver"]["likelihood_temperature"])+prior.clamp_min(1e-8).log()[None,:]+float(cfg["solver"]["graph_strength"])*propagated+float(cfg["solver"]["text_anchor_strength"])*y_hat.clamp_min(1e-8).log(); z=F.softmax(logits,dim=1)
        mu=_update_mu(image_features,z); variance=_update_shared_variance(image_features,z,mu,float(cfg["solver"]["covariance_floor"]))
        if cfg["solver"].get("use_active_prior",True) and cfg["active_prior"].get("enabled",True): prior=_update_prior(z,initial_prior,prior,cfg["active_prior"])
        if cfg["solver"].get("update_prompt_weights",True): prompt_result=estimate_prompt_reliability(image_features,all_text_prototypes,cfg["prompt_reliability"],logit_scale=logit_scale,target_assignments=z,previous_weights=prompt_result.weights); y_hat=zero_shot_assignments(image_features,prompt_result.prototypes,logit_scale)
        delta=(z-previous_z).abs().mean().item(); previous_z=z.clone(); completed=outer+1
        if delta<float(cfg["solver"].get("convergence_tolerance",1e-5)): break
    return SolverOutput(z,mu,prompt_result.weights,prior,sample_reliability,completed,time.perf_counter()-started,{"graph_backend":graph.backend,"graph_edges":graph.num_edges,"mean_sample_reliability":float(sample_reliability.mean().item()),"active_prior_entropy":float(-(prior.clamp_min(1e-8)*prior.clamp_min(1e-8).log()).sum().item())})
