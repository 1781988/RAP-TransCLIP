from __future__ import annotations
import copy
from pathlib import Path
import torch
from rap_transclip.config import load_config
from rap_transclip.solver import solve_rap_transclip,solve_rs_transclip
from rap_transclip.utils import l2_normalize

def synthetic_problem(seed:int=3):
    generator=torch.Generator().manual_seed(seed); n_per_class,classes,prompts,dim=12,3,7,16; centers=l2_normalize(torch.randn(classes,dim,generator=generator)); features=[]
    for class_id in range(classes): features.append(l2_normalize(centers[class_id]+0.12*torch.randn(n_per_class,dim,generator=generator)))
    images=torch.cat(features); texts=[]
    for prompt_id in range(prompts): texts.append(l2_normalize(centers+(0.04+prompt_id*0.01)*torch.randn(classes,dim,generator=generator)))
    texts_all=torch.stack(texts); return images,texts_all,l2_normalize(texts_all.mean(0))
def cpu_config():
    cfg=copy.deepcopy(load_config(Path(__file__).parents[1]/"configs"/"standard.yaml")); cfg["project"]["device"]="cpu"; cfg["graph"]["backend"]="torch"; cfg["graph"]["k"]=3; cfg["graph"]["chunk_size"]=64; cfg["solver"]["outer_iterations"]=2; cfg["solver"]["assignment_iterations"]=2; return cfg
def assert_probabilities(tensor):
    assert torch.isfinite(tensor).all(); assert torch.allclose(tensor.sum(dim=1),torch.ones(tensor.shape[0]),atol=1e-5)
def test_rs_transclip_smoke():
    images,_,texts_uniform=synthetic_problem(); output=solve_rs_transclip(images,texts_uniform,cpu_config()); assert output.assignments.shape==(36,3); assert_probabilities(output.assignments)
def test_rap_transclip_smoke():
    images,texts_all,_=synthetic_problem(); output=solve_rap_transclip(images,texts_all,cpu_config()); assert output.assignments.shape==(36,3); assert output.prompt_weights is not None; assert output.prompt_weights.shape==(7,3); assert_probabilities(output.assignments); assert torch.allclose(output.prompt_weights.sum(dim=0),torch.ones(3),atol=1e-5)
