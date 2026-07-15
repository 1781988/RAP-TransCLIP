#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy
from rap_transclip.config import apply_overrides,load_config
from rap_transclip.runner import run_experiment

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/standard.yaml"); p.add_argument("--datasets",nargs="+"); p.add_argument("--model",default="GeoRSCLIP"); p.add_argument("--architecture",default="ViT-L-14"); p.add_argument("--extended",action="store_true"); args=p.parse_args(); base=load_config(args.config); datasets=args.datasets or list(base["datasets"])
    variants=[("A0_rs_transclip","rs_transclip",[]),("A1_prompt_only","rap_transclip",["solver.use_active_prior=false","graph.reliability_weighting=false"]),("A2_prompt_prior","rap_transclip",["solver.use_active_prior=true","graph.reliability_weighting=false"]),("A3_full","rap_transclip",[])]
    if args.extended: variants += [("no_confidence","rap_transclip",["prompt_reliability.confidence_weight=0.0"]),("no_entropy","rap_transclip",["prompt_reliability.entropy_weight=0.0"]),("no_margin","rap_transclip",["prompt_reliability.margin_weight=0.0"]),("no_agreement","rap_transclip",["prompt_reliability.agreement_weight=0.0"]),("directed_cosine_graph","rap_transclip",["graph.mutual=false","graph.kernel=cosine","graph.reliability_weighting=false"]),("mutual_rbf_no_reliability","rap_transclip",["graph.mutual=true","graph.kernel=rbf","graph.reliability_weighting=false"])]
    for tag,method,overrides in variants:
        cfg=apply_overrides(copy.deepcopy(base),overrides); cfg.setdefault("runtime",{})["experiment_tag"]=tag
        for dataset in datasets: run_experiment(cfg,dataset,args.model,args.architecture,method)
if __name__=="__main__": main()
