#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy
from rap_transclip.config import load_config
from rap_transclip.runner import run_experiment

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/standard.yaml"); p.add_argument("--datasets",nargs="+"); p.add_argument("--model",default="GeoRSCLIP"); p.add_argument("--architecture",default="ViT-L-14"); args=p.parse_args(); base=load_config(args.config); datasets=args.datasets or list(base["datasets"]); protocols=[("partial_25","partial_class",{"class_fraction":0.25}),("partial_50","partial_class",{"class_fraction":0.50}),("partial_75","partial_class",{"class_fraction":0.75}),("long_tail_01","dirichlet_long_tail",{"alpha":0.1}),("long_tail_05","dirichlet_long_tail",{"alpha":0.5}),("long_tail_10","dirichlet_long_tail",{"alpha":1.0})]
    for tag,protocol,protocol_args in protocols:
        cfg=copy.deepcopy(base); cfg.setdefault("runtime",{})["experiment_tag"]=tag
        for dataset in datasets:
            for method in ("rs_transclip","rap_transclip"): run_experiment(cfg,dataset,args.model,args.architecture,method,protocol_name=protocol,protocol_args=protocol_args)
if __name__=="__main__": main()
