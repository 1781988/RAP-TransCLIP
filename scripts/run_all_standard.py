#!/usr/bin/env python3
from __future__ import annotations
import argparse
from rap_transclip.config import apply_overrides,load_config
from rap_transclip.feature_extraction import extract_features
from rap_transclip.runner import run_experiment

def selected(values,defaults): return values if values else defaults

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/standard.yaml"); p.add_argument("--stage",choices=["features","evaluate","all"],default="all"); p.add_argument("--datasets",nargs="+"); p.add_argument("--models",nargs="+"); p.add_argument("--architectures",nargs="+"); p.add_argument("--methods",nargs="+",default=["zero_shot","rs_transclip","rap_transclip"]); p.add_argument("--override",action="append",default=[]); p.add_argument("--overwrite",action="store_true"); args=p.parse_args()
    cfg=apply_overrides(load_config(args.config),args.override); datasets=selected(args.datasets,list(cfg["datasets"])); enabled_models=[name for name,spec in cfg["models"].items() if spec.get("enabled",False)]; models=selected(args.models,enabled_models)
    for dataset in datasets:
        for model in models:
            available=list(cfg["models"][model]["architectures"].keys()); architectures=selected(args.architectures,available)
            for architecture in architectures:
                if architecture not in cfg["models"][model]["architectures"]: continue
                print(f"\n=== {dataset} | {model} | {architecture} ===")
                if args.stage in {"features","all"}: extract_features(cfg,dataset,model,architecture,args.overwrite)
                if args.stage in {"evaluate","all"}:
                    for method in args.methods: run_experiment(cfg,dataset,model,architecture,method)
if __name__=="__main__": main()
