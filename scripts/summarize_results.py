#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="outputs/results/raw_results.csv"); p.add_argument("--output",default="outputs/results/summary.csv"); args=p.parse_args(); frame=pd.read_csv(args.input); metrics=["top1","macro_f1","ece","solver_seconds","peak_cuda_memory_mb"]; group=["dataset","model","architecture","method","protocol"]; summary=frame.groupby(group,dropna=False)[metrics].agg(["mean","std","count"]).reset_index(); output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); summary.to_csv(output,index=False); print(summary.to_string(index=False)); print(f"\nSaved: {output}")
if __name__=="__main__": main()
