#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, tempfile, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download
import requests
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[1]; CHECKPOINTS=ROOT/"checkpoints"
REMOTE={"RN50":"RemoteCLIP-RN50.pt","ViT-B-32":"RemoteCLIP-ViT-B-32.pt","ViT-L-14":"RemoteCLIP-ViT-L-14.pt"}
GEO={"ViT-B-32":"ckpt/RS5M_ViT-B-32.pt","ViT-L-14":"ckpt/RS5M_ViT-L-14.pt","ViT-H-14":"ckpt/RS5M_ViT-H-14.pt"}
SKY={"ViT-B-32":("https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip","SkyCLIP_ViT_B32_top50pct_epoch_20.pt"),"ViT-L-14":("https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_L14_top50pct.zip","SkyCLIP_ViT_L14_top50pct_epoch_20.pt")}
def copy_hf(repo_id,filename,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): print(f"Exists: {target}"); return
    shutil.copy2(Path(hf_hub_download(repo_id=repo_id,filename=filename)),target); print(f"Saved: {target}")
def download_zip(url,output):
    with requests.get(url,stream=True,timeout=60) as response:
        response.raise_for_status(); total=int(response.headers.get("content-length",0))
        with output.open("wb") as handle,tqdm(total=total,unit="B",unit_scale=True,desc=output.name) as bar:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk: handle.write(chunk); bar.update(len(chunk))
def copy_sky(architecture,target):
    if target.exists(): print(f"Exists: {target}"); return
    url,_=SKY[architecture]; target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td); archive=temp/"checkpoint.zip"; download_zip(url,archive)
        with zipfile.ZipFile(archive) as zf: zf.extractall(temp/"extracted")
        candidates=list((temp/"extracted").rglob("epoch_20.pt"))
        if not candidates: raise FileNotFoundError("Could not locate epoch_20.pt")
        shutil.copy2(candidates[0],target)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--models",nargs="+",default=["RemoteCLIP","GeoRSCLIP","SkyCLIP50"],choices=["RemoteCLIP","GeoRSCLIP","SkyCLIP50"]); p.add_argument("--architectures",nargs="+",default=None); args=p.parse_args(); requested=set(args.architectures or [])
    if "RemoteCLIP" in args.models:
        for arch,filename in REMOTE.items():
            if not requested or arch in requested: copy_hf("chendelong/RemoteCLIP",filename,CHECKPOINTS/"RemoteCLIP"/filename)
    if "GeoRSCLIP" in args.models:
        for arch,filename in GEO.items():
            if not requested or arch in requested: copy_hf("Zilun/GeoRSCLIP",filename,CHECKPOINTS/"GeoRSCLIP"/Path(filename).name)
    if "SkyCLIP50" in args.models:
        for arch,(_,target_name) in SKY.items():
            if not requested or arch in requested: copy_sky(arch,CHECKPOINTS/"SkyCLIP50"/target_name)
if __name__=="__main__": main()
