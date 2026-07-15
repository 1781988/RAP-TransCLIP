from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import IndexedImageDataset, load_dataset_metadata
from .models import load_vlm
from .prompts import instantiate_prompts, load_prompt_prefixes
from .utils import l2_normalize, write_json


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def extract_features(cfg: dict,dataset_name: str,model_name: str,architecture: str,overwrite: bool=False)->Path:
    device_name=cfg["project"].get("device","cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available(): device_name="cpu"
    device=torch.device(device_name)
    model_cfg=cfg["models"][model_name]; arch_cfg=model_cfg["architectures"][architecture]
    loaded=load_vlm(model_cfg["backend"],architecture,arch_cfg.get("pretrained"),arch_cfg.get("checkpoint"),device)
    output_dir=Path(cfg["paths"]["features"])/dataset_name/model_name/architecture
    marker=output_dir/"metadata.json"
    if marker.exists() and not overwrite:
        print(f"Features already exist; skipping: {output_dir}"); return output_dir
    output_dir.mkdir(parents=True,exist_ok=True)
    index_path=Path(cfg["paths"]["indexes"])/f"{dataset_name}.jsonl"
    dataset=IndexedImageDataset(index_path,loaded.preprocess)
    loader=DataLoader(dataset,batch_size=int(cfg["feature_extraction"]["batch_size"]),shuffle=False,num_workers=int(cfg["feature_extraction"]["num_workers"]),pin_memory=device.type=="cuda")
    image_features=[]; labels=[]; image_paths=[]; started=time.perf_counter()
    with torch.inference_mode():
        for images,batch_labels,batch_paths in tqdm(loader,desc="image features"):
            images=images.to(device,non_blocking=True)
            with _autocast_context(device,bool(cfg["feature_extraction"].get("amp",True))): features=loaded.model.encode_image(images)
            image_features.append(l2_normalize(features.float()).cpu()); labels.append(batch_labels.long().cpu()); image_paths.extend(batch_paths)
    all_images=torch.cat(image_features); all_labels=torch.cat(labels)
    metadata=load_dataset_metadata(cfg["paths"]["datasets"],dataset_name)
    prefixes=load_prompt_prefixes(cfg["feature_extraction"]["prompt_file"]); prompt_grid=instantiate_prompts(prefixes,metadata.class_names)
    text_features=[]
    with torch.inference_mode():
        for class_prompts in tqdm(prompt_grid,desc="text prompts"):
            tokens=loaded.tokenizer(class_prompts).to(device)
            with _autocast_context(device,bool(cfg["feature_extraction"].get("amp",True))): features=loaded.model.encode_text(tokens)
            text_features.append(l2_normalize(features.float()).cpu())
    texts_all=torch.stack(text_features); texts_uniform=l2_normalize(texts_all.mean(dim=0))
    torch.save(all_images,output_dir/"images.pt"); torch.save(all_labels,output_dir/"labels.pt"); torch.save(texts_all,output_dir/"texts_all.pt"); torch.save(texts_uniform,output_dir/"texts_uniform.pt")
    write_json(output_dir/"classes.json",{"tokens":metadata.class_tokens,"names":metadata.class_names}); write_json(output_dir/"image_paths.json",image_paths)
    write_json(output_dir/"metadata.json",{"dataset":dataset_name,"model":model_name,"architecture":architecture,"num_images":int(all_images.shape[0]),"num_classes":int(texts_all.shape[1]),"num_prompts":int(texts_all.shape[0]),"embedding_dim":int(all_images.shape[1]),"elapsed_seconds":time.perf_counter()-started})
    return output_dir
