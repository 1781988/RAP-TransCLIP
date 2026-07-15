from __future__ import annotations
from pathlib import Path

def load_prompt_prefixes(path:str|Path)->list[str]:
    prefixes=[line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    prefixes=[line for line in prefixes if line and not line.startswith("#")]
    if not prefixes: raise ValueError(f"No prompts found in {path}")
    return prefixes

def instantiate_prompts(prefixes:list[str],class_names:list[str])->list[list[str]]:
    return [[f"{prefix} {name}." for name in class_names] for prefix in prefixes]
