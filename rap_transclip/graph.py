from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import torch

@dataclass
class GraphResult:
    matrix: torch.Tensor
    backend: str
    num_edges: int

def _faiss_knn(features: np.ndarray,k:int):
    import faiss
    index=faiss.IndexFlatIP(features.shape[1]); index.add(features.astype(np.float32,copy=False))
    s,i=index.search(features.astype(np.float32,copy=False),k+1); return s[:,1:],i[:,1:]

def _torch_knn(features:torch.Tensor,k:int,chunk_size:int):
    vals=[]; inds=[]; n=features.shape[0]
    for start in range(0,n,chunk_size):
        end=min(start+chunk_size,n); sim=features[start:end]@features.T
        sim[torch.arange(end-start,device=features.device),torch.arange(start,end,device=features.device)]=-float("inf")
        v,i=sim.topk(k=k,dim=1); vals.append(v.cpu()); inds.append(i.cpu())
    return torch.cat(vals).numpy(),torch.cat(inds).numpy()

def build_graph(features:torch.Tensor,k:int=5,backend:str="auto",mutual:bool=True,kernel:str="rbf",local_scale_rank:int=5,reliability:torch.Tensor|None=None,chunk_size:int=2048,minimum_similarity:float=0.0)->GraphResult:
    if features.ndim!=2: raise ValueError("features must have shape [N, D]")
    n=features.shape[0]
    if n<=1:
        idx=torch.zeros((2,0),dtype=torch.long,device=features.device); val=torch.zeros(0,device=features.device)
        return GraphResult(torch.sparse_coo_tensor(idx,val,(n,n)).coalesce(),"none",0)
    k=min(k,n-1); selected_backend=backend
    if backend in {"auto","faiss"}:
        try: values_np,indices_np=_faiss_knn(features.detach().cpu().numpy(),k); selected_backend="faiss"
        except Exception:
            if backend=="faiss": raise
            values_np,indices_np=_torch_knn(features,k,chunk_size); selected_backend="torch"
    elif backend=="torch": values_np,indices_np=_torch_knn(features,k,chunk_size)
    else: raise ValueError(f"Unsupported graph backend: {backend}")
    similarities=np.maximum(values_np,minimum_similarity); rows=np.repeat(np.arange(n),k); cols=indices_np.reshape(-1)
    if kernel=="cosine": weights=similarities.reshape(-1)
    elif kernel=="rbf":
        rank=min(max(1,local_scale_rank),k)-1; local_scale=np.maximum(1.0-similarities[:,rank],1e-4)
        weights=np.exp(-np.maximum(1.0-similarities,0.0).reshape(-1)/np.repeat(local_scale,k))
    else: raise ValueError(f"Unsupported graph kernel: {kernel}")
    directed=sp.csr_matrix((weights,(rows,cols)),shape=(n,n)); graph=directed.minimum(directed.T) if mutual else directed.maximum(directed.T)
    graph.setdiag(0); graph.eliminate_zeros()
    if reliability is not None:
        q=reliability.detach().cpu().numpy().astype(np.float64); graph=sp.diags(q)@graph@sp.diags(q)
    degree=np.asarray(graph.sum(axis=1)).reshape(-1); inv=np.zeros_like(degree); mask=degree>0; inv[mask]=1.0/np.sqrt(degree[mask]); graph=(sp.diags(inv)@graph@sp.diags(inv)).tocoo()
    idx=torch.tensor(np.vstack([graph.row,graph.col]),dtype=torch.long,device=features.device); ev=torch.tensor(graph.data,dtype=features.dtype,device=features.device)
    sparse=torch.sparse_coo_tensor(idx,ev,(n,n),device=features.device).coalesce(); return GraphResult(sparse,selected_backend,int(sparse._nnz()))
