"""Tensor-Ring fitting from a GPU-resident sparse observation reservoir.

CuPy arrays are consumed by PyTorch through DLPack.  No observation matrix or
DP field is copied through host RAM.  Only final scalar metrics are returned.
"""
from __future__ import annotations
import math, time


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def tr_parameter_count(shape, rank):
    r=int(rank); return int(sum(int(n)*r*r for n in shape))


def _from_dlpack(x):
    import torch
    # torch.from_dlpack accepts objects implementing __dlpack__ on modern builds.
    try:
        return torch.from_dlpack(x)
    except Exception:
        from torch.utils.dlpack import from_dlpack
        return from_dlpack(x.toDlpack())


def tr_predict(cores, coords):
    import torch
    M=None
    for d,G in enumerate(cores):
        A=G[:,coords[:,d],:].permute(1,0,2).contiguous()
        M=A if M is None else torch.bmm(M,A)
    return M.diagonal(dim1=1,dim2=2).sum(-1)


def fit_tr_from_cupy(coords_cp, target_cp, shape, rank=4, epochs=60,
                     batch_size=65536, lr=2e-3, weight_decay=0.0,
                     val_fraction=0.1, seed=12345, functional_coord_cp=None,
                     functional_target=None, functional_weight=20.0,
                     dtype='float32', early_stop_patience=8):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('PyTorch CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32=True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    tdtype=torch.float32 if dtype=='float32' else torch.float64
    coords=_from_dlpack(coords_cp).to(dtype=torch.long)
    y=_from_dlpack(target_cp).to(dtype=tdtype)
    n=int(coords.shape[0])
    if n<8: raise ValueError('too few observations for Tensor-Ring fitting')
    gen=torch.Generator(device='cuda'); gen.manual_seed(int(seed))
    perm=torch.randperm(n,generator=gen,device='cuda')
    nv=max(1,int(round(float(val_fraction)*n))); vi=perm[:nv]; ti=perm[nv:]
    if ti.numel()==0: ti=perm
    r=int(rank); k=len(shape)
    # Scale initial cores to the target magnitude without copying the targets.
    yscale=float(y.abs().mean().clamp_min(1e-12).item())
    scale=(yscale**(1.0/max(1,k)))/max(1,r)
    cores=torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn((r,int(dim),r),device='cuda',dtype=tdtype,generator=gen)*scale)
        for dim in shape
    ])
    opt=torch.optim.Adam(cores.parameters(),lr=float(lr),weight_decay=float(weight_decay))
    fcoord=None; ft=None
    if functional_coord_cp is not None and functional_target is not None:
        fcoord=_from_dlpack(functional_coord_cp).to(dtype=torch.long)
        if fcoord.ndim==1: fcoord=fcoord[None,:]
        ft=torch.tensor(float(functional_target),device='cuda',dtype=tdtype)
    bs=max(256,int(batch_size)); best_val=math.inf; best=None; noimp=0; hist=[]
    torch.cuda.synchronize(); t0=time.perf_counter()
    for ep in range(int(epochs)):
        order=ti[torch.randperm(int(ti.numel()),generator=gen,device='cuda')]
        for p in range(0,int(order.numel()),bs):
            ids=order[p:p+bs]
            pred=tr_predict(cores,coords[ids]); loss=torch.mean((pred-y[ids])**2)
            if fcoord is not None:
                fp=tr_predict(cores,fcoord)[0]
                denom=torch.clamp(ft.abs(),min=torch.tensor(1.0,device='cuda',dtype=tdtype))
                loss=loss+float(functional_weight)*((fp-ft)/denom)**2
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            vp=tr_predict(cores,coords[vi]); vy=y[vi]
            den=torch.linalg.vector_norm(vy).clamp_min(torch.finfo(tdtype).tiny)
            verr=float((torch.linalg.vector_norm(vp-vy)/den).item())
        hist.append(verr)
        if verr < best_val*(1-1e-5):
            best_val=verr; best=[p.detach().clone() for p in cores]; noimp=0
        else: noimp+=1
        if noimp>=int(early_stop_patience) and ep>=10: break
    torch.cuda.synchronize(); fit_sec=time.perf_counter()-t0
    fp=float('nan')
    if fcoord is not None and best is not None:
        with torch.no_grad(): fp=float(tr_predict(best,fcoord)[0].item())
    # Peak allocated excludes the upstream CuPy pool; experiment logs both pools separately.
    peak=int(torch.cuda.max_memory_allocated())
    return dict(val_rel_error=float(best_val),fit_sec=float(fit_sec),epochs_run=len(hist),
                n_fit_samples=n,n_train=int(ti.numel()),n_val=int(vi.numel()),
                functional_pred_scaled=fp,params=tr_parameter_count(shape,r),
                peak_torch_bytes=peak,rank=r,dtype=dtype)
