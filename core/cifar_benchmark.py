"""Optional real CIFAR-10 benchmark with independent retraining references.

No synthetic fallback is used. Candidate and calibration/reference models are
trained with disjoint seeds; never reuse a reference model as the candidate.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, time
import numpy as np

@dataclass
class CifarConfig:
    data_root: str = "data/cifar10"
    forget_fraction: float = 0.10
    seed: int = 42
    reference_retrains: int = 3
    epochs: int = 2
    batch_size: int = 128
    max_train_samples: int | None = 5000
    architecture: str = "both"  # cnn, resnet18, or both
    device: str = "cpu"


def _deps():
    try:
        import torch
        from torchvision import datasets, transforms, models
        return torch, datasets, transforms, models
    except Exception as exc:
        raise RuntimeError(f"CIFAR benchmark requires compatible torch + torchvision: {exc}") from exc


def build_cifar10_split(cfg: CifarConfig):
    torch, datasets, transforms, _ = _deps()
    root=Path(cfg.data_root)
    tfm=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train=datasets.CIFAR10(root=str(root),train=True,download=True,transform=tfm)
    test=datasets.CIFAR10(root=str(root),train=False,download=True,transform=tfm)
    rng=np.random.RandomState(cfg.seed)
    idx=np.arange(len(train)); rng.shuffle(idx)
    if cfg.max_train_samples: idx=idx[:min(len(idx),cfg.max_train_samples)]
    n_forget=max(1,int(round(len(idx)*cfg.forget_fraction)))
    forget_idx=idx[:n_forget]; retain_idx=idx[n_forget:]
    return train,test,retain_idx,forget_idx


def make_cnn(num_classes=10):
    _,_,_,_= _deps()
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),
        nn.Linear(64*8*8,128),nn.ReLU(),nn.Linear(128,num_classes))


def make_resnet18(num_classes=10):
    """CIFAR-adapted ResNet-18 trained from scratch (no pretrained weights)."""
    _,_,_,models = _deps()
    model=models.resnet18(weights=None, num_classes=num_classes)
    import torch.nn as nn
    model.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity()
    return model


def _make_model(architecture):
    if architecture == "cnn": return make_cnn()
    if architecture == "resnet18": return make_resnet18()
    raise ValueError("architecture must be 'cnn' or 'resnet18'")


def train_model(dataset, indices, seed, epochs=2, batch_size=128, device="cpu", architecture="cnn"):
    torch,_,_,_ = _deps()
    from torch.utils.data import DataLoader, Subset
    torch.manual_seed(seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model=_make_model(architecture).to(device)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(Subset(dataset,indices.tolist()),batch_size=batch_size,shuffle=True,num_workers=0,generator=generator)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()
    return model


def evaluate_model(model,dataset,indices,batch_size=256,device="cpu"):
    torch,_,_,_ = _deps()
    from torch.utils.data import DataLoader, Subset
    loader=DataLoader(Subset(dataset,indices.tolist()),batch_size=batch_size,shuffle=False,num_workers=0)
    correct=total=0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            pred=model(x.to(device)).argmax(1).cpu(); correct += int((pred==y).sum()); total += len(y)
    return float(correct/max(1,total))


def _split_hash(retain_idx, forget_idx):
    payload=np.concatenate([np.asarray(retain_idx,dtype=np.int64),[-1],np.asarray(forget_idx,dtype=np.int64)]).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _run_architecture(cfg, train, test, retain_idx, forget_idx, architecture):
    t0=time.time(); device=cfg.device
    candidate_seed=cfg.seed+1
    candidate_model=train_model(train,retain_idx,candidate_seed,cfg.epochs,cfg.batch_size,device,architecture)
    candidate={"seed":candidate_seed,"retain_accuracy":evaluate_model(candidate_model,train,retain_idx,cfg.batch_size,device),
        "forget_accuracy":evaluate_model(candidate_model,train,forget_idx,cfg.batch_size,device),
        "test_accuracy":evaluate_model(candidate_model,test,np.arange(len(test)),cfg.batch_size,device)}
    refs=[]
    for i in range(cfg.reference_retrains):
        seed=cfg.seed+100+i
        m=train_model(train,retain_idx,seed,cfg.epochs,cfg.batch_size,device,architecture)
        refs.append({"seed":seed,"retain_accuracy":evaluate_model(m,train,retain_idx,cfg.batch_size,device),
            "forget_accuracy":evaluate_model(m,train,forget_idx,cfg.batch_size,device),
            "test_accuracy":evaluate_model(m,test,np.arange(len(test)),cfg.batch_size,device)})
    return {"architecture":architecture,"candidate":candidate,"reference_distribution":refs,
        "runtime_seconds":time.time()-t0,"candidate_seed_disjoint_from_references":candidate_seed not in {r['seed'] for r in refs}}


def run_cifar_reference_benchmark(cfg: CifarConfig | None = None):
    """Train independent candidate and reference ensembles for one or both architectures."""
    cfg=cfg or CifarConfig()
    if not 0 < cfg.forget_fraction < .5: raise ValueError("forget_fraction must be between 0 and 0.5")
    if cfg.reference_retrains < 2: raise ValueError("at least two independent reference retrains are required")
    if cfg.architecture not in {"cnn","resnet18","both"}: raise ValueError("architecture must be cnn, resnet18, or both")
    torch,_,_,_=_deps()
    started=time.time(); train,test,retain_idx,forget_idx=build_cifar10_split(cfg)
    archs=["cnn","resnet18"] if cfg.architecture=="both" else [cfg.architecture]
    results={a:_run_architecture(cfg,train,test,retain_idx,forget_idx,a) for a in archs}
    return {"config":cfg.__dict__,"dataset":"CIFAR-10","dataset_version":"torchvision.datasets.CIFAR10",
        "split_sha256":_split_hash(retain_idx,forget_idx),"train_pool_size":len(retain_idx)+len(forget_idx),
        "retain_size":len(retain_idx),"forget_size":len(forget_idx),"architectures":results,
        "runtime_seconds":time.time()-started,"status":"completed",
        "scientific_note":"Candidate and references use disjoint seeds but the same retain split. Results are diagnostic, not publication-grade evidence until methods, multiple forget fractions, repeated splits, and held-out attack conditions are evaluated."}
