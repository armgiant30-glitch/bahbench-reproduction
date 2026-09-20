"""Paper-style downstream Transformer for BAHBench.

Paper pipeline:
    audio -> frozen foundation encoder -> projection to d_model (768)
          -> one standard Transformer encoder layer -> classifier on token 0.
"""
import argparse, json, math, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import load_manifest, read_audio, set_seed, split_by_label
from metrics import compute_metrics
from lora import LORA_TARGET_NAMES, inject_lora


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/hubert-base-ls960')
    p.add_argument('--manifest', required=True)
    p.add_argument('--tiny-model', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--max-seconds', type=float, default=6.0)
    p.add_argument('--layer', type=int, default=-1, help='hidden_states index; -1 means last_hidden_state')
    p.add_argument('--d-model', type=int, default=768)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--dim-feedforward', type=int, default=3072)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--mode', default='frozen', choices=['frozen','lora','full'])
    p.add_argument('--r', type=int, default=16)
    p.add_argument('--lora-targets', default='qkv', choices=['qkv','qv'])
    p.add_argument('--class-weighted', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    p.add_argument('--out', default='results/paper_transformer')
    return p.parse_args()


def pick_device(s):
    if s == 'cpu': return torch.device('cpu')
    if s == 'cuda': return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prep_items(args):
    items = load_manifest(args.manifest)
    if args.limit:
        items = items[:args.limit]
    data = []
    for path, label in items:
        try:
            w = read_audio(path, target_sr=16000)
        except Exception:
            continue
        if w.ndim == 2: w = w[0]
        if args.max_seconds:
            w = w[:int(args.max_seconds * 16000)]
        if w.numel() < 1600:
            continue
        w = (w - w.mean()) / torch.sqrt(w.var() + 1e-6)
        data.append((w, str(label)))
    classes = sorted({x[1] for x in data})
    lab2id = {c:i for i,c in enumerate(classes)}
    y = np.array([lab2id[l] for _,l in data], dtype=np.int64)
    return data, y, classes


def batch_iter(data, y, idx, batch_size, shuffle, seed):
    idx = list(idx)
    if shuffle:
        random.Random(seed).shuffle(idx)
    for s in range(0, len(idx), batch_size):
        sel = idx[s:s+batch_size]
        xs = [data[i][0] for i in sel]
        L = max(x.numel() for x in xs)
        x = torch.zeros(len(sel), L)
        mask = torch.zeros(len(sel), L, dtype=torch.long)
        for j, w in enumerate(xs):
            x[j,:w.numel()] = w
            mask[j,:w.numel()] = 1
        yield x, mask, torch.tensor([y[i] for i in sel], dtype=torch.long)


class PaperTransformer(nn.Module):
    def __init__(self, foundation, n_classes, d_model, nhead, num_layers,
                 dim_feedforward, dropout, mode='frozen', r=16, lora_targets='qkv', layer=-1):
        super().__init__()
        self.foundation = foundation
        self.mode = mode
        self.layer = layer
        for p in self.foundation.parameters():
            p.requires_grad_(False)
        if mode == 'lora':
            count, names = inject_lora(self.foundation, r=r,
                                       targets=LORA_TARGET_NAMES[lora_targets])
            if count == 0:
                raise RuntimeError('No Q/K/V linear layers found for LoRA')
            print(f'[lora] injected {count} modules, r={r}, targets={lora_targets}', flush=True)
        elif mode == 'full':
            for p in self.foundation.parameters():
                p.requires_grad_(True)
        in_dim = foundation.config.hidden_size
        self.projection = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1,4096,d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, mask):
        if self.mode == 'frozen':
            self.foundation.eval()
            with torch.no_grad():
                out = self.foundation(input_values=x, output_hidden_states=True)
                h = out.last_hidden_state if self.layer < 0 else out.hidden_states[self.layer]
        else:
            out = self.foundation(input_values=x, output_hidden_states=True)
            h = out.last_hidden_state if self.layer < 0 else out.hidden_states[self.layer]
        h = self.projection(h)
        L = min(h.size(1), self.pos_embed.size(1))
        h = h[:,:L] + self.pos_embed[:,:L]
        if mask.size(1) >= L:
            mbase = mask[:,:L]
        else:
            extra = torch.ones(mask.size(0), L-mask.size(1), device=mask.device, dtype=mask.dtype)
            mbase = torch.cat([mask, extra], dim=1)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        m = torch.cat([torch.ones(mask.size(0),1,device=mask.device,dtype=mask.dtype), mbase], dim=1)
        h = self.encoder(h, src_key_padding_mask=(m == 0))
        return self.classifier(h[:,0])


def evaluate(model, data, y, idx, batch_size, device):
    model.eval(); preds=[]; ys=[]
    with torch.no_grad():
        for x,m,lab in batch_iter(data,y,idx,batch_size,False,0):
            logits = model(x.to(device), m.to(device))
            preds += logits.argmax(1).cpu().tolist(); ys += lab.tolist()
    return np.array(ys), np.array(preds)


def main():
    args = parse_args(); set_seed(args.seed); device = pick_device(args.device)
    print(f'[env] device={device}')
    if args.tiny_model:
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        cfg = Wav2Vec2Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=2,
                             intermediate_size=128, conv_dim=(32,32,32,32,32,32,32),
                             conv_stride=(5,2,2,2,2,2,2), conv_kernel=(10,3,3,3,3,2,2),
                             num_conv_pos_embeddings=32, num_conv_pos_embedding_groups=4)
        foundation = Wav2Vec2Model(cfg)
        if args.d_model == 768:
            args.d_model, args.nhead, args.dim_feedforward = 64, 4, 256
    else:
        from transformers import AutoModel
        foundation = AutoModel.from_pretrained(args.model)
    data, y, classes = prep_items(args)
    tr, va, te = split_by_label([(i,str(v)) for i,v in enumerate(y.tolist())], seed=args.seed)
    tr=[i for i,_ in tr]; va=[i for i,_ in va]; te=[i for i,_ in te]
    n_classes=len(classes)
    model=PaperTransformer(foundation,n_classes,args.d_model,args.nhead,args.num_layers,
                           args.dim_feedforward,args.dropout,args.mode,args.r,
                           args.lora_targets,args.layer).to(device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    print(f'[model] classes={n_classes}, trainable={sum(p.numel() for p in trainable):,}')
    weight=None
    if args.class_weighted:
        cnt=np.bincount(y[tr],minlength=n_classes).astype(np.float32)
        weight=torch.tensor(len(tr)/(n_classes*np.maximum(cnt,1)),dtype=torch.float32,device=device)
    crit=nn.CrossEntropyLoss(weight=weight)
    opt=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:(1-e/args.epochs)**0.9 if e<args.epochs else 0)
    best=None; best_score=-1
    for ep in range(args.epochs):
        model.train(); total=0.0; nb=0
        if args.mode != 'full':
            model.foundation.eval()
        for x,m,lab in batch_iter(data,y,tr,args.batch_size,True,args.seed+ep):
            x=x.to(device); m=m.to(device); lab=lab.to(device)
            opt.zero_grad(); logits=model(x,m); loss=crit(logits,lab)
            loss.backward(); opt.step(); total+=float(loss.item()); nb+=1
        sched.step()
        yv,pv=evaluate(model,data,y,va,args.batch_size,device)
        met=compute_metrics(yv.tolist(),pv.tolist(),labels=list(range(n_classes)))
        score=met['WF1']
        print(f'[epoch {ep+1:02d}/{args.epochs}] loss={total/max(nb,1):.4f} val_WF1={score:.4f}',flush=True)
        if score>best_score:
            best_score=score
            best={k:v.detach().cpu() for k,v in model.state_dict().items()
                  if ('foundation' not in k) or ('lora_' in k)}
    if best is not None:
        model.load_state_dict(best,strict=False)
    yt,pt=evaluate(model,data,y,te,args.batch_size,device)
    met=compute_metrics(yt.tolist(),pt.tolist(),labels=list(range(n_classes)))
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    result={'args':vars(args),'classes':classes,'split':{'train':len(tr),'val':len(va),'test':len(te)},'metrics':met,'predictions':{'y_true':yt.tolist(),'y_pred':pt.tolist()}}
    suffix=args.mode if args.mode != 'lora' else f"lora_r{args.r}"
    if args.layer >= 0:
        suffix += f"_layer{args.layer}"
    path=out/f"paper_transformer_{suffix}_{args.model.replace('/','_')}.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[result] WA={met["WA"]:.4f} UA={met["UA"]:.4f} WF1={met["WF1"]:.4f}')
    print(f'[out] {path}')

if __name__=='__main__':
    main()


