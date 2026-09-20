"""
common.py —— 共用工具：HF 镜像、随机种子、音频读取、分层特征提取、线性探针

⚠️ 重要：本模块在导入时就会设置 HF_ENDPOINT 镜像。
   因为国内直连 huggingface.co 不通，而 huggingface_hub 在 import 时就会读取该变量，
   所以必须在 import transformers / huggingface_hub **之前** 导入本模块。

用法：
    from common import setup_hf_mirror, set_seed, load_manifest, ...
"""

# ---------------------------------------------------------------------------
# 1) HF 镜像 —— 必须在任何 huggingface 相关 import 之前执行
# ---------------------------------------------------------------------------
import os


def setup_hf_mirror(endpoint="https://hf-mirror.com"):
    """设置 HuggingFace 镜像端点（幂等，已设置则不覆盖）。"""
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"[env] HF_ENDPOINT = {endpoint}")
    else:
        print(f"[env] HF_ENDPOINT 已设置 = {os.environ['HF_ENDPOINT']}")
    return os.environ["HF_ENDPOINT"]


setup_hf_mirror()   # ← 副作用式初始化，保证后续 import 拿到镜像地址

# ---------------------------------------------------------------------------
# 2) 常规依赖
# ---------------------------------------------------------------------------
import csv
import math
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"}


# ---------------------------------------------------------------------------
# 3) 随机种子
# ---------------------------------------------------------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda":
        raise RuntimeError("指定了 cuda，但当前环境没有可用的 GPU")
    print("[env] 未检测到 GPU，回退到 CPU（会很慢，建议在 AutoDL 上用 GPU 跑）")
    return torch.device("cpu")


def describe_env():
    print(f"[env] torch {torch.__version__}　cuda 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"[env]   GPU{i}: {p.name}　显存 {p.total_memory/1024**3:.1f} GB")


# ---------------------------------------------------------------------------
# 4) 音频读取与重采样
# ---------------------------------------------------------------------------
def _read_wav_stdlib(path):
    """标准库兜底（仅支持 16-bit PCM wav）。返回 (np.float32[1,T], sr)。"""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: 标准库兜底只支持 16-bit wav（当前 {sw*8} bit）")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data[None, :], sr


def read_audio(path, target_sr=16000):
    """读音频 → (torch.FloatTensor[1, T] @ target_sr)。

    依次尝试 soundfile → torchaudio → 标准库 wave。
    """
    path = str(path)
    data, sr = None, None

    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)[None, :]        # 多声道求平均
    except Exception:
        pass

    if data is None:
        try:
            import torchaudio
            wav, sr = torchaudio.load(path)
            data = wav.mean(dim=0, keepdim=True).numpy()
        except Exception:
            pass

    if data is None:
        data, sr = _read_wav_stdlib(path)

    wav = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
    return wav


def resample(wav, sr, target_sr):
    """重采样。优先用 torchaudio，缺失则用线性插值（够用于演示）。"""
    if sr == target_sr:
        return wav
    try:
        import torchaudio
        return torchaudio.functional.resample(wav, sr, target_sr)
    except Exception:
        T = wav.shape[-1]
        T_out = int(round(T * target_sr / sr))
        idx = torch.linspace(0, T - 1, T_out)
        lo = idx.floor().long().clamp(0, T - 1)
        hi = (lo + 1).clamp(0, T - 1)
        frac = (idx - lo).unsqueeze(0)
        return wav[:, lo] * (1 - frac) + wav[:, hi] * frac


# ---------------------------------------------------------------------------
# 5) 数据集清单（manifest）
# ---------------------------------------------------------------------------
def build_manifest_from_dir(root, max_per_class=None, seed=0):
    """扫描目录树，子目录名当标签。返回 [(path, label), ...]

    目录结构要求：
        root/
          class_a/ *.wav
          class_b/ *.wav

    ⚠️ 特例（很重要）：ESC-50 官方解压出来是**平铺**的 ——
        ESC-50-master/audio/1-100032-A-0.wav（2000 个文件挤在一个目录里），
        标签在 ESC-50-master/meta/esc50.csv。
        如果不特判，`root.iterdir()` 只会看到 audio/ 和 meta/ 两个"类别"，结果是错的；
        而直接传 audio/ 目录则一个子目录都没有，会抛 FileNotFoundError。
        所以这里检测到 esc50.csv 就改用 csv 里的 category 当标签。
        传 ESC-50-master/ 或 ESC-50-master/audio/ 都行。
    """
    root = Path(root)

    # ---- 特例：ESC-50（平铺 wav + meta/esc50.csv）----
    esc_csv = None
    for cand in (root / "meta" / "esc50.csv",
                 root.parent / "meta" / "esc50.csv",
                 root / "esc50.csv"):
        if cand.exists():
            esc_csv = cand
            break
    if esc_csv is not None:
        with open(esc_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        audio_dir = root / "audio" if (root / "audio").is_dir() else root
        by_cat = {}
        for r in rows:
            p = audio_dir / r["filename"]
            if p.exists():
                by_cat.setdefault(r["category"], []).append(str(p))
        items = []
        for cat in sorted(by_cat):
            files = sorted(by_cat[cat])
            if max_per_class:
                rng = random.Random(seed)
                rng.shuffle(files)
                files = files[:max_per_class]
            items += [(f, cat) for f in files]
        if items:
            return items

    # ---- 常规：子目录名当标签 ----
    items = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        # 跳过 _background_noise_ 这类辅助目录（Speech Commands 里是长音频，
        # 不是关键词样本，当成类别会污染任务）
        if d.name.startswith(("_", ".")):
            continue
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        if max_per_class:
            rng = random.Random(seed)
            rng.shuffle(files)
            files = files[:max_per_class]
        items += [(str(f), d.name) for f in files]
    if not items:
        raise FileNotFoundError(
            f"{root} 下没找到音频。\n"
            f"  支持两种摆法：\n"
            f"    ① 子目录名当标签：<dir>/<类别名>/*.wav\n"
            f"    ② ESC-50 官方结构：<dir>/audio/*.wav + <dir>/meta/esc50.csv\n"
            f"  （扩展名认这些：{AUDIO_EXTS}）")
    return items


def save_manifest(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(items)
    print(f"[data] 已写清单 {path}（{len(items)} 条）")


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [(row["path"], row["label"]) for row in r]


def split_by_label(items, ratios=(0.7, 0.15, 0.15), seed=0, stratify=True):
    """划分 train/val/test。stratify=True 时按类别等比例划分。"""
    rng = random.Random(seed)
    if not stratify:
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        return items[:n_tr], items[n_tr:n_tr + n_va], items[n_tr + n_va:]

    by_label = defaultdict(list)
    for it in items:
        by_label[it[1]].append(it)
    tr, va, te = [], [], []
    for lab, group in sorted(by_label.items()):
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_tr = max(1, int(n * ratios[0]))
        n_va = int(n * ratios[1])
        # 小类别保底：至少留 1 条给 test
        if n >= 3 and n - n_tr - n_va < 1:
            n_tr = max(1, n - n_va - 1)
        tr += group[:n_tr]
        va += group[n_tr:n_tr + n_va]
        te += group[n_tr + n_va:]
    for lst in (tr, va, te):
        rng.shuffle(lst)
    return tr, va, te


# ---------------------------------------------------------------------------
# 6) 合成音频（用于 smoke test：不下载任何数据就能验证流程通不通）
# ---------------------------------------------------------------------------
def make_synthetic_items(n_classes=4, n_per_class=40, seed=0):
    """生成「可学习」的合成音频：每类一个基频 + 不同的谐波结构 + 噪声。

    返回 [(waveform_tensor, label), ...]，直接进内存，不落盘。
    这不是为了刷分，只是为了让整条流水线（提取特征→训练→评估）能跑通。
    """
    rng = random.Random(seed)
    sr = 16000
    dur = 1.0
    T = int(sr * dur)
    t = torch.arange(T, dtype=torch.float32) / sr

    items = []
    for c in range(n_classes):
        f0 = 120.0 + c * 90.0                     # 每类一个基频
        harmonics = [1.0, 0.6 / (c + 1), 0.3 * (c % 2 + 1), 0.15]
        for _ in range(n_per_class):
            wav = torch.zeros(T)
            for k, amp in enumerate(harmonics, start=1):
                wav += amp * torch.sin(2 * math.pi * f0 * k * t + rng.random() * 6.28)
            wav += 0.15 * torch.randn(T)          # 噪声
            # 随机增益与轻微时长抖动，避免模型只靠能量作弊
            wav *= 0.5 + rng.random()
            wav = wav / (wav.abs().max() + 1e-6) * 0.8
            items.append((wav[None, :], f"class{c}"))
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# 7) 分层特征提取
# ---------------------------------------------------------------------------
def default_layers(n_layers, stride=3):
    """模仿论文「每隔三层测一次」：返回 [1, 1+stride, ...] 并确保含最后一层。"""
    idx = list(range(1, n_layers + 1, stride))
    if idx[-1] != n_layers:
        idx.append(n_layers)
    return idx


@torch.no_grad()
def extract_layer_features(model, wavs, layers, device, pool="mean", batch_size=8,
                           max_seconds=6.0):
    """对一批音频提取指定层的隐状态，时间维池化后返回。

    参数
        model   : 已 .eval() 的 transformers 模型，需支持 output_hidden_states=True
        wavs    : list[torch.FloatTensor[1, T]]
        layers  : list[int]，隐状态索引（0 = 卷积特征提取器输出，i = 第 i 层之后）
        pool    : "mean" | "max"

    返回
        dict[int, np.ndarray[n_samples, hidden_dim]]
    """
    import torch.nn.functional as F

    if max_seconds:
        limit = int(max_seconds * 16000)
        wavs = [w[:, :limit] for w in wavs]

    n_layers_total = model.config.num_hidden_layers
    for L in layers:
        if not (0 <= L <= n_layers_total):
            raise ValueError(f"层索引 {L} 越界（模型共 {n_layers_total} 层，合法 0..{n_layers_total}）")

    out = {L: [] for L in layers}

    for i in range(0, len(wavs), batch_size):
        batch = wavs[i:i + batch_size]
        pad = max(w.shape[-1] for w in batch)
        x = torch.zeros(len(batch), pad)
        mask = torch.zeros(len(batch), pad, dtype=torch.long)
        for j, w in enumerate(batch):
            x[j, :w.shape[-1]] = w[0]
            mask[j, :w.shape[-1]] = 1
        x, mask = x.to(device), mask.to(device)

        hs = model(x, attention_mask=mask, output_hidden_states=True).hidden_states

        for L in layers:
            h = hs[L]                                  # (B, T', D)
            # 卷积下采样后长度变了，按比例把原始长度换算成 T' 长度，做逐样本掩码池化
            lens = (mask.sum(dim=1).float() * h.shape[1] / pad).round().clamp(min=1).long()
            t = torch.arange(h.shape[1], device=h.device)[None, :]
            keep = (t < lens[:, None]).unsqueeze(-1).to(h.dtype)   # (B, T', 1)
            if pool == "mean":
                h = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
            else:                                       # max：先把 padding 位置压到 -inf
                h = h.masked_fill(keep == 0, float("-inf")).max(dim=1).values
            out[L].append(h.float().cpu().numpy())

        done = min(i + batch_size, len(wavs))
        if done % (batch_size * 5) == 0 or done == len(wavs):
            print(f"    [feat] {done}/{len(wavs)}")

    return {L: np.concatenate(v, axis=0) for L, v in out.items()}


def standardize(Xtr, Xva, Xte):
    """按训练集统计量做 z-score 标准化（线性探针的标准做法）。"""
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


# ---------------------------------------------------------------------------
# 8) 线性探针（逻辑回归）
# ---------------------------------------------------------------------------
class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(self.drop(x))


def train_linear_probe(Xtr, ytr, Xva, yva, n_classes, device,
                       epochs=60, lr=1e-2, wd=1e-4, batch_size=128,
                       class_weighted=False, verbose=False, seed=0):
    """训练线性探针，返回 (模型, 最佳验证准确率)。

    class_weighted=True 时给交叉熵加类别权重（对抗不平衡的常用手段）。
    """
    set_seed(seed)
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(np.asarray(ytr)).long()
    Xva_t = torch.from_numpy(Xva).float().to(device)
    yva_t = torch.from_numpy(np.asarray(yva)).long().to(device)

    model = LinearProbe(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    weight = None
    if class_weighted:
        cnt = Counter(np.asarray(ytr).tolist())
        n = len(ytr)
        w = torch.tensor([n / (n_classes * cnt.get(c, 1)) for c in range(n_classes)],
                         dtype=torch.float32)
        weight = w.to(device)

    lossf = nn.CrossEntropyLoss(weight=weight)

    n = Xtr_t.shape[0]
    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva_t).argmax(1) == yva_t).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % 20 == 0:
            print(f"      epoch {ep+1:3d}  val_acc={acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


@torch.no_grad()
def predict_probe(model, X, device, batch_size=512):
    model.eval()
    preds = []
    X = torch.from_numpy(X).float()
    for i in range(0, X.shape[0], batch_size):
        preds.append(model(X[i:i + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()
