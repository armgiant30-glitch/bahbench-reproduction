# 数据路径与 split 说明

## 当前 AutoDL 数据

| 数据 | 路径 | 当前样本数 | 标签 |
|---|---|---:|---|
| ICBHI | `/root/autodl-tmp/extra/icbhi` | 920 | 8 类 |
| TORGO | `/root/autodl-tmp/extra/torgo/wav` | 8,990 | 2 类 |
| SEP-28k | `/root/autodl-tmp/bahbench-data/SEP-28k` | 8,800 | 2 类 |

## 当前 split

| 数据 | 当前 split |
|---|---|
| ICBHI | 642 / 134 / 144 |
| TORGO | 6,292 / 1,347 / 1,351 |
| SEP-28k | 6,159 / 1,319 / 1,322 |

## 论文 split

| 数据 | 论文 train / val / test |
|---|---|
| ICBHI | 539 / 0 / 381 |
| TORGO | 6,572 / 0 / 1,644 |
| SEP-28k | 24,922 / 2,000 / 1,000 |

当前代码默认使用 `common.py` 中的分层随机 split；要严格复现论文，需要把 `paper_transformer.py` 的数据准备部分替换为论文官方 split。
