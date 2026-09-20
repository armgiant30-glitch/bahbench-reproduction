"""
metrics.py —— BAHBench 的三个评价指标（纯 Python，零第三方依赖）

对应论文第 II-E 节公式 (6)(7)(8)：

    WA  = (1 / Σ_c N_c) · Σ_c N_c × Acc(c)          ... (6) 加权准确率
    UA  = (1 / C)       · Σ_c Acc(c)                ... (7) 非加权准确率
    WF1 = (1 / Σ_c N_c) · Σ_c N_c × F1(c)           ... (8) 加权 F1

其中 N_c 是类别 c 的真实样本数，Acc(c) 与 F1(c) 是类别 c 的准确率与 F1。

为什么要有 UA？
    类别不平衡时，把全部样本判成多数类就能拿到很高的 WA，但 UA 会立刻暴露问题。
    论文在 ICBHI 数据集上就抓到了这个现象（见 exp_c_imbalance.py）。
"""

from collections import OrderedDict

__all__ = ["confusion_matrix", "compute_metrics", "format_report"]


def confusion_matrix(y_true, y_pred, labels=None):
    """返回 (矩阵, 标签列表)。矩阵第 i 行第 j 列 = 真实为 i 却被判成 j 的样本数。"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    index = {c: i for i, c in enumerate(labels)}

    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t not in index or p not in index:
            raise ValueError(f"标签 {t!r} 或 {p!r} 不在 labels 里")
        cm[index[t]][index[p]] += 1
    return cm, labels


def compute_metrics(y_true, y_pred, labels=None):
    """计算 WA / UA / WF1。

    返回 dict：
        {
          "WA": float, "UA": float, "WF1": float,
          "n": 样本总数, "n_classes": 类别数,
          "per_class": [ {label, n, acc, precision, recall, f1}, ... ]
        }
    """
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    C = len(labels)
    N = sum(sum(row) for row in cm)
    if N == 0:
        raise ValueError("没有样本")

    per_class = []
    counts, accs, f1s = [], [], []

    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(C)) - tp   # 别人被判成我
        fn = sum(cm[i]) - tp                        # 我被判成别人
        n_c = tp + fn                               # 该类真实样本数

        acc = tp / n_c if n_c else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        counts.append(n_c)
        accs.append(acc)
        f1s.append(f1)
        per_class.append({
            "label": lab, "n": n_c, "acc": acc,
            "precision": precision, "recall": recall, "f1": f1,
        })

    WA = sum(c * a for c, a in zip(counts, accs)) / N   # 等价于整体准确率
    UA = sum(accs) / C
    WF1 = sum(c * f for c, f in zip(counts, f1s)) / N

    return {
        "WA": WA, "UA": UA, "WF1": WF1,
        "n": N, "n_classes": C,
        "per_class": per_class,
    }


def format_report(result, title=None, max_classes=10):
    """把 compute_metrics 的结果排成可读的文本报告。"""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"  样本数 {result['n']}　类别数 {result['n_classes']}")
    lines.append(f"  WA  = {result['WA']:.4f}   (加权准确率)")
    lines.append(f"  UA  = {result['UA']:.4f}   (非加权准确率)")
    lines.append(f"  WF1 = {result['WF1']:.4f}   (加权 F1)")
    gap = result["WA"] - result["UA"]
    lines.append(f"  WA - UA = {gap:+.4f}   {'← 差距大，说明模型在偏袒多数类' if gap > 0.15 else ''}")

    rows = sorted(result["per_class"], key=lambda d: -d["n"])
    lines.append(f"  {'类别':<16}{'样本数':>8}{'准确率':>10}{'精确率':>10}{'召回率':>10}{'F1':>10}")
    for d in rows[:max_classes]:
        lines.append(
            f"  {str(d['label']):<16}{d['n']:>8}{d['acc']:>10.4f}"
            f"{d['precision']:>10.4f}{d['recall']:>10.4f}{d['f1']:>10.4f}"
        )
    if len(rows) > max_classes:
        lines.append(f"  ... 还有 {len(rows) - max_classes} 个类别未显示")
    return "\n".join(lines)


if __name__ == "__main__":
    # 自检：手工可算的例子
    # 100 个样本：90 个 A、10 个 B；模型全猜 A
    y_true = ["A"] * 90 + ["B"] * 10
    y_pred = ["A"] * 100
    r = compute_metrics(y_true, y_pred)
    print(format_report(r, "自检：全猜多数类"))
    assert abs(r["WA"] - 0.90) < 1e-9, r["WA"]
    assert abs(r["UA"] - 0.50) < 1e-9, r["UA"]
    assert abs(r["WF1"] - 0.8526) < 1e-3, r["WF1"]

    # 完美分类
    r2 = compute_metrics(y_true, y_true)
    assert abs(r2["WA"] - 1.0) < 1e-9 and abs(r2["UA"] - 1.0) < 1e-9

    # 一个手工可验的多类例子
    # 3 类：A(2 个) B(1 个) C(1 个)
    #   真实 A A B C
    #   预测 A B B C  -> A 类 1/2，B 类 1/1，C 类 1/1
    r3 = compute_metrics(["A", "A", "B", "C"], ["A", "B", "B", "C"])
    assert abs(r3["WA"] - 0.75) < 1e-9, r3["WA"]
    assert abs(r3["UA"] - (0.5 + 1.0 + 1.0) / 3) < 1e-9, r3["UA"]
    print("\n自检全部通过 ✔")
