# 实验三：混淆矩阵可视化

## 目的

查看模型在测试集上把哪些类别混淆，特别是类别不平衡任务的少数类召回。

## 对比点

- 每个任务的最佳模型
- 行归一化混淆矩阵
- 对角线代表召回率
- 非对角元素代表误分类方向

## 输入

- `paper_transformer.py` 生成的 JSON
- JSON 必须包含 `predictions.y_true` 和 `predictions.y_pred`

## 输出

- `confusion_matrix_grid.png`
- 每个任务/模型单独的混淆矩阵 PNG
