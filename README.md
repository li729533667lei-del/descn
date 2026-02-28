# DESCN (Deep Entire Space Cross Networks) Demo

基于 **PyTorch 2.5.1** 的可运行项目示例，包含：

- 单值连续特征（`dense_cont`）
- 序列连续特征（`seq_cont`）
- 单值类别特征（`dense_cat`）
- 序列类别特征（`seq_cat`）

模型为 DESCN（Cross Network + Deep Network），并支持 **4 个输出头**：

- `control`
- `treatment_1`
- `treatment_2`
- `treatment_3`

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行 Demo

```bash
python train_demo.py
```

运行后会输出每个 epoch 的 loss，并打印四个 head 的预测样例。

## 目录说明

- `descn/model.py`: DESCN 模型定义与特征编码逻辑
- `descn/data.py`: 合成数据集（可直接训练）
- `train_demo.py`: 训练与推理演示脚本
