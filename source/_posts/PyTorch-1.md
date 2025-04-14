title: 🔥 PyTorch 入門指南
date: 2025-04-14
categories: 
  - 機器學習
  - 深度學習

tags:
  - PyTorch
  - 神經網路
  - AI開發
---

> 🧠 *"PyTorch，自由度高、控制力強，是深度學習工程師的最愛！"*

---

## 🚀 PyTorch 前言？

PyTorch 是一個 **開源深度學習框架**，由 Meta AI 所開發，它提供：

- 🧮 強大的 **Tensor 運算** 功能（就像 Numpy，但有 GPU 支援）
- 🧱 靈活的 **神經網路建構系統（torch.nn）**
- 🔁 支援 **動態計算圖**（Dynamic Computation Graph），可即時調整模型架構

---

## 🧠 PyTorch 的核心概念

### 🧊 1. Tensor（張量）

PyTorch 的資料基本單位，是個可在 GPU 上運算的多維陣列：

```python
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
```
➡️ 類似 NumPy 的 ndarray，但 .cuda() 是使用 GPU 的，運算比 CPU 快超多。

### 🧱2. 建構神經網路

使用 **torch.nn.Module** 實作模型結構：

```python 
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(784, 10)  
        # 全連接層

    def forward(self, x):
        return self.fc(x)
```
🔧 每一層都是可以組裝的「模組積木」。

### 🔄 3. 前向傳播 & 反向傳播

PyTorch 使用 動態計算圖（Define-by-Run），每次執行時才建圖：

```python
y = model(x)
loss = loss_fn(y, label)
loss.backward()  # 自動反向傳播
optimizer.step() # 更新參數
```

### 🧮 常見演算法應用
|  演算法  | 	PyTorch 支援方式  |  應用場景  |
|  ----  | ------------  | ----- |
| 🎯 CNN  | **nn.Conv2d**, **MaxPool2d** | 影像辨識 |
| 🧠 RNN / LSTM  | **nn.RNN**, **nn.LSTM** | 時序預測 |
| 🧠 Transformer  | **torch.nn.TransformerEncoder** | NLP 任務 |
| 🎲 GAN  | 自定義兩個模型（Generator / Discriminator） | 圖像生成 |

### 📌 心得筆記

- 💡 PyTorch 就像是程式語言等級的工具，可以非常靈活地設計模型結構。

- 💪 比 TensorFlow 直觀、易除錯，非常適合研究與原型開發！