---
title: 🔥 PyTorch 入門指南
date: 2025-04-22
categories: 
  - 機器學習
  - 深度學習
tags:
  - PyTorch
  - 神經網路
  - AI開發
---

> 🧠 *"PyTorch 就像樂高積木，自由度高、控制力強，是深度學習工程師的最愛！"*

---

## 🚀 PyTorch 是什麼？

<!-- more -->

PyTorch 是一個 **開源深度學習框架**，由 Facebook AI Research 所開發，它提供：

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

➡️ 類似 NumPy 的 **ndarray**，但可以 .cuda() 上 GPU！

### 🧱 2. 建構神經網路

使用 **torch.nn.Module** 實作模型結構：

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(784, 10)  # 全連接層

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

| 演算法 | PyTorch 支援方式 | 應用場景 |
| -----| ---- | ---- |
| 🎯 CNN | **nn.Conv2d**, **MaxPool2d** | 影像辨識 |
| 🧠 RNN / LSTM | **nn.RNN**, **nn.LSTM** | 時序預測 |
| 🧠 Transformer | **torch.nn.TransformerEncoder** | NLP 任務 |
| 🎲 GAN | 自定義兩個模型（Generator / Discriminator） | 圖像生成 |

### 🛠 PyTorch 開發流程（六大步）

#### 1. 📦 資料處理（**torch.utils.data.Dataset**, DataLoader）
#### 2. 🧱 定義模型架構（繼承 **nn.Module**）
#### 3. 🧮 定義損失函數（**nn.CrossEntropyLoss()** 等）
#### 4. 🛞 選擇優化器（如 SGD, Adam）
#### 5. 🔁 訓練迴圈（前向傳播 → 計算 Loss → 反向傳播 → 更新參數）
#### 6. 🧪 測試模型（使用驗證資料集）

### 📌 心得筆記

- 💡 PyTorch 就像是程式語言等級的工具，可以讓你非常靈活地設計模型結構。
- 💪 比 TensorFlow 直觀、易除錯，非常適合研究與原型開發！