# Neural-Collaborative-Filtering-NCF-Recommendation-System-with-PyTorch

Neural Collaborative Filtering (NCF) is a **deep learning-based recommendation system** approach that combines the power of **neural networks** with **collaborative filtering (CF)** to model user-item interactions.

Let’s break down the **core concepts behind NCF**, especially with a focus on how you’d implement it in **PyTorch**.

---

## 🔑 Core Concepts of Neural Collaborative Filtering (NCF)

### 1. **Collaborative Filtering Background**
- Traditional CF approaches (like Matrix Factorization) model the interaction between user and item using **dot products** of latent embeddings.
- These are linear models, which limit their ability to capture complex, non-linear patterns in the data.

### 2. **Neural CF Idea**
- NCF replaces the dot product with a **neural network** that learns the interaction function from data.
- The user and item embeddings are **concatenated** and passed through **multiple non-linear layers** to capture complex interactions.

---

## 🧠 Architecture Overview

### ✨ Embedding Layer
- Users and items are represented by **learnable embeddings**:
```python
user_embedding = nn.Embedding(num_users, embed_dim)
item_embedding = nn.Embedding(num_items, embed_dim)
```

### 🧱 Concatenation
- The user and item embeddings are **concatenated** instead of using a dot product:
```python
interaction = torch.cat([user_embed, item_embed], dim=-1)
```

### 🔁 Multi-Layer Perceptron (MLP)
- This is the core part of the model where the concatenated embeddings are passed through an MLP:
```python
mlp = nn.Sequential(
    nn.Linear(2 * embed_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()  # for implicit feedback
)
```

### 🎯 Output Layer
- A single value is predicted — the **probability of interaction** (e.g., user likes item).

---

## 🛠 PyTorch Implementation (Simplified)

Here’s a basic NCF model using PyTorch:

```python
import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32):
        super(NCF, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # for implicit feedback
        )

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embed(user_ids)
        item_vec = self.item_embed(item_ids)
        x = torch.cat([user_vec, item_vec], dim=-1)
        out = self.mlp(x)
        return out.squeeze()
```
## For detailed implementation see the colab notebook 
---

## ⚙️ Training Objective

Usually, **binary cross-entropy loss** is used (especially for implicit feedback like clicks):

```python
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 💡 Variants of NCF

1. **GMF (Generalized Matrix Factorization)**: Learns element-wise product interaction using embeddings.
2. **MLP (Multi-layer Perceptron)**: Uses the architecture above.
3. **NeuMF (Neural Matrix Factorization)**: Combines GMF + MLP.

> In NeuMF, GMF and MLP are trained in parallel and their outputs are concatenated for final prediction.

---

## ✅ Summary

| Component       | Role |
|----------------|------|
| User & Item Embeddings | Learn latent features |
| MLP | Captures complex interactions |
| Sigmoid Output | Predicts interaction probability |
| BCE Loss | Optimizes implicit feedback prediction |
| PyTorch Modules | `nn.Embedding`, `nn.Linear`, `nn.Sequential`, `torch.cat` |

---
