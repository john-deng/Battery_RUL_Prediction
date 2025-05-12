import os
import argparse
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm.auto import tqdm

# ------ 可复现性 ------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------ 参数与设备 ------
def parse_args():
    p = argparse.ArgumentParser(description="Battery RUL Training")
    p.add_argument("--data-csv", type=str, required=True, help="训练数据 CSV 文件路径")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="best_model.pth")
    return p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------ 数据集 ------
class BatteryDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(features).float()
        self.y = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_and_scale(path: str):
    df = pd.read_csv(path)
    features = df.iloc[:, 2:-1].values
    labels = df.iloc[:, -1].values.reshape(-1, 1)
    print(f"数据样本: {features.shape[0]}，特征维度: {features.shape[1]}")
    feat_scaler = StandardScaler().fit(features)
    label_scaler = StandardScaler().fit(labels)
    return (
        feat_scaler.transform(features),
        label_scaler.transform(labels),
        feat_scaler, label_scaler
    )

# ------ 模型 ------
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

# ------ 训练与验证 ------
def train_one_epoch(model, loader, optimizer, loss_fn, scaler):
    model.train()
    running = 0.0
    loop = tqdm(loader, desc="Train", leave=False)
    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            preds = model(x)
            loss = loss_fn(preds, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        loop.set_postfix(loss=running / (loop.n + 1))
    return running / len(loader)

@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    running = 0.0
    loop = tqdm(loader, desc="Valid", leave=False)
    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x)
        loss = loss_fn(preds, y)
        running += loss.item()
        loop.set_postfix(loss=running / (loop.n + 1))
    return running / len(loader)

def main():
    args = parse_args()
    set_seed(args.seed)

    # 数据准备
    feats, labs, feat_sc, lab_sc = load_and_scale(args.data_csv)
    # 保存 scaler 对象
    joblib.dump(feat_sc, 'feat_scaler.pkl')
    joblib.dump(lab_sc, 'label_scaler.pkl')
    print("✔ scaler 已保存到 feat_scaler.pkl 与 label_scaler.pkl")

    dataset = BatteryDataset(feats, labs)
    n_train = int(len(dataset) * 0.8)
    train_ds, valid_ds = random_split(dataset, [n_train, len(dataset) - n_train], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    # 模型、优化器、调度器
    model = FeedForwardNet(feats.shape[1], args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 推荐使用 OneCycleLR，加速收敛
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.lr * 10,
                           steps_per_epoch=len(train_loader),
                           epochs=args.epochs)
    # 或者启用 ReduceLROnPlateau：
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
    loss_fn = nn.HuberLoss()
    scaler = GradScaler()

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler)
        valid_loss = validate(model, valid_loader, loss_fn)

        # 更新学习率
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()
        else:
            scheduler.step(valid_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Loss: {valid_loss:.4f} | LR: {lr:.1e}")

        # 早停与模型保存
        if valid_loss < best_loss:
            best_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"→ 保存最佳模型: {args.save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f">> 早停触发：已连续 {args.patience} 轮验证集 loss 未下降，训练结束。")
                break

    print("训练完成。")

if __name__ == "__main__":
    main()
