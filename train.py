import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ------ 系统配置 ------
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------ 配置参数 ------
class Config:
    hidden_dim = 2048
    output_dim = 1
    batch_size = 4096
    lr = 1e-4
    epochs = 200
    patience = 15
    grad_clip = 1.0
    mix_precision = True


# ------ 数据流水线 ------
class BatteryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(path):
    df = pd.read_csv(path)

    # 特征提取与验证
    features = df.iloc[:, 2:-1].values  # 确保维度一致性
    labels = df.iloc[:, -1].values.reshape(-1, 1)

    print(f"特征维度: {features.shape[1]} | 样本数: {features.shape[0]}")

    # 标准化处理
    feat_scaler = StandardScaler().fit(features)
    label_scaler = StandardScaler().fit(labels)

    return (
        feat_scaler.transform(features),
        label_scaler.transform(labels),
        feat_scaler, label_scaler
    )


# ------ 网络架构 ------
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.hidden_dim),
            nn.BatchNorm1d(Config.hidden_dim),
            nn.GELU(),

            nn.Linear(Config.hidden_dim, Config.hidden_dim // 2),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(Config.hidden_dim // 2, Config.output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):
        return self.net(x)


# ------ 训练引擎 ------
class Trainer:
    def __init__(self, model, train_loader, valid_loader):
        self.model = model.to(device)
        self.scaler = torch.amp.GradScaler(enabled=Config.mix_precision)
        self.optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.loss_fn = nn.HuberLoss()

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.best_loss = float('inf')
        self.stale_epochs = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        for inputs, targets in self.valid_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.valid_loader)
        self.scheduler.step(avg_loss)
        return avg_loss

    def early_stop(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.stale_epochs = 0
            torch.save(self.model.state_dict(), 'best_model.pth')
        else:
            self.stale_epochs += 1

        return self.stale_epochs >= Config.patience


# ------ 主程序 ------
if __name__ == "__main__":
    # 数据加载
    features, labels, feat_scaler, label_scaler = load_data(os.getcwd() + '/Datasets/HNEI_Processed/Final Database.csv')
    Config.input_dim = features.shape[1]  # 动态设置输入维度

    # 数据集划分
    dataset = BatteryDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    train_set, valid_set = random_split(dataset, [train_size, len(dataset) - train_size])

    # 数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=Config.batch_size * 2,
        num_workers=4,
        pin_memory=True
    )

    # 模型初始化
    model = FeedForwardNet(Config.input_dim)
    print(f"模型架构:\n{model}")

    # 训练流程
    trainer = Trainer(model, train_loader, valid_loader)
    for epoch in range(Config.epochs):
        train_loss = trainer.train_epoch()
        valid_loss = trainer.validate()

        print(f"Epoch {epoch + 1:03d} | Train: {train_loss:.4f} | Valid: {valid_loss:.4f}")

        if trainer.early_stop(valid_loss):
            print(f"早停于第 {epoch + 1} 轮")
            break

    print("训练完成，最佳模型已保存")

