import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# 启用cuDNN自动调优和确定性配置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # 允许非确定性优化以获得更快速度

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 调整后的超参数
batch_size = 256  # 大幅增大批量以利用大显存
input_size = 7
hidden_size = 512  # 增大隐藏层维度以更好利用并行计算
num_classes = 1
learning_rate = 0.001  # 增大学习率配合大batch
epochs = 200  # 减少总epoch数（因为每个epoch处理更多数据）


class BatteryDataSet(Dataset):
    # 保持原有数据预处理逻辑不变
    def __init__(self):
        dataset_raw = pd.read_csv(os.getcwd() + '/Datasets/HNEI_Processed/Final Database.csv')
        dataset_raw.drop('Unnamed: 0', axis=1, inplace=True)

        data = dataset_raw.values[:, :-1]
        trans = MinMaxScaler()
        data = trans.fit_transform(data)
        dataset = pd.DataFrame(data)
        dataset_scaled = dataset.join(dataset_raw['RUL'])
        scaled_df_np = dataset_scaled.to_numpy(dtype=np.float32)

        self.x = torch.from_numpy(scaled_df_np[:, 2:-1])
        self.y = torch.from_numpy(scaled_df_np[:, [-1]])
        self.n_samples = scaled_df_np.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def create_loaders(dataset, batch_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))

    # 使用更高效的数据分割方式
    train_indices, test_indices = indices[:split], indices[split:]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=8,  # 增加数据加载线程
        pin_memory=True,  # 启用锁页内存
        persistent_workers=True  # 保持workers存活
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size * 2,  # 增大测试batch size
        sampler=SubsetRandomSampler(test_indices),
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader, test_loader


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),  # 使用inplace操作节省内存
            nn.Linear(hidden_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.layers(x)


def train_loop(train_loader, model, loss_fn, optimizer, scaler):
    model.train()
    for features, rul in train_loader:
        # 异步数据迁移
        features = features.to(device, non_blocking=True)
        rul = rul.to(device, non_blocking=True)

        # 自动混合精度
        with torch.cuda.amp.autocast():
            outputs = model(features)
            loss = loss_fn(outputs, rul)

        # 梯度缩放和反向传播优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零


def test_loop(test_loader, model, loss_fn):
    model.eval()
    test_loss = 0
    diff_list = []

    with torch.inference_mode():  # 更高效的推理模式
        for features, rul in test_loader:
            features = features.to(device, non_blocking=True)
            rul = rul.to(device, non_blocking=True)

            # 保持全精度进行计算
            with torch.cuda.amp.autocast(enabled=False):
                preds = model(features.float())  # 显式转换为float32
                test_loss += loss_fn(preds, rul).item()

            # 原位计算指标
            diff = torch.abs(preds - rul) / rul
            diff_list.append(diff.mean().cpu())

    avg_diff = torch.mean(torch.tensor(diff_list)).item() * 100
    avg_loss = test_loss / len(test_loader)
    print(f"Test Results:\nAvg Difference: {avg_diff:.2f}%\nAvg Loss: {avg_loss:.4f}\n")


if __name__ == "__main__":
    # 初始化
    dataset = BatteryDataSet()
    train_loader, test_loader = create_loaders(dataset, batch_size)

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # 使用LAMB优化器更适合大batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.HuberLoss()  # 改用更鲁棒的损失函数
    scaler = torch.cuda.amp.GradScaler()  # 梯度缩放器

    # 学习率预热
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )

    # 训练循环
    best_diff = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loop(train_loader, model, loss_fn, optimizer, scaler)
        scheduler.step()

        # 每5个epoch验证一次
        if (epoch + 1) % 5 == 0:
            test_loop(test_loader, model, loss_fn)

    # Save model
    torch.save(model.state_dict(), 'battery_rul_predictor.pth')
    print("Training complete! Model saved.")