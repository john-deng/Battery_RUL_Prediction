import os
import torch
import torch.backends.cudnn as cudnn
#from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# ─── Hyperparameters & Config ────────────────────────────────────────────────
batch_size     = 128      # 根据显存可调至更大
hidden_size    = 10
num_classes    = 1
learning_rate  = 1e-4
epochs         = 500
test_interval  = 5        # 每 5 个 epoch 做一次测试+绘图
num_workers    = 16       # DataLoader 并行 worker 数
pin_memory     = True     # 加速 GPU 拷贝
prefetch_factor= 2        # 每个 worker 预抓取 batch 数
EPS            = 1e-6     # 防止除零

# ─── Dataset ────────────────────────────────────────────────────────────────
class BatteryDataSet(Dataset):
    def __init__(self, scaled_np):
        # 前 N-1 列为特征，最后一列为 RUL
        self.x = torch.from_numpy(scaled_np[:, :-1])
        self.y = torch.from_numpy(scaled_np[:, [-1]])
        self.n = scaled_np.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_loaders(dataset, batch_size, shuffle=True):
    n = len(dataset)
    idxs = list(range(n))
    split = int(0.8 * n)
    train_idx, test_idx = idxs[:split], idxs[split:]
    if shuffle:
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)

    train_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    test_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(test_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    return train_loader, test_loader

# ─── Model ─────────────────────────────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ─── Training & Testing ────────────────────────────────────────────────────
def train_loop(loader, model, loss_fn, optimizer, device, scaler):
    model.train()
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            preds = model(X)
            loss  = loss_fn(preds, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return

def test_loop(loader, model, loss_fn, device, epoch, record_dict, label_scaler, do_plot=True):
    model.eval()
    total_loss = 0.0
    diff_list  = []
    all_preds  = []
    all_tgts   = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(X)
            total_loss += loss_fn(out, y).item()

            out_np_scaled = out.cpu().numpy().reshape(-1,1)
            y_np_scaled   = y.cpu().numpy().reshape(-1,1)
            out_np = label_scaler.inverse_transform(out_np_scaled).reshape(-1)
            y_np   = label_scaler.inverse_transform(y_np_scaled).reshape(-1)

            denom = np.maximum(y_np, 1.0)
            diff  = np.abs(y_np - out_np) / denom
            diff_list.append(diff.mean())

            all_preds.extend(out_np.tolist())
            all_tgts.extend(y_np.tolist())

    avg_loss = total_loss / len(loader)
    avg_diff = np.mean(diff_list) * 100
    record_dict[epoch+1] = avg_diff

    print(f"[Test] Epoch {epoch+1}: Avg Loss={avg_loss:.6f}, Avg Diff={avg_diff:.2f}%")
    best_epoch, best_diff = min(record_dict.items(), key=lambda x: x[1])
    print(f"→ Best so far: Epoch {best_epoch}, Diff={best_diff:.2f}%")

    if do_plot:
        plt.figure(dpi=200)
        plt.scatter(all_tgts, all_preds, s=5)
        plt.xlabel("Target RUL")
        plt.ylabel("Predicted RUL")
        plt.ylim(0, max(all_tgts) * 1.1)
        plt.title(f"Epoch {epoch+1}")
        out_dir = "plots"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/epoch_{epoch+1}.png")
        plt.close('all')

    return avg_loss

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # cuDNN & TF32
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32   = True

    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载 & 预处理
    csv_path = os.path.join(os.getcwd(), 'Datasets', 'HNEI_Processed', 'Final Database.csv')
    df = pd.read_csv(csv_path)
    df.drop(columns=['Unnamed: 0'], inplace=True)

    feat_scaler  = MinMaxScaler()
    label_scaler = MinMaxScaler()

    feats = feat_scaler.fit_transform(df.iloc[:, :-1].values)
    labels = df[['RUL']].values
    labels_scaled = label_scaler.fit_transform(labels)

    df_s = pd.DataFrame(feats, columns=df.columns[:-1])
    df_s['RUL'] = labels_scaled.flatten()
    np_s = df_s.to_numpy(dtype=np.float32)

    dataset = BatteryDataSet(np_s)
    train_loader, test_loader = make_loaders(dataset, batch_size, shuffle=True)

    # 模型/损失/优化器/调度器/混合精度
    input_size = dataset.x.shape[1]
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scaler    = GradScaler()

    records = {}

    # # 训练 + 测试 循环
    # for epoch in range(epochs):
    #     print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    #     train_loop(train_loader, model, loss_fn, optimizer, device, scaler)
    #
    #     do_plot = ((epoch+1) % test_interval == 0) or (epoch == epochs-1)
    #     avg_loss = test_loop(
    #         test_loader, model, loss_fn, device,
    #         epoch, records, label_scaler, do_plot
    #     )
    #
    #     scheduler.step(avg_loss)

    # 在训练循环前，初始化早停参数
    best_loss = float('inf')
    no_improve = 0
    patience = 15  # 连续多少个 epoch 无提升就停止

    # ─── 训练 + 测试 + 早停 循环 ────────────────────────────────────────────────
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        train_loop(train_loader, model, loss_fn, optimizer, device, scaler)

        do_plot = ((epoch + 1) % test_interval == 0) or (epoch == epochs - 1)
        avg_loss = test_loop(
            test_loader, model, loss_fn, device,
            epoch, records, label_scaler, do_plot
        )

        # 学习率调度
        scheduler.step(avg_loss)

        # 早停判断
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✔  Epoch {epoch + 1}: validation loss improved to {best_loss:.6f}, saved best_model.pth")
        else:
            no_improve += 1
            print(f"✖  No improvement for {no_improve}/{patience} epochs.")
            if no_improve >= patience:
                print(f"\n🔔 Early stopping: 连续 {patience} 轮验证集 loss 无提升，终止训练。")
                break

    # 保存模型
    save_path = os.path.join(os.getcwd(), 'Datasets', 'FF_Net_optimized.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\n训练完成！Model saved to: {save_path}")
    exit(0)
