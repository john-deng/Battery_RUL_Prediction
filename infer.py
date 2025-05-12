import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# ------ 网络定义，与训练时保持一致 ------
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

# ------ 推理流程 ------
def parse_args():
    parser = argparse.ArgumentParser(description="Battery RUL Inference")
    parser.add_argument("--input-csv", type=str, required=True, help="待预测数据 CSV 文件")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="训练好的模型文件")
    parser.add_argument("--feat-scaler", type=str, default="feat_scaler.pkl", help="特征标准化器文件")
    parser.add_argument("--label-scaler", type=str, default="label_scaler.pkl", help="标签反标准化器文件")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="模型隐藏层维度，应与训练时一致")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="输出带预测结果的 CSV 文件")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    df = pd.read_csv(args.input_csv)
    features = df.iloc[:, 2:-1].values  # 根据训练脚本的切片方式

    # 加载标准化器
    feat_scaler: StandardScaler = joblib.load(args.feat_scaler)
    label_scaler: StandardScaler = joblib.load(args.label_scaler)
    feats_scaled = feat_scaler.transform(features)

    # 构建模型并加载权重
    input_dim = feats_scaled.shape[1]
    model = FeedForwardNet(input_dim, args.hidden_dim).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 推理
    with torch.no_grad():
        x = torch.from_numpy(feats_scaled).float().to(device)
        preds_scaled = model(x).cpu().numpy()

    # 反标准化
    preds = label_scaler.inverse_transform(preds_scaled)

    # 保存结果
    df['pred_RUL'] = preds.flatten()
    df.to_csv(args.output_csv, index=False)
    print(f"预测完成，结果已保存至 {args.output_csv}")


if __name__ == "__main__":
    main()
