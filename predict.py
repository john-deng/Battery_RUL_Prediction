import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


class InferenceEngine:
    def __init__(self, model_path, feat_scaler, label_scaler):
        """
        初始化推理引擎
        :param model_path: 模型文件路径
        :param feat_scaler: 特征标准化器
        :param label_scaler: 标签标准化器
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feat_scaler = feat_scaler
        self.label_scaler = label_scaler

        # 加载模型架构
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path):
        """动态构建模型架构"""

        class DynamicFFN(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 2048),
                    torch.nn.BatchNorm1d(2048),
                    torch.nn.GELU(),
                    torch.nn.Linear(2048, 1024),
                    torch.nn.Dropout(0.2),
                    torch.nn.GELU(),
                    torch.nn.Linear(1024, 1)
                )

            def forward(self, x):
                return self.net(x)

        # 动态获取输入维度
        input_dim = self.feat_scaler.n_features_in_
        model = DynamicFFN(input_dim).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        return model

    def preprocess(self, raw_data):
        """
        数据预处理管道
        :param raw_data: 原始输入数据 (n_samples, n_features)
        :return: 标准化后的张量数据
        """
        if raw_data.shape[1] != self.feat_scaler.n_features_in_:
            raise ValueError(
                f"输入特征维度不匹配，预期 {self.feat_scaler.n_features_in_} 维，实际收到 {raw_data.shape[1]} 维")

        scaled_data = self.feat_scaler.transform(raw_data)
        return torch.tensor(scaled_data, dtype=torch.float32, device=self.device)

    def predict(self, raw_data, batch_size=4096):
        """
        批量预测函数
        :param raw_data: 原始输入数据 (n_samples, n_features)
        :param batch_size: 推理批大小
        :return: 反标准化后的预测结果 (n_samples, 1)
        """
        tensor_data = self.preprocess(raw_data)

        # 分批次推理
        predictions = []
        with torch.no_grad():
            for i in range(0, len(tensor_data), batch_size):
                batch = tensor_data[i:i + batch_size]
                pred = self.model(batch)
                predictions.append(pred.cpu())

        # 后处理
        combined = torch.cat(predictions).numpy()
        return self.label_scaler.inverse_transform(combined)


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 模拟加载训练时的scaler（实际应从训练流程保存）
    feat_scaler = StandardScaler().fit(np.random.randn(100, 8))  # 示例数据
    label_scaler = StandardScaler().fit(np.random.randn(100, 1))

    # 初始化推理引擎
    engine = InferenceEngine(
        model_path="best_model.pth",
        feat_scaler=feat_scaler,
        label_scaler=label_scaler
    )

    # 生成测试数据
    test_data = np.random.randn(5, 8)  # 5个样本，8维特征

    # 执行预测
    results = engine.predict(test_data)
    print("预测结果矩阵:\n", results)
