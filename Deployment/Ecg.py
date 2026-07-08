from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import torch

# đảm bảo import được models.py ở thư mục gốc project
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models import ecgTransForm


class ECG:
    def __init__(self):
        self.transformer_model = None
        self.transformer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expected_signal_length = 3060  # khớp pipeline cũ: 12 leads x 255

        # mapping đúng theo data train
        self.label_map_4 = {
            0: "AHB - Abnormal Heartbeat",
            1: "MI - Myocardial Infarction",
            2: "NORMAL - Normal",
            3: "PM - History of MI",
        }

    def _build_transformer_configs(self):
        """
        Config phải khớp với lúc train train_ecgtransform.py
        """
        configs = SimpleNamespace(
            input_channels=1,
            mid_channels=32,
            stride=1,
            dropout=0.2,
            final_out_channels=128,
            trans_dim=385,
            num_heads=5,
            num_classes=4,
        )

        hparams = {
            "feature_dim": 128
        }
        return configs, hparams

    def load_transformer_checkpoint_4class(self, checkpoint_name="checkpoint_best_4class.pt"):
        """
        Load checkpoint 4 lớp đã train bằng ECGTransForm.
        Mặc định file nằm trong thư mục Deployment.
        """
        ckpt_path = BASE_DIR / checkpoint_name

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.transformer_device, weights_only=False)

        # ưu tiên dùng config lưu trong checkpoint
        if "configs" in ckpt and "hparams" in ckpt:
            cfg_dict = ckpt["configs"]
            hparams = ckpt["hparams"]
            configs = SimpleNamespace(**cfg_dict)
        else:
            configs, hparams = self._build_transformer_configs()

        model = ecgTransForm(configs=configs, hparams=hparams).to(self.transformer_device)

        if "model" not in ckpt:
            raise KeyError("Checkpoint không có key 'model'.")

        model.load_state_dict(ckpt["model"])
        model.eval()

        self.transformer_model = model
        return model

    def preprocess_signal_for_transformer(self, ecg_1dsignal):
        """
        Nhận:
            - pandas.DataFrame 1 hàng nhiều cột
            - numpy.ndarray
            - list
        Trả về tensor shape [1, 1, L]
        """
        if isinstance(ecg_1dsignal, pd.DataFrame):
            arr = ecg_1dsignal.values
        elif isinstance(ecg_1dsignal, pd.Series):
            arr = ecg_1dsignal.values
        else:
            arr = np.asarray(ecg_1dsignal)

        arr = np.asarray(arr, dtype=np.float32)

        # chuẩn hóa shape về [L]
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = arr.reshape(-1)
        elif arr.ndim == 1:
            pass
        else:
            raise ValueError(f"ecg_1dsignal có shape không hợp lệ: {arr.shape}")

        if np.isnan(arr).any():
            raise ValueError("ecg_1dsignal chứa NaN.")

        if len(arr) != self.expected_signal_length:
            raise ValueError(
                f"Độ dài tín hiệu không khớp dữ liệu train. "
                f"Expected={self.expected_signal_length}, got={len(arr)}"
            )

        x = torch.tensor(arr, dtype=torch.float32).contiguous().unsqueeze(0).unsqueeze(0)
        x = x.to(self.transformer_device)
        return x

    def predict_index_4class(self, ecg_1dsignal):
        """
        Trả về index 0..3
        """
        if self.transformer_model is None:
            self.load_transformer_checkpoint_4class("checkpoint_best_4class.pt")

        x = self.preprocess_signal_for_transformer(ecg_1dsignal)

        with torch.no_grad():
            logits = self.transformer_model(x)
            pred_idx = int(torch.argmax(logits, dim=1).cpu().item())

        return pred_idx

    def predict_label_4class(self, ecg_1dsignal):
        """
        Trả về nhãn đầy đủ
        """
        pred_idx = self.predict_index_4class(ecg_1dsignal)
        return self.label_map_4.get(pred_idx, str(pred_idx))

    def ModelLoad_predict_transformer_4class(self, ecg_1dsignal):
        """
        Hàm này dùng trực tiếp trong final_app.py
        """
        pred_idx = self.predict_index_4class(ecg_1dsignal)
        pred_label = self.label_map_4.get(pred_idx, str(pred_idx))
        return f"Prediction: {pred_label}"