# 3D 鎖骨下動脈分類系統

使用傳統機器學習和深度學習方法對鎖骨下動脈 3D 模型進行分類的綜合機器學習專案。

## 專案概述

本專案實現了多種方法來分類 3D 血管模型（STL 檔案）並結合解剖測量：

- **傳統機器學習**：使用手工提取的幾何特徵的隨機森林、XGBoost、梯度提升
- **深度學習**：使用體素表示的 3D CNN
- **混合方法**：點雲、體素和解剖測量的多模態融合

## 成果摘要

| 模型 | 交叉驗證準確率 | 訓練時間 |
|-----|---------------|---------|
| **隨機森林** | 82.98% ± 3.91% | < 1 秒 |
| **梯度提升** | 82.98% ± 6.12% | < 1 秒 |
| 混合深度學習 | 79.77% ± 4.03% | ~5 分鐘 |
| XGBoost | 77.60% ± 4.28% | < 1 秒 |

## 資料設置步驟

### 📁 資料結構

```
專案資料夾/
├── STL/                     # 94個STL檔案（約600 MB）- 3D血管模型
├── numpy_arrays/            # 預處理的陣列（約300 MB）
├── voxel_data/             # 體素表示（約150 MB）
├── hybrid_data/            # 混合模型資料（約400 MB）
│   ├── pointclouds/        # 點雲資料
│   └── voxels/            # 體素資料
├── models/                 # 訓練好的模型（約100 MB）
└── classification_labels_with_measurements.csv  # 標籤檔案（含解剖測量）
```

### 🚀 快速開始

#### 步驟 1：複製專案並安裝依賴

```bash
# 複製儲存庫
git clone https://github.com/yourusername/subclavian-artery-classification.git
cd subclavian-artery-classification

# 安裝相依套件
pip install -r requirements.txt
```

#### 步驟 2：設置資料

執行互動式設置腳本：
```bash
python setup_data.py
```

### 📥 資料取得方式

#### 方式一：Google Drive 下載（推薦）

1. 資料提供者將資料上傳至 Google Drive
2. 更新 `setup_data.py` 中的連結
3. 執行自動下載：
   ```bash
   python setup_data.py --source gdrive
   ```

#### 方式二：本地複製

從本地備份位置複製資料：
```bash
python setup_data.py --source local --path "D:\備份\專案資料"
```

#### 方式三：從 STL 檔案生成

如果只有 STL 檔案，可以自動生成所有預處理資料：
```bash
python setup_data.py --generate
```

### ✅ 設置驗證

成功設置後應顯示：
- ✓ STL 檔案：94 個檔案就緒
- ✓ 找到標籤檔案（含解剖測量）
- ✓ 體素資料：94 個檔案
- ✓ 找到訓練模型

### 📊 檔案大小參考

| 元件 | 檔案數量 | 總大小 | 是否必要？ |
|-----|---------|--------|-----------|
| STL 檔案 | 94 | ~600 MB | 必要 |
| 標籤 CSV | 2 | <1 MB | 必要 |
| Numpy 陣列 | 94 | ~300 MB | 可重新生成 |
| 體素資料 | 94 | ~150 MB | 可重新生成 |
| 混合資料 | 188 | ~400 MB | 可重新生成 |
| 訓練模型 | 5-10 | ~100 MB | 可重新訓練 |

**最小需求**：STL 檔案 + 標籤 CSV（約 601 MB）  
**完整套件**：所有元件（約 1.5 GB）

## 系統需求

- Python 3.8+
- PyTorch 1.9+（建議支援 CUDA）
- scikit-learn
- trimesh
- numpy
- pandas
- matplotlib
- xgboost

## 使用方法

### 1. 傳統機器學習分類
```python
python traditional_ml_approach.py
```

### 2. 混合深度學習
```python
python hybrid_multimodal_model.py
```

### 3. 交叉驗證分析
```python
python cross_validation_analysis.py
```

### 4. 基於體素的 CNN
```python
python gpu_voxel_training.py
```

## 資料格式

### STL 檔案
將 3D 血管模型放置在 `STL/` 目錄中

### 標籤 CSV
創建 `classification_labels_with_measurements.csv`，包含以下欄位：
- `filename`：STL 檔名（不含副檔名）
- `label`：二元分類（0 或 1）
- `left_subclavian_diameter_mm`：血管直徑
- `aortic_arch_diameter_mm`：主動脈弓直徑
- `angle_degrees`：解剖角度

## 主要發現

1. **傳統機器學習**在小資料集（< 100 樣本）表現最佳
2. **隨機森林**提供最佳的穩定性和速度
3. **深度學習**需要 500+ 樣本才能達到最佳性能
4. **解剖測量**是關鍵特徵（23% 重要性）

## 性能分析

使用 95 個樣本：
- 傳統機器學習達到 ~83% 準確率
- 深度學習因資料不足限制在 ~80%
- 交叉驗證顯示各折之間有 3-6% 的變異

## 故障排除

### "找不到 STL 檔案"
- 檢查 `STL/` 目錄是否存在
- 驗證檔案是否有 `.stl` 副檔名
- 執行：`python setup_data.py` 並選擇下載選項

### "缺少標籤檔案"
- 確保 `classification_labels_with_measurements.csv` 在根目錄中
- 檢查是否被 `.gitignore` 排除

### "匯入錯誤：trimesh"
```bash
pip install trimesh
```

### "Google Drive 下載失敗"
- 安裝 gdown：`pip install gdown`
- 檢查檔案共享設定（必須是"知道連結的任何人"）
- 嘗試直接下載並手動解壓縮

## 未來改進

1. **資料收集**：目標 500+ 樣本以達到 90%+ 準確率
2. **遷移學習**：使用預訓練的 3D 醫學模型
3. **集成方法**：結合多種方法
4. **資料增強**：合成資料生成

## 資料隱私注意事項

如果您的 STL 檔案包含敏感醫療資料：
1. 使用私人 Google Drive 連結
2. 為 zip 檔案添加密碼保護
3. 考慮加密：`7z a -p"密碼" secure_data.7z STL/`
4. 使用安全傳輸方法
5. 在儲存庫中添加資料使用協議

## 聯繫方式

如需存取原始資料集或有任何問題：
- 在 GitHub 上開啟 issue
- 請說明您的使用案例和所屬機構

## 授權

MIT 授權 - 詳見 LICENSE 檔案

## 引用

如果您使用此程式碼，請引用：
```
@software{subclavian_classification,
  title = {3D Subclavian Artery Classification},
  year = {2024},
  url = {https://github.com/yourusername/subclavian-artery-classification}
}
```