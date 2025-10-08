# 🐟 魚類體重預測系統（Lasso＋Optuna）—技術報告（Markdown，繁體中文）

> 本文件針對你提供的 `Streamlit` 應用程式進行**關鍵程式碼擷取**與**逐行說明**，並以 **CRISP-DM** 步驟撰寫。文末補充常見錯誤與修正建議（含資料路徑小陷阱）。
> Demo Link [https://aiot-hw2.streamlit.app/](https://aiot-hw2.streamlit.app/)
> GPT Prompt Link [https://chatgpt.com/share/68e5cf49-d8d0-8013-b6b4-984a12600050](https://chatgpt.com/share/68e5cf49-d8d0-8013-b6b4-984a12600050)
---

## 🔗 資料集連結（Fish Market Dataset）

Kaggle：[https://www.kaggle.com/datasets/salman1127/fish-market-dataset](https://www.kaggle.com/datasets/salman1127/fish-market-dataset)

> 通常檔名為 **`Fish.csv`**。請留意你的程式中的 `PATH` 目前寫成 `./dataset/Fishers maket.csv`（有拼字與檔名誤差），建議改為：`PATH = "./dataset/Fish.csv"`。

---

## 📘 作業指引（原樣重現）

1. 再次使用chatGPT 工具利用CRISP-DM模板解決 多元回歸 Regression Problem
2. Step 1: 爬蟲抓取Boston房價
3. Step 2: Preprocessing : train test split
4. Step 3: Build Model using Lasso
5. Step 4: Evaluation: MSE, MAE, R2 metrics 的意義, overfit and underfit 的判斷（畫出 training, test curve）, 優化模型 optuna
6. Step 5: Deployment

> 本專案以**魚類資料集**替代 Boston 房價，仍依 CRISP-DM 流程完成多元迴歸專題。

---

# CRISP-DM 架構對應

* **Business & Data Understanding（Tab1）：** 載入資料、統計摘要、目標分佈
* **Data Preparation（Tab2）：** 類別編碼、資料分割、標準化、相關性圖
* **Modeling（Tab3）：** Lasso 迴歸（手動 α 或 Optuna 搜尋）
* **Evaluation（Tab4/Tab5）：** 訓練/測試指標、過/欠擬合判斷、殘差診斷、Optuna 視覺化
* **Deployment（Tab6）：** 即時與批次推論、結果下載

---

## 0) 執行方式

```bash
streamlit run app.py
```

---

## 1) 匯入與全域設定（逐行重點）

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
import warnings
warnings.filterwarnings('ignore')
```

* `streamlit`：前端互動式儀表板。
* `pandas/numpy`：資料處理、向量運算。
* `sklearn`：資料分割、特徵標準化、Lasso 模型、評估指標、交叉驗證。
* `matplotlib/seaborn`：繪圖。
* `optuna`：自動化超參數優化（這裡用來尋找 **alpha**）。
* 關閉警告訊息以保持 UI 乾淨。

```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

* 設定中文字型與負號顯示，避免亂碼/負號方塊。

```python
st.set_page_config(page_title="魚類體重預測系統", page_icon="🐟", layout="wide")
```

* 設定頁面標題、favicon、寬版版面。

```python
PATH = "./dataset/Fishers maket.csv"
```

* **注意：** 建議改為 `PATH = "./dataset/Fish.csv"`（Kaggle 典型檔名）。

---

## 2) 側邊欄參數（逐行重點）

```python
st.sidebar.header("🎛️ 參數設定")
st.sidebar.subheader("📊 資料分割參數")
test_size = st.sidebar.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("隨機種子", 1, 100, 42)

st.sidebar.subheader("🤖 模型參數")
use_optuna = st.sidebar.checkbox("使用 Optuna 優化", value=False)

if not use_optuna:
    alpha = st.sidebar.slider("Alpha (正則化強度)", 0.01, 10.0, 1.0, 0.1)
else:
    st.sidebar.info("✨ Optuna 將自動尋找最佳 Alpha 值")
    n_trials = st.sidebar.slider("Optuna 試驗次數", 10, 200, 50, 10)
```

* **互動控制**：可調整資料切分比例、隨機種子；選擇手動 α 或 Optuna 搜尋 α；設定 Optuna 試驗數。

---

## 3) 載入資料（cache＋例外處理）

```python
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_data(PATH)

if error:
    st.error(f"❌ {error}")
    st.stop()
```

* `@st.cache_data`：相同輸入時快取結果，避免重複讀檔。
* 以 `try/except` 回傳錯誤字串；若失敗直接終止應用（顯示錯誤）。

---

## 4) 資料預處理（編碼、切分、標準化）

```python
@st.cache_data
def preprocess_data(dataframe, test_sz, rand_state):
    X = dataframe.drop('Weight', axis=1)
    y = dataframe['Weight']
  
    if 'Species' in X.columns:
        X = pd.get_dummies(X, columns=['Species'], drop_first=True)
  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=rand_state
    )
  
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
  
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, test_size, random_state)
```

* **目標欄位**：`Weight`
* **類別編碼**：`Species` → one-hot（`drop_first=True` 防多重共線性）
* **資料切分**：訓練/測試（保留 `random_state` 可重現）
* **標準化**：`StandardScaler`（以訓練集 fit、套用至測試集）
* **回傳**：縮放後資料、`scaler`（部署時要一致轉換）、`feature_names`（畫圖/推論排列對齊）

---

## 5) Optuna 目標函數與優化（交叉驗證 MSE）

```python
def objective(trial, X_tr, y_tr):
    alpha_param = trial.suggest_float('alpha', 0.001, 10.0, log=True)
    model = Lasso(alpha=alpha_param, random_state=42, max_iter=10000)
    scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()
```

* **搜尋空間**：`alpha ∈ [0.001, 10]`，log 尺度（適合正則化強度）
* **評估**：5-fold CV 的 **MSE**（`sklearn` 回傳負值，故取負號）

```python
@st.cache_resource
def optimize_with_optuna(X_tr, y_tr, n_trials_param):
    study = optuna.create_study(direction='minimize', study_name='lasso_optimization')
    study.optimize(lambda trial: objective(trial, X_tr, y_tr), n_trials=n_trials_param, show_progress_bar=False)
    return study
```

* **資源型快取**：避免每次互動都重跑試驗
* **目標**：最小化 CV-MSE

---

## 6) 模型訓練（Lasso）

```python
@st.cache_resource
def train_lasso_model(X_tr, y_tr, alpha_param):
    model = Lasso(alpha=alpha_param, random_state=42, max_iter=10000)
    model.fit(X_tr, y_tr)
    return model
```

* 指定 `max_iter=10000` 以確保收斂；回傳已擬合的模型。

---

## 7) 統一評估函式（訓練/測試＋預測回傳）

```python
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)
  
    return {
        'train': {
            'mse': mean_squared_error(y_tr, y_train_pred),
            'mae': mean_absolute_error(y_tr, y_train_pred),
            'r2': r2_score(y_tr, y_train_pred)
        },
        'test': {
            'mse': mean_squared_error(y_te, y_test_pred),
            'mae': mean_absolute_error(y_te, y_test_pred),
            'r2': r2_score(y_te, y_test_pred)
        },
        'predictions': {
            'train': (y_tr, y_train_pred),
            'test': (y_te, y_test_pred)
        }
    }
```

* 同時計算 **MSE／MAE／R²**，並回傳實際與預測值供後續視覺化使用。

---

## 8) 多分頁（Tabs）— 對應 CRISP-DM

```python
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Step 1: 資料載入", 
    "⚙️ Step 2: 資料預處理",
    "🤖 Step 3: 模型建立",
    "📈 Step 4: 模型評估",
    "✨ Step 4+: Optuna 優化",
    "🚀 Step 5: 模型部署"
])
```

* 呈現完整流程導覽，評量與優化拆頁顯示。

---

## 9) Tab1：資料理解（統計＋分佈）

**關鍵片段：**

```python
st.success(f"✅ 成功載入資料！檔案路徑: {PATH}")
st.dataframe(df.head(15), use_container_width=True)
st.dataframe(df.describe(), use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df['Weight'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
st.pyplot(fig)
```

* 顯示資料預覽、描述統計、**目標變數分佈**（檢查偏態與極端值）。

---

## 10) Tab2：資料準備（切分、特徵、相關性）

**關鍵片段：**

```python
st.metric("訓練集樣本數", len(X_train))
st.metric("測試集樣本數", len(X_test))
st.write(list(feature_names))

X_df = pd.DataFrame(X_train, columns=feature_names)
X_df['Weight'] = y_train.values
sns.heatmap(X_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
```

* 顯示切分後數量與**編碼後特徵清單**。
* 以訓練集（已標準化值）估算相關性（加入 `Weight` 檢視關聯方向/強度）。

---

## 11) Tab3：建模（手動 α vs Optuna）

**關鍵片段（Optuna 分支）**

```python
study = optimize_with_optuna(X_train, y_train, n_trials)
best_alpha = study.best_params['alpha']
model = train_lasso_model(X_train, y_train, best_alpha)
current_alpha = best_alpha
```

**關鍵片段（手動分支）**

```python
model = train_lasso_model(X_train, y_train, alpha)
current_alpha = alpha
```

**係數視覺化與稀疏度**

```python
coef_df = pd.DataFrame({
    '特徵名稱': feature_names,
    '係數': model.coef_,
    '絕對值': np.abs(model.coef_)
}).sort_values('絕對值', ascending=False)

colors = ['green' if x != 0 else 'lightgray' for x in coef_df['係數']]
ax.barh(coef_df['特徵名稱'], coef_df['係數'], color=colors, edgecolor='black')
non_zero = np.sum(model.coef_ != 0)
```

* **Lasso 特性**：以 L1 正則化實現**特徵選擇**（係數推為 0）。
* 圖上以灰色突出被「壓成 0」的特徵，綠色為保留特徵。

---

## 12) Tab4：模型評估與診斷

**整體指標**

```python
metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
st.metric("MSE", f"{metrics['test']['mse']:.2f}")
st.metric("MAE", f"{metrics['test']['mae']:.2f}")
st.metric("R²", f"{metrics['test']['r2']:.4f}")
```

**過/欠擬合偵測**

```python
train_r2 = metrics['train']['r2']; test_r2 = metrics['test']['r2']
r2_diff = train_r2 - test_r2

if r2_diff > 0.15 and train_r2 > 0.7:
    st.warning("過擬合")
elif train_r2 < 0.5 and test_r2 < 0.5:
    st.warning("欠擬合")
else:
    st.success("模型表現良好")
```

* **判準直觀**：
  * 訓練很好但測試明顯差 → 過擬合
  * 兩邊都差 → 欠擬合
  * 兩邊都不錯、差距小 → 合理擬合

**預測對角線散佈圖＋殘差圖**

```python
# y_true vs y_pred；45° 對角線理想
ax1.scatter(y_tr, y_train_pred)
ax1.plot([y_tr.min(), y_tr.max()], [y_tr.min(), y_tr.max()], 'r--')

# 殘差 vs 預測；檢查是否有結構性偏誤（非隨機）
ax2.scatter(y_test_pred, test_residuals)
ax2.axhline(y=0, color='r', linestyle='--')
```

---

## 13) Tab5：Optuna 視覺化與成效對比

**最佳解摘要與歷史**

```python
st.metric("最佳 Alpha", f"{study.best_params['alpha']:.6f}")
st.metric("最佳 MSE", f"{study.best_value:.2f}")

fig = plot_optimization_history(study)  # 若失敗則自繪曲線
```

**參數分佈與表現**

```python
trials_df = study.trials_dataframe()[['number','value','params_alpha','state']]
ax.scatter(trials_df['Alpha'], trials_df['MSE'])
ax.set_xscale('log')
```

* 檢視 α（log 軸）與 MSE 的關係，標出最佳 α。

**與預設 α=1.0 對比**

```python
default_model = train_lasso_model(X_train, y_train, 1.0)
default_metrics = evaluate_model(default_model, X_train, X_test, y_train, y_test)

improvement_r2  = ((metrics['test']['r2'] - default_metrics['test']['r2']) / abs(default_metrics['test']['r2']) * 100)
improvement_mse = ((default_metrics['test']['mse'] - metrics['test']['mse']) / default_metrics['test']['mse'] * 100)
```

* 提供**效能改善百分比**，量化 Optuna 帶來的好處。

---

## 14) Tab6：部署（即時／批次推論）

**即時推論 UI 與前處理**

```python
# 以原訓練特徵順序對齊，缺少的 dummy 欄補 0
input_encoded = pd.get_dummies(input_data, columns=['Species'], drop_first=True)
for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_names]

input_scaled = scaler.transform(input_encoded)
prediction = model.predict(input_scaled)[0]
```

* **關鍵**：**推論時的特徵欄位順序**必須與訓練一致；沒出現的類別 dummy 欄補 0。
* 以訓練時 `scaler` 進行相同標準化，再 `model.predict`。

**以 MAE 粗估 95% 信賴區間（近似）**

```python
confidence_interval = metrics['test']['mae'] * 1.96
```

* 實務上 MAE 並非標準差，但可作為**粗略**不確定性範圍示意。

**批次推論與下載**

```python
uploaded_file = st.file_uploader("上傳 CSV 檔案", type=['csv'])
# ... 同步一樣的編碼/標準化/預測 ...
st.download_button("📥 下載結果", csv, "predictions.csv", "text/csv", type="primary")
```

---

## 15) 頁尾總結與視覺效果

```python
st.balloons()
st.success("🎉 專案完成！感謝使用本系統！")
```

* 完成動畫與摘要指標（顯示 α、R²、MAE、最重要特徵）。

---

# 評估指標意義（快速複習）

* **MSE**（均方誤差）：懲罰大誤差更重；對離群值敏感。
* **MAE**（平均絕對誤差）：誤差的平均絕對值，易解釋、抗離群。
* **R²**（決定係數）：解釋變異比例，越接近 1 越好；低於 0 代表比常數模型還差。
* **Overfit／Underfit 判斷**：
  * **Overfit**：訓練 R² 高、測試 R² 顯著低（差距大）；
  * **Underfit**：訓練、測試 R² 都低；
  * **Balanced**：兩者皆高且差距小。

---

# 常見錯誤與修正建議（務必檢查）

1. **資料路徑錯誤**

```python
# 原本
PATH = "./dataset/Fishers maket.csv"

# 建議（典型檔名）
PATH = "./dataset/Fish.csv"
```

2. **類別欄位對齊**：部署（即時/批次）時一定要

```python
# 逐欄補齊缺失 dummy
for col in feature_names:
    if col not in X_batch_encoded.columns:
        X_batch_encoded[col] = 0
X_batch_encoded = X_batch_encoded[feature_names]
```

3. **快取失效**（當你改了資料路徑或參數）

* 若遇到內容不更新，請在 Streamlit 介面點擊 **Rerun** 或清快取。
* `@st.cache_data` 針對資料、`@st.cache_resource` 針對模型/Study 物件。

4. **Optuna 圖形相依**

* `plot_optimization_history` 失敗時你已提供後備手繪折線，這很實用。
* 若要再加「參數重要度」，可補：
  ```python
  try:
      fig_imp = plot_param_importances(study)
      st.pyplot(fig_imp)
  except:
      pass
  ```

5. **收斂問題**

* 若出現 Lasso 未收斂，可再調高 `max_iter` 或調整 `alpha` 範圍。

---

# 可延伸改進（選讀）

* **交叉驗證切分法**：以 `KFold`/`GroupKFold` 提升估計穩定性。
* **特徵工程**：長度、寬度、身高等欄位交互項（例如 `Length×Width`），或對高度做對數變換。
* **穩健評估**：加入 `MedAE`（中位數絕對誤差）、`RMSE`（根號 MSE）。
* **比較基準**：加入 Ridge／ElasticNet／RandomForest 對照。
* **模型解釋**：以 `Permutation Importance` 或 SHAP（回歸）補充重要度。

---

## 小抄：本系統與 CRISP-DM 的對齊（一句話版本）

* **理解**（Tab1）：看資料、看 Weight 分佈；
* **準備**（Tab2）：One-hot＋Split＋Scale＋看相關性；
* **建模**（Tab3）：Lasso（手動/Optuna）＋係數稀疏；
* **評估**（Tab4/5）：R²/MSE/MAE、過擬合檢查、殘差、Optuna 視覺化與對比；
* **部署**（Tab6）：欄位對齊→標準化→推論→下載。

---

> 以上 Markdown 可直接作為作業報告提交；若需要我把它**排版成更正式的教學講義**（含圖標、章節編號、縮排表格），或**改寫為中英雙語版**，告訴我你偏好的格式即可。
>
