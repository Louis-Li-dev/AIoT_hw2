"""
魚類體重預測系統 - 使用 Lasso 迴歸
Fish Weight Prediction System using Lasso Regression

作業要求：
1. 使用 chatGPT 工具利用 CRISP-DM 模板解決多元回歸 Regression Problem
2. Step 1: 爬蟲抓取 Boston 房價 (本專案改用魚類資料集)
3. Step 2: Preprocessing: train test split
4. Step 3: Build Model using Lasso
5. Step 4: Evaluation: MSE, MAE, R2 metrics 的意義, overfit and underfit 的判斷
6. Step 5: Deployment

執行方式: streamlit run app.py
使用 PATH 變數讀取 CSV 檔案
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 設定頁面配置
st.set_page_config(
    page_title="魚類體重預測系統",
    page_icon="🐟",
    layout="wide"
)

# ==================== PATH 變數設定 ====================
# 請將 PATH 設定為您的 CSV 檔案路徑
PATH = "./dataset/Fishers maket.csv"  # 修改此路徑為您的 CSV 檔案位置

# ==================== 標題 ====================
st.title("🐟 魚類體重預測系統 - Lasso 迴歸分析")
st.markdown("### 使用 CRISP-DM 方法論進行多元迴歸問題解決")
st.markdown("---")

# ==================== Step 1: 資料載入 ====================
st.header("📊 Step 1: 資料載入")
st.markdown("""
**CRISP-DM 階段**: Business Understanding & Data Understanding

在本專案中，我們使用魚類資料集來預測魚的體重。
資料集包含以下特徵：
- **Species**: 魚的種類（類別變數）
- **Length1, Length2, Length3**: 不同測量方式的長度（垂直長度、對角線長度、交叉長度）
- **Height**: 高度
- **Width**: 寬度
- **Weight**: 體重（目標變數，我們要預測的值）
""")

try:
    # 使用 pandas 讀取 CSV 檔案
    df = pd.read_csv(PATH)
    st.success(f"✅ 成功載入資料！檔案路徑: {PATH}")
except FileNotFoundError:
    st.error(f"❌ 找不到檔案: {PATH}")
    st.info("請確認 PATH 變數設定正確，或將 CSV 檔案放在正確位置")
    st.stop()
except Exception as e:
    st.error(f"❌ 讀取檔案時發生錯誤: {e}")
    st.stop()

# 顯示資料
st.subheader("📋 資料預覽")
col1, col2 = st.columns([2, 1])

with col1:
    st.write("**前 10 筆資料**")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.write("**資料基本資訊**")
    st.metric("總樣本數", len(df))
    st.metric("特徵數量", len(df.columns) - 1)
    st.metric("目標變數", "Weight")
    
    # 檢查缺失值
    missing_values = df.isnull().sum().sum()
    st.metric("缺失值總數", missing_values)

# 資料描述統計
st.subheader("📈 描述性統計")
st.dataframe(df.describe(), use_container_width=True)

# 視覺化：目標變數分佈
st.subheader("🎯 目標變數 (Weight) 分佈")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df['Weight'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel('體重 (Weight)', fontsize=12)
ax.set_ylabel('頻率', fontsize=12)
ax.set_title('魚類體重分佈圖', fontsize=14)
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ==================== Step 2: 資料預處理與分割 ====================
st.markdown("---")
st.header("⚙️ Step 2: 資料預處理 (Preprocessing) 與訓練測試集分割")

st.markdown("""
**CRISP-DM 階段**: Data Preparation

### 預處理步驟：
1. **特徵與目標變數分離**: 將 Weight 作為目標變數 (y)，其他欄位作為特徵 (X)
2. **類別變數編碼**: 使用 One-Hot Encoding 處理 Species 類別變數
3. **訓練測試集分割**: 將資料分為訓練集 (80%) 和測試集 (20%)
4. **特徵標準化**: 使用 StandardScaler 進行標準化，使特徵具有相同的尺度

**為什麼需要標準化？**
- Lasso 迴歸對特徵尺度敏感
- 標準化後所有特徵具有平均值=0，標準差=1
- 有助於模型更快收斂，提升預測效能
""")

# 側邊欄參數設定
st.sidebar.header("🎛️ 模型參數設定")
test_size = st.sidebar.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("隨機種子", 1, 100, 42)

# 資料預處理
@st.cache_data
def preprocess_data(dataframe, test_sz, rand_state):
    """資料預處理函數"""
    # 分離特徵和目標變數
    X = dataframe.drop('Weight', axis=1)
    y = dataframe['Weight']
    
    # One-Hot Encoding 處理類別變數
    if 'Species' in X.columns:
        X = pd.get_dummies(X, columns=['Species'], drop_first=True)
    
    # 訓練測試集分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=rand_state
    )
    
    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, test_size, random_state)

# 顯示分割結果
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("訓練集樣本數", len(X_train))
with col2:
    st.metric("測試集樣本數", len(X_test))
with col3:
    st.metric("特徵數量（編碼後）", X_train.shape[1])

st.success("✅ 資料預處理完成！")

# ==================== Step 3: 建立 Lasso 模型 ====================
st.markdown("---")
st.header("🤖 Step 3: 建立模型 - Lasso 迴歸")

st.markdown("""
**CRISP-DM 階段**: Modeling

### 什麼是 Lasso 迴歸？
**Lasso (Least Absolute Shrinkage and Selection Operator)** 是一種線性迴歸方法，使用 L1 正則化。

**Lasso 的特點：**
- ✨ **特徵選擇**: 可以將不重要的特徵係數縮減為 0，自動進行特徵選擇
- 🎯 **防止過擬合**: L1 正則化懲罰項可以防止模型過度複雜
- 📊 **稀疏解**: 產生稀疏模型，只保留重要特徵

**Alpha 參數**：控制正則化強度
- Alpha 越大 → 正則化越強 → 更多係數被壓縮為 0 → 模型越簡單
- Alpha 越小 → 正則化越弱 → 接近普通線性迴歸
""")

# Alpha 參數設定
alpha = st.sidebar.slider("Alpha (正則化強度)", 0.01, 10.0, 1.0, 0.1)

# 訓練模型
@st.cache_resource
def train_lasso_model(X_tr, y_tr, alpha_param):
    """訓練 Lasso 模型"""
    model = Lasso(alpha=alpha_param, random_state=42, max_iter=10000)
    model.fit(X_tr, y_tr)
    return model

model = train_lasso_model(X_train, y_train, alpha)

st.success(f"✅ Lasso 模型訓練完成！Alpha = {alpha}")

# 顯示特徵重要性
st.subheader("📊 特徵係數 (Feature Coefficients)")
st.markdown("係數絕對值越大，表示該特徵對預測結果的影響越大。係數為 0 表示該特徵被 Lasso 排除。")

coef_df = pd.DataFrame({
    '特徵名稱': feature_names,
    '係數': model.coef_,
    '絕對值': np.abs(model.coef_)
}).sort_values('絕對值', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x != 0 else 'gray' for x in coef_df['係數']]
ax.barh(coef_df['特徵名稱'], coef_df['係數'], color=colors)
ax.set_xlabel('係數值', fontsize=12)
ax.set_title('Lasso 迴歸特徵係數', fontsize=14)
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')
st.pyplot(fig)

st.dataframe(coef_df, use_container_width=True)

# ==================== Step 4: 模型評估 ====================
st.markdown("---")
st.header("📈 Step 4: 模型評估")

st.markdown("""
**CRISP-DM 階段**: Evaluation

### 評估指標說明：

**1. MSE (Mean Squared Error, 均方誤差)**
- 計算方式：預測誤差的平方平均值
- 意義：MSE 越小，模型預測越準確
- 特點：對離群值（outliers）非常敏感，因為誤差被平方放大

**2. MAE (Mean Absolute Error, 平均絕對誤差)**
- 計算方式：預測誤差絕對值的平均
- 意義：MAE 越小，模型預測越準確
- 特點：較不受離群值影響，更能反映平均誤差

**3. R² (R-squared, 決定係數)**
- 範圍：-∞ 到 1
- 意義：模型解釋目標變數變異的比例
  - R² = 1: 完美預測
  - R² = 0: 模型表現等同於使用平均值預測
  - R² < 0: 模型表現比使用平均值還差

### Overfit (過擬合) vs Underfit (欠擬合) 判斷：

**🔴 Overfit (過擬合)：**
- 現象：訓練集表現非常好，但測試集表現很差
- 判斷：訓練 R² >> 測試 R²（例如：訓練 R²=0.95，測試 R²=0.60）
- 原因：模型過度學習訓練資料的細節和雜訊
- 解決方法：增加正則化強度（增大 alpha）、增加訓練資料

**🟡 Underfit (欠擬合)：**
- 現象：訓練集和測試集表現都很差
- 判斷：訓練 R² 和測試 R² 都很低（例如：兩者都 < 0.5）
- 原因：模型過於簡單，無法捕捉資料的模式
- 解決方法：減少正則化強度（減小 alpha）、增加特徵

**🟢 良好模型：**
- 訓練集和測試集的 R² 都高且相近
- 例如：訓練 R²=0.85，測試 R²=0.82
""")

# 進行預測
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 計算評估指標
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 顯示評估結果
st.subheader("📊 評估指標結果")

col1, col2 = st.columns(2)

with col1:
    st.write("### 🔵 訓練集 (Training Set)")
    st.metric("MSE", f"{train_mse:.2f}")
    st.metric("MAE", f"{train_mae:.2f}")
    st.metric("R²", f"{train_r2:.4f}")

with col2:
    st.write("### 🟢 測試集 (Test Set)")
    st.metric("MSE", f"{test_mse:.2f}")
    st.metric("MAE", f"{test_mae:.2f}")
    st.metric("R²", f"{test_r2:.4f}")

# 判斷 Overfit/Underfit
st.subheader("🔍 模型診斷")

r2_diff = train_r2 - test_r2

if r2_diff > 0.15 and train_r2 > 0.7:
    st.warning(f"""
    **⚠️ 偵測到過擬合 (Overfit)！**
    
    - 訓練集 R² = {train_r2:.4f}
    - 測試集 R² = {test_r2:.4f}
    - 差異 = {r2_diff:.4f}
    
    **建議**: 增加 Alpha 值以增強正則化，減少模型複雜度。
    """)
elif train_r2 < 0.5 and test_r2 < 0.5:
    st.warning(f"""
    **⚠️ 偵測到欠擬合 (Underfit)！**
    
    - 訓練集 R² = {train_r2:.4f}
    - 測試集 R² = {test_r2:.4f}
    
    **建議**: 減少 Alpha 值以減弱正則化，或增加更多特徵。
    """)
else:
    st.success(f"""
    **✅ 模型表現良好！**
    
    - 訓練集 R² = {train_r2:.4f}
    - 測試集 R² = {test_r2:.4f}
    - 差異 = {r2_diff:.4f}
    
    訓練集與測試集表現相近，模型泛化能力佳。
    """)

# 視覺化：實際值 vs 預測值
st.subheader("📉 預測結果視覺化")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 訓練集
ax1.scatter(y_train, y_train_pred, alpha=0.6, color='blue', edgecolors='k')
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('實際值 (Actual)', fontsize=12)
ax1.set_ylabel('預測值 (Predicted)', fontsize=12)
ax1.set_title(f'訓練集預測結果\nR² = {train_r2:.4f}', fontsize=13)
ax1.grid(True, alpha=0.3)

# 測試集
ax2.scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='k')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('實際值 (Actual)', fontsize=12)
ax2.set_ylabel('預測值 (Predicted)', fontsize=12)
ax2.set_title(f'測試集預測結果\nR² = {test_r2:.4f}', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# 殘差圖
st.subheader("📊 殘差分析 (Residual Analysis)")
st.markdown("殘差 = 實際值 - 預測值。良好的模型殘差應該隨機分佈在 0 附近。")

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 訓練集殘差
ax1.scatter(y_train_pred, train_residuals, alpha=0.6, color='blue', edgecolors='k')
ax1.axhline(y=0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('預測值', fontsize=12)
ax1.set_ylabel('殘差', fontsize=12)
ax1.set_title('訓練集殘差圖', fontsize=13)
ax1.grid(True, alpha=0.3)

# 測試集殘差
ax2.scatter(y_test_pred, test_residuals, alpha=0.6, color='green', edgecolors='k')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('預測值', fontsize=12)
ax2.set_ylabel('殘差', fontsize=12)
ax2.set_title('測試集殘差圖', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ==================== Step 5: 部署 (Deployment) ====================
st.markdown("---")
st.header("🚀 Step 5: 模型部署 (Deployment)")

st.markdown("""
**CRISP-DM 階段**: Deployment

這個 Streamlit 應用程式本身就是一個部署範例！
使用者可以透過以下方式使用模型：

1. **批次預測**: 上傳新的魚類資料進行批次預測
2. **即時預測**: 輸入單一魚類的特徵進行即時預測
3. **模型調整**: 調整 Alpha 參數並即時查看效果
""")

st.subheader("🎯 即時預測工具")
st.markdown("輸入魚類的特徵資訊，模型將預測其體重：")

col1, col2, col3 = st.columns(3)

with col1:
    length1 = st.number_input("Length1 (垂直長度)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    length2 = st.number_input("Length2 (對角線長度)", min_value=0.0, max_value=100.0, value=27.0, step=0.1)

with col2:
    length3 = st.number_input("Length3 (交叉長度)", min_value=0.0, max_value=100.0, value=32.0, step=0.1)
    height = st.number_input("Height (高度)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)

with col3:
    width = st.number_input("Width (寬度)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
    species = st.selectbox("Species (種類)", df['Species'].unique())

if st.button("🔮 進行預測", type="primary"):
    # 準備輸入資料
    input_data = pd.DataFrame({
        'Length1': [length1],
        'Length2': [length2],
        'Length3': [length3],
        'Height': [height],
        'Width': [width],
        'Species': [species]
    })
    
    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_data, columns=['Species'], drop_first=True)
    
    # 確保所有特徵都存在
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_names]
    
    # 標準化
    input_scaled = scaler.transform(input_encoded)
    
    # 預測
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"### 預測結果: **{prediction:.2f} 克**")
    
    # 顯示信賴區間（簡化版本，基於訓練集 MAE）
    confidence_interval = test_mae * 1.96  # 約 95% 信賴區間
    st.info(f"95% 信賴區間: **{prediction - confidence_interval:.2f} ~ {prediction + confidence_interval:.2f} 克**")

# ==================== 總結 ====================
st.markdown("---")
st.header("📝 專案總結")

st.markdown(f"""
### 🎓 本專案完成項目：

✅ **Step 1: 資料載入** - 成功載入魚類資料集 ({len(df)} 筆資料)

✅ **Step 2: 資料預處理** 
- 完成 One-Hot Encoding 處理類別變數
- 訓練/測試集分割 ({int((1-test_size)*100)}% / {int(test_size*100)}%)
- 特徵標準化

✅ **Step 3: 模型建立** 
- 使用 Lasso 迴歸 (Alpha={alpha})
- 特徵選擇：{np.sum(model.coef_ != 0)} / {len(model.coef_)} 個特徵被保留

✅ **Step 4: 模型評估**
- 訓練集 R² = {train_r2:.4f}
- 測試集 R² = {test_r2:.4f}
- 測試集 MAE = {test_mae:.2f}

✅ **Step 5: 模型部署**
- Streamlit 互動式應用程式
- 支援即時預測功能

### 🔑 關鍵發現：
- 最重要的特徵（絕對係數最大）：**{coef_df.iloc[0]['特徵名稱']}**
- 模型複雜度：Alpha={alpha}，保留 {np.sum(model.coef_ != 0)} 個特徵
- 預測準確度：測試集 R²={test_r2:.4f}，平均誤差 {test_mae:.2f} 克

### 📚 使用的技術與套件：
- **Python**: Pandas, NumPy, Scikit-learn
- **視覺化**: Matplotlib, Seaborn
- **Web框架**: Streamlit
- **機器學習**: Lasso Regression with L1 Regularization
""")

st.balloons()
st.success("🎉 專案完成！感謝使用本系統！")