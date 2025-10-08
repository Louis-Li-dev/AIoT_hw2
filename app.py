"""
魚類體重預測系統 - 使用 Lasso 迴歸與 Optuna 超參數優化
Fish Weight Prediction System using Lasso Regression with Optuna

作業要求：
1. 使用 chatGPT 工具利用 CRISP-DM 模板解決多元回歸 Regression Problem
2. Step 1: 爬蟲抓取 Boston 房價 (本專案改用魚類資料集)
3. Step 2: Preprocessing: train test split
4. Step 3: Build Model using Lasso
5. Step 4: Evaluation: MSE, MAE, R2 metrics 的意義, overfit and underfit 的判斷, 優化模型 Optuna
6. Step 5: Deployment

執行方式: streamlit run app.py
"""

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

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 設定頁面配置
st.set_page_config(page_title="魚類體重預測系統", page_icon="🐟", layout="wide")

# PATH 變數設定
PATH = "./dataset/Fishers maket.csv"

# 標題
st.title("🐟 魚類體重預測系統 - Lasso 迴歸分析")
st.markdown("### 使用 CRISP-DM 方法論與 Optuna 超參數優化")
st.markdown("---")

# 側邊欄參數設定
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

# 載入資料
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

# 資料預處理
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

# Optuna 優化函數
def objective(trial, X_tr, y_tr):
    alpha_param = trial.suggest_float('alpha', 0.001, 10.0, log=True)
    model = Lasso(alpha=alpha_param, random_state=42, max_iter=10000)
    scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()

@st.cache_resource
def optimize_with_optuna(X_tr, y_tr, n_trials_param):
    study = optuna.create_study(direction='minimize', study_name='lasso_optimization')
    study.optimize(lambda trial: objective(trial, X_tr, y_tr), n_trials=n_trials_param, show_progress_bar=False)
    return study

# 訓練模型
@st.cache_resource
def train_lasso_model(X_tr, y_tr, alpha_param):
    model = Lasso(alpha=alpha_param, random_state=42, max_iter=10000)
    model.fit(X_tr, y_tr)
    return model

# 模型評估函數
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

# 主要 Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Step 1: 資料載入", 
    "⚙️ Step 2: 資料預處理",
    "🤖 Step 3: 模型建立",
    "📈 Step 4: 模型評估",
    "✨ Step 4+: Optuna 優化",
    "🚀 Step 5: 模型部署"
])

# Tab 1: 資料載入
with tab1:
    st.header("📊 Step 1: 資料載入")
    st.markdown("**CRISP-DM 階段**: Business Understanding & Data Understanding")
    st.success(f"✅ 成功載入資料！檔案路徑: {PATH}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📋 資料預覽")
        st.dataframe(df.head(15), use_container_width=True)
    
    with col2:
        st.subheader("📊 資料資訊")
        st.metric("總樣本數", len(df))
        st.metric("特徵數量", len(df.columns) - 1)
        st.metric("缺失值總數", df.isnull().sum().sum())
    
    st.subheader("📈 描述性統計")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("🎯 目標變數 (Weight) 分佈")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['Weight'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('體重 (Weight)', fontsize=12)
    ax.set_ylabel('頻率', fontsize=12)
    ax.set_title('魚類體重分佈圖', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

# Tab 2: 資料預處理
with tab2:
    st.header("⚙️ Step 2: 資料預處理與分割")
    st.markdown("**CRISP-DM 階段**: Data Preparation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("訓練集樣本數", len(X_train))
    with col2:
        st.metric("測試集樣本數", len(X_test))
    with col3:
        st.metric("特徵數量（編碼後）", X_train.shape[1])
    
    st.success("✅ 資料預處理完成！")
    
    st.subheader("📋 處理後的特徵列表")
    st.write(list(feature_names))
    
    st.subheader("🔥 特徵相關性熱圖")
    X_df = pd.DataFrame(X_train, columns=feature_names)
    X_df['Weight'] = y_train.values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(X_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('特徵相關性矩陣', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tab 3: 模型建立
with tab3:
    st.header("🤖 Step 3: 建立模型 - Lasso 迴歸")
    st.markdown("**CRISP-DM 階段**: Modeling")
    
    if use_optuna:
        st.info("🔄 使用 Optuna 尋找最佳 Alpha 值...")
        with st.spinner("正在優化超參數..."):
            study = optimize_with_optuna(X_train, y_train, n_trials)
            best_alpha = study.best_params['alpha']
            st.success(f"✅ Optuna 優化完成！最佳 Alpha = {best_alpha:.6f}")
        
        model = train_lasso_model(X_train, y_train, best_alpha)
        current_alpha = best_alpha
    else:
        model = train_lasso_model(X_train, y_train, alpha)
        current_alpha = alpha
        st.success(f"✅ Lasso 模型訓練完成！Alpha = {current_alpha}")
    
    st.subheader("📊 特徵係數")
    coef_df = pd.DataFrame({
        '特徵名稱': feature_names,
        '係數': model.coef_,
        '絕對值': np.abs(model.coef_)
    }).sort_values('絕對值', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x != 0 else 'lightgray' for x in coef_df['係數']]
        ax.barh(coef_df['特徵名稱'], coef_df['係數'], color=colors, edgecolor='black')
        ax.set_xlabel('係數值', fontsize=12)
        ax.set_title('Lasso 迴歸特徵係數', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.dataframe(coef_df, use_container_width=True, height=400)
    
    non_zero = np.sum(model.coef_ != 0)
    total = len(model.coef_)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("保留特徵數", non_zero)
    with col2:
        st.metric("排除特徵數", total - non_zero)
    with col3:
        st.metric("特徵保留率", f"{non_zero/total*100:.1f}%")

# Tab 4: 模型評估
with tab4:
    st.header("📈 Step 4: 模型評估")
    st.markdown("**CRISP-DM 階段**: Evaluation")
    
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    st.subheader("📊 評估指標結果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔵 訓練集 (Training Set)")
        st.metric("MSE", f"{metrics['train']['mse']:.2f}")
        st.metric("MAE", f"{metrics['train']['mae']:.2f}")
        st.metric("R²", f"{metrics['train']['r2']:.4f}")
    
    with col2:
        st.write("### 🟢 測試集 (Test Set)")
        st.metric("MSE", f"{metrics['test']['mse']:.2f}")
        st.metric("MAE", f"{metrics['test']['mae']:.2f}")
        st.metric("R²", f"{metrics['test']['r2']:.4f}")
    
    st.subheader("🔍 模型診斷 - Overfit/Underfit 判斷")
    
    train_r2 = metrics['train']['r2']
    test_r2 = metrics['test']['r2']
    r2_diff = train_r2 - test_r2
    
    if r2_diff > 0.15 and train_r2 > 0.7:
        st.warning(f"**⚠️ 偵測到過擬合 (Overfit)！** 訓練 R²={train_r2:.4f}, 測試 R²={test_r2:.4f}, 差異={r2_diff:.4f}")
    elif train_r2 < 0.5 and test_r2 < 0.5:
        st.warning(f"**⚠️ 偵測到欠擬合 (Underfit)！** 訓練 R²={train_r2:.4f}, 測試 R²={test_r2:.4f}")
    else:
        st.success(f"**✅ 模型表現良好！** 訓練 R²={train_r2:.4f}, 測試 R²={test_r2:.4f}, 差異={r2_diff:.4f}")
    
    st.subheader("📉 預測結果視覺化")
    
    y_tr, y_train_pred = metrics['predictions']['train']
    y_te, y_test_pred = metrics['predictions']['test']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(y_tr, y_train_pred, alpha=0.6, color='blue', edgecolors='k', s=50)
    ax1.plot([y_tr.min(), y_tr.max()], [y_tr.min(), y_tr.max()], 'r--', lw=2)
    ax1.set_xlabel('實際值', fontsize=12)
    ax1.set_ylabel('預測值', fontsize=12)
    ax1.set_title(f'訓練集 (R²={train_r2:.4f})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(y_te, y_test_pred, alpha=0.6, color='green', edgecolors='k', s=50)
    ax2.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', lw=2)
    ax2.set_xlabel('實際值', fontsize=12)
    ax2.set_ylabel('預測值', fontsize=12)
    ax2.set_title(f'測試集 (R²={test_r2:.4f})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("📊 殘差分析")
    train_residuals = y_tr - y_train_pred
    test_residuals = y_te - y_test_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(y_train_pred, train_residuals, alpha=0.6, color='blue', edgecolors='k', s=50)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('預測值', fontsize=12)
    ax1.set_ylabel('殘差', fontsize=12)
    ax1.set_title('訓練集殘差圖', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(y_test_pred, test_residuals, alpha=0.6, color='green', edgecolors='k', s=50)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('預測值', fontsize=12)
    ax2.set_ylabel('殘差', fontsize=12)
    ax2.set_title('測試集殘差圖', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tab 5: Optuna 優化
with tab5:
    st.header("✨ Step 4+: 使用 Optuna 優化模型")
    
    st.markdown("""
    **什麼是 Optuna？**
    
    Optuna 是一個自動化超參數優化框架，可以幫助我們找到最佳的模型參數。
    
    ### Optuna 的優勢：
    - 🎯 **自動化搜尋**: 自動探索參數空間
    - 📊 **高效率**: 使用 TPE 演算法，比網格搜尋更快
    - 📈 **視覺化**: 提供豐富的視覺化工具
    """)
    
    if not use_optuna:
        st.info("💡 請在左側邊欄勾選「使用 Optuna 優化」來啟用超參數優化功能")
    else:
        st.success(f"✅ Optuna 優化已啟用！已完成 {n_trials} 次試驗")
        
        st.subheader("🏆 最佳參數")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("最佳 Alpha", f"{study.best_params['alpha']:.6f}")
        with col2:
            st.metric("最佳 MSE", f"{study.best_value:.2f}")
        with col3:
            st.metric("完成試驗數", len(study.trials))
        
        st.subheader("📈 優化歷史")
        try:
            fig = plot_optimization_history(study)
            fig.set_figwidth(10)
            fig.set_figheight(5)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"無法顯示優化歷史圖: {str(e)}")
            # 手動繪製優化歷史
            trials_values = [trial.value for trial in study.trials]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trials_values, marker='o', linestyle='-', alpha=0.7)
            ax.set_xlabel('試驗次數', fontsize=12)
            ax.set_ylabel('目標值 (MSE)', fontsize=12)
            ax.set_title('優化歷史', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        st.subheader("📊 所有試驗結果")
        trials_df = study.trials_dataframe()
        trials_df = trials_df[['number', 'value', 'params_alpha', 'state']]
        trials_df.columns = ['試驗編號', 'MSE', 'Alpha', '狀態']
        trials_df = trials_df.sort_values('MSE')
        st.dataframe(trials_df.head(20), use_container_width=True)
        
        st.subheader("📊 Alpha 參數分佈")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(trials_df['Alpha'], trials_df['MSE'], alpha=0.6, s=50, edgecolors='k')
        ax.axvline(x=study.best_params['alpha'], color='r', linestyle='--', lw=2, 
                   label=f"最佳 Alpha = {study.best_params['alpha']:.4f}")
        ax.set_xlabel('Alpha 值', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title('Alpha 參數與 MSE 的關係', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("⚖️ 優化效果比較")
        default_model = train_lasso_model(X_train, y_train, 1.0)
        default_metrics = evaluate_model(default_model, X_train, X_test, y_train, y_test)
        
        comparison_df = pd.DataFrame({
            '模型': ['預設 Alpha (1.0)', f'Optuna 最佳 Alpha ({study.best_params["alpha"]:.4f})'],
            '訓練集 R²': [default_metrics['train']['r2'], metrics['train']['r2']],
            '測試集 R²': [default_metrics['test']['r2'], metrics['test']['r2']],
            '測試集 MSE': [default_metrics['test']['mse'], metrics['test']['mse']],
            '測試集 MAE': [default_metrics['test']['mae'], metrics['test']['mae']]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        improvement_r2 = ((metrics['test']['r2'] - default_metrics['test']['r2']) / 
                         abs(default_metrics['test']['r2']) * 100)
        improvement_mse = ((default_metrics['test']['mse'] - metrics['test']['mse']) / 
                          default_metrics['test']['mse'] * 100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² 改善", f"{improvement_r2:+.2f}%")
        with col2:
            st.metric("MSE 改善", f"{improvement_mse:+.2f}%")

# Tab 6: 模型部署
with tab6:
    st.header("🚀 Step 5: 模型部署")
    st.markdown("**CRISP-DM 階段**: Deployment")
    
    deploy_tab1, deploy_tab2 = st.tabs(["🎯 即時預測", "📤 批次預測"])
    
    with deploy_tab1:
        st.subheader("🎯 即時預測工具")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            length1 = st.number_input("Length1", 0.0, 100.0, 25.0, 0.1)
            length2 = st.number_input("Length2", 0.0, 100.0, 27.0, 0.1)
        
        with col2:
            length3 = st.number_input("Length3", 0.0, 100.0, 32.0, 0.1)
            height = st.number_input("Height", 0.0, 50.0, 12.0, 0.1)
        
        with col3:
            width = st.number_input("Width", 0.0, 20.0, 4.5, 0.1)
            species = st.selectbox("Species", df['Species'].unique())
        
        if st.button("🔮 進行預測", type="primary"):
            input_data = pd.DataFrame({
                'Length1': [length1], 'Length2': [length2], 'Length3': [length3],
                'Height': [height], 'Width': [width], 'Species': [species]
            })
            
            input_encoded = pd.get_dummies(input_data, columns=['Species'], drop_first=True)
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[feature_names]
            
            input_scaled = scaler.transform(input_encoded)
            prediction = model.predict(input_scaled)[0]
            
            st.success(f"### 🎯 預測結果: **{prediction:.2f} 克**")
            
            confidence_interval = metrics['test']['mae'] * 1.96
            st.info(f"📊 95% 信賴區間: **{prediction - confidence_interval:.2f} ~ {prediction + confidence_interval:.2f} 克**")
    
    with deploy_tab2:
        st.subheader("📤 批次預測工具")
        uploaded_file = st.file_uploader("上傳 CSV 檔案", type=['csv'])
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🚀 開始批次預測", type="primary"):
                has_weight = 'Weight' in batch_df.columns
                X_batch = batch_df.drop('Weight', axis=1) if has_weight else batch_df.copy()
                
                if 'Species' in X_batch.columns:
                    X_batch_encoded = pd.get_dummies(X_batch, columns=['Species'], drop_first=True)
                else:
                    X_batch_encoded = X_batch.copy()
                
                for col in feature_names:
                    if col not in X_batch_encoded.columns:
                        X_batch_encoded[col] = 0
                
                X_batch_encoded = X_batch_encoded[feature_names]
                X_batch_scaled = scaler.transform(X_batch_encoded)
                predictions = model.predict(X_batch_scaled)
                
                results_df = batch_df.copy()
                results_df['預測體重'] = predictions
                
                st.success(f"✅ 批次預測完成！共 {len(predictions)} 筆")
                st.dataframe(results_df, use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下載結果", csv, "predictions.csv", "text/csv", type="primary")

# 頁尾總結
st.markdown("---")
st.header("📝 專案總結")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    ### ✅ 完成項目
    
    - ✅ 載入 {len(df)} 筆魚類資料
    - ✅ 資料預處理與分割
    - ✅ Lasso 迴歸模型 (Alpha={current_alpha:.6f})
    - ✅ 保留 {np.sum(model.coef_ != 0)}/{len(model.coef_)} 個特徵
    - ✅ {'Optuna 超參數優化' if use_optuna else '手動參數設定'}
    """)

with col2:
    st.markdown(f"""
    ### 📊 模型效能
    
    - 訓練集 R² = {metrics['train']['r2']:.4f}
    - 測試集 R² = {metrics['test']['r2']:.4f}
    - 測試集 MAE = {metrics['test']['mae']:.2f} 克
    - 最重要特徵: {coef_df.iloc[0]['特徵名稱']}
    """)

st.balloons()
st.success("🎉 專案完成！感謝使用本系統！")

st.sidebar.markdown("---")
st.sidebar.info("""
### 💡 使用提示

1. 調整參數後模型自動重新訓練
2. 使用 Optuna 找到最佳 Alpha
3. 切換 Tab 查看各步驟
4. 在「模型部署」進行預測
""")