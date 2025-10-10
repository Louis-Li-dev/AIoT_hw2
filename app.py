# app.py
# -*- coding: utf-8 -*-
#
# 房屋價格預測 Streamlit 應用（現代化 UI + 三分頁）
# - 讀取 dataset/train.csv 與 dataset/test.csv（固定路徑）
# - 目標：SalePrice
# - 缺值策略：>50% 缺值之欄位刪除；其餘數值以中位數、類別以眾數
# - 特徵工程：One-Hot（類別）+ 標準化（數值）
# - 特徵選擇：KBest(f_regression) 或 L1/Lasso（自動 α）
# - 模型：NumPy 閉式解 OLS（提供 95% CI/PI），完全不依賴 statsmodels
#
# 執行：
#   streamlit run app.py
#
from pathlib import Path
import io
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import sparse
from scipy.stats import t as student_t


# ──────────────────────────────────────────────────────────────────────────────
# 基本設定
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="房屋價格預測 — 多元線性回歸",
    page_icon="🏠",
    layout="wide"
)

DEFAULT_TARGET = "SalePrice"
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "dataset" / "train.csv"
TEST_PATH  = BASE_DIR / "dataset" / "test.csv"

# 環境健檢（不要求上傳；固定讀取路徑）
if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    st.error("找不到資料檔。請建立 `dataset/` 資料夾並放入 `train.csv` 與 `test.csv`。")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# 資料載入與快取
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csvs():
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)
    return train_raw, test_raw

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame):
    """智慧缺值處理：
    - 缺值比例 > 50%：整欄刪除
    - 其餘：數值→中位數；類別→眾數（若無則 'Unknown'）
    備註：不在此處丟棄 target 欄位
    """
    df = df.copy()
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    df = df.drop(columns=cols_to_drop)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                fillv = mode.iloc[0] if not mode.empty else "Unknown"
                df[col] = df[col].fillna(fillv)
    return df, cols_to_drop

def _make_ohe():
    """為了兼容不同版本 scikit-learn 的 OneHotEncoder 參數差異。"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # 舊版 API
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def prepare_features(df: pd.DataFrame, target: str):
    """分離特徵與目標、建立 ColumnTransformer。"""
    y = df[target].astype(float)
    X = df.drop(columns=[target])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", _make_ohe(), cat_cols),
        ],
        remainder="drop",
    )
    return X, y, pre, num_cols, cat_cols

def fig_to_png(fig):
    """Matplotlib 圖轉 PNG bytes。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# OLS（NumPy 閉式解）計算 95% CI / 95% PI
# ──────────────────────────────────────────────────────────────────────────────
def ols_ci_pi_numpy(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, alpha: float = 0.05):
    """
    以 NumPy（閉式解）計算 OLS 參數、均值預測、95% 信賴區間（CI）與預測區間（PI）。
    X_* 需為「已完成所有前處理/特徵選擇」後的設計矩陣（不含常數欄）。
    """
    # 加上常數項
    Xtr = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    Xte = np.c_[np.ones((X_test.shape[0], 1)),  X_test]

    # β̂ = (X'X)^(-1) X'y ；使用 pinv 比 inv 穩定
    XtX_inv = np.linalg.pinv(Xtr.T @ Xtr)
    beta_hat = XtX_inv @ (Xtr.T @ y_train)

    # 殘差與殘差方差 s^2 = RSS / (n - p)
    resid = y_train - Xtr @ beta_hat
    n, p = Xtr.shape
    dof = max(n - p, 1)
    sigma2 = float((resid @ resid) / dof)

    # 預測均值與標準誤
    y_mean = Xte @ beta_hat
    # Var(mean) = s^2 * x0' (X'X)^(-1) x0
    mean_var = np.einsum("ij,jk,ik->i", Xte, XtX_inv, Xte)
    se_mean = np.sqrt(np.maximum(sigma2 * mean_var, 0.0))
    # 預測區間 Var(pred) = s^2 * (1 + x0'(X'X)^(-1)x0)
    se_pred = np.sqrt(np.maximum(sigma2 * (1.0 + mean_var), 0.0))

    # t 臨界值
    tcrit = float(student_t.ppf(1.0 - alpha / 2.0, dof))

    ci_lo = y_mean - tcrit * se_mean
    ci_hi = y_mean + tcrit * se_mean
    pi_lo = y_mean - tcrit * se_pred
    pi_hi = y_mean + tcrit * se_pred

    return {
        "beta": beta_hat,          # [intercept, coef...]
        "y_mean": y_mean,          # 預測均值
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "pi_lo": pi_lo, "pi_hi": pi_hi,
        "sigma2": sigma2, "dof": dof
    }


# ──────────────────────────────────────────────────────────────────────────────
# 訓練與評估主流程（可呼叫）
# ──────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(df_clean: pd.DataFrame, *,
                       test_size: float,
                       seed: int,
                       feat_sel: str,
                       k: int,
                       target: str = DEFAULT_TARGET):
    if target not in df_clean.columns:
        raise ValueError(f"目標欄位「{target}」不存在於資料集中。")

    X, y, pre, num_cols, cat_cols = prepare_features(df_clean, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)

    # fit/transform
    pre.fit(Xtr)
    Xtr_t = pre.transform(Xtr)
    Xte_t = pre.transform(Xte)

    # 回推特徵名稱（數值 + OHE 後）
    feat_names = []
    if num_cols:
        feat_names += num_cols
    if cat_cols:
        ohe = pre.named_transformers_["cat"]
        try:
            feat_names += list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            feat_names += [f"{c}" for c in cat_cols]

    # 稠密化
    Xtr_d = Xtr_t.toarray() if sparse.issparse(Xtr_t) else np.asarray(Xtr_t)
    Xte_d = Xte_t.toarray() if sparse.issparse(Xte_t) else np.asarray(Xte_t)

    # 特徵選擇
    selected_idx = np.arange(Xtr_d.shape[1])
    feat_sel_desc = "不進行特徵選擇"

    if feat_sel == "kbest":
        k_use = max(1, min(int(k), Xtr_d.shape[1]))
        skb = SelectKBest(score_func=f_regression, k=k_use)
        skb.fit(Xtr_d, ytr.values)
        selected_idx = np.where(skb.get_support())[0]
        feat_sel_desc = f"KBest (K={k_use})"

    elif feat_sel == "lasso":
        lasso = LassoCV(cv=5, random_state=seed, n_alphas=100, max_iter=5000)
        lasso.fit(Xtr_d, ytr.values)
        nz = np.where(np.abs(lasso.coef_) > 1e-8)[0]
        if len(nz) == 0:
            # 萬一全 0，保底取 |coef| 最大的 10 個
            k_fb = max(1, min(10, Xtr_d.shape[1]))
            nz = np.argsort(np.abs(lasso.coef_))[-k_fb:]
        selected_idx = np.sort(nz)
        feat_sel_desc = f"L1/Lasso（{len(selected_idx)} features）"

    # 篩選後資料
    Xtr_s = Xtr_d[:, selected_idx]
    Xte_s = Xte_d[:, selected_idx]
    feat_names_s = [feat_names[i] if i < len(feat_names) else f"f_{i}" for i in selected_idx]

    # ── OLS（NumPy 閉式解）與區間計算 ─────────────────────────────
    ols_out = ols_ci_pi_numpy(Xtr_s, ytr.values, Xte_s, alpha=0.05)
    yhat  = ols_out["y_mean"]
    ci_lo = ols_out["ci_lo"]; ci_hi = ols_out["ci_hi"]
    pi_lo = ols_out["pi_lo"]; pi_hi = ols_out["pi_hi"]

    # 指標
    mae = mean_absolute_error(yte.values, yhat)
    rmse = mean_squared_error(yte.values, yhat, squared=False)
    r2 = r2_score(yte.values, yhat)

    # 係數（含截距在 beta[0]）
    beta = np.asarray(ols_out["beta"]).reshape(-1)
    intercept = float(beta[0]) if beta.size > 0 else 0.0
    coefs = np.asarray(beta[1:], dtype=float)

    # Top-10 係數
    names = feat_names_s if len(feat_names_s) == len(coefs) else [f"feature_{i}" for i in range(len(coefs))]
    if len(coefs) > 0 and np.any(np.isfinite(coefs)):
        order = np.argsort(-np.abs(coefs))
        top = [(i + 1, names[idx], float(coefs[idx])) for i, idx in enumerate(order[: min(10, len(coefs))])]
    else:
        top = [(1, "N/A", 0.0)]

    # 視覺化：Predicted vs Actual（排序以便觀察趨勢）
    order_idx = np.argsort(yte.values)
    y_true = yte.values[order_idx]
    y_pred = yhat[order_idx]
    ci_lo_s, ci_hi_s = ci_lo[order_idx], ci_hi[order_idx]
    pi_lo_s, pi_hi_s = pi_lo[order_idx], pi_hi[order_idx]

    fig = plt.figure(figsize=(8.5, 6.2))
    ax = fig.add_subplot(111)
    ax.fill_between(range(len(y_true)), pi_lo_s, pi_hi_s, alpha=0.12, label="95% Prediction Interval")
    ax.fill_between(range(len(y_true)), ci_lo_s, ci_hi_s, alpha=0.20, label="95% Confidence Interval")
    ax.plot(range(len(y_true)), y_true, lw=2, label="Actual", marker="o", markersize=3, alpha=0.75)
    ax.plot(range(len(y_true)), y_pred, lw=2, label="Predicted", marker="s", markersize=3, alpha=0.75)
    ax.set_title("Predicted vs Actual (with 95% CI / PI)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Validation Samples (sorted by Actual)")
    ax.set_ylabel("SalePrice")
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    fig.tight_layout()

    return {
        "n_train": int(Xtr_s.shape[0]),
        "n_test": int(Xte_s.shape[0]),
        "n_features": int(Xtr_s.shape[1]),
        "target": target,
        "feat_sel_desc": feat_sel_desc,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "top_coefs": top,
        "figure": fig,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 頁面 UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>房屋價格預測</h1>"
    "<div style='color:#64748b;margin-top:4px'>Multiple Linear Regression with Advanced Feature Selection</div>",
    unsafe_allow_html=True
)
st.markdown("---")

# 載入與清理
train_raw, test_raw = load_csvs()
train_clean, dropped_cols = clean_data(train_raw.copy())
test_clean, _ = clean_data(test_raw.copy())

# 快速統計
train_n_rows, n_cols = train_clean.shape
test_n_rows = test_clean.shape[0]

# Tabs
tab_preview, tab_train, tab_results = st.tabs(["📄 資料預覽", "⚙️ 訓練設定", "📈 訓練結果"])

with tab_preview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("訓練樣本", f"{train_n_rows:,}")
    c2.metric("測試樣本", f"{test_n_rows:,}")
    c3.metric("特徵欄位數", f"{n_cols:,}")
    c4.metric("預測目標", DEFAULT_TARGET)

    st.success("已完成缺值清理；>50% 缺值欄位已刪除並記錄。")
    st.caption(f"刪除欄位數（>50% 缺值）：{len(dropped_cols)}")
    with st.expander("查看被刪欄位名稱", expanded=False):
        if len(dropped_cols) == 0:
            st.write("（無）")
        else:
            st.write(dropped_cols)

    st.subheader("訓練資料前 10 列")
    st.dataframe(train_clean.head(10), use_container_width=True)

with tab_train:
    st.subheader("訓練配置")
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        test_size = st.selectbox("驗證集比例", options=[0.2, 0.25, 0.3], index=0, format_func=lambda x: f"{int(x*100)}%")
    with colB:
        seed = st.number_input("隨機種子", min_value=1, value=42, step=1)
    with colC:
        feat_sel = st.selectbox("特徵選擇方法", options=["kbest", "lasso"], index=0,
                                format_func=lambda v: "KBest (f_regression)" if v == "kbest" else "L1 / Lasso（自動 α）")
    with colD:
        k = st.number_input("K 值（KBest）", min_value=1, value=10, step=1)

    start_btn = st.button("開始訓練", type="primary")

    if start_btn:
        # 假進度條：增進體驗（不影響實際運算）
        progress = st.progress(0, text="訓練中...請稍候")
        for p in [10, 25, 45, 65, 80, 90]:
            progress.progress(p, text="訓練中...請稍候")
            time.sleep(0.15)

        try:
            results = train_and_evaluate(
                train_clean,
                test_size=float(test_size),
                seed=int(seed),
                feat_sel=feat_sel,
                k=int(k),
                target=DEFAULT_TARGET
            )
            # 訓練結束
            progress.progress(100, text="完成！")
            st.success("✅ 訓練完成！請切換到「📈 訓練結果」分頁查看。")
            # 儲存於 session_state 以便結果頁顯示
            st.session_state["results"] = results
        except Exception as e:
            progress.empty()
            st.error(f"❌ 訓練失敗：{e}")

with tab_results:
    results = st.session_state.get("results")
    if not results:
        st.info("尚無結果。請到「⚙️ 訓練設定」分頁執行訓練。")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("訓練樣本", f"{results['n_train']:,}")
        c2.metric("驗證樣本", f"{results['n_test']:,}")
        c3.metric("使用特徵數", f"{results['n_features']:,}")
        c4.metric("MAE", f"{results['mae']:,.2f}")
        c5.metric("RMSE", f"{results['rmse']:,.2f}")
        st.metric(label="R²", value=f"{results['r2']:.4f}")
        st.caption(f"特徵選擇：{results['feat_sel_desc']}｜目標：{results['target']}")

        # Top-10 係數
        st.subheader("特徵係數（Top 10 by |coef|）")
        coef_df = pd.DataFrame(results["top_coefs"], columns=["Rank", "Feature", "Coefficient"])
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # 圖
        st.subheader("Predicted vs Actual（含 95% CI / 95% PI）")
        st.pyplot(results["figure"], clear_figure=True)

        st.caption("CI = Confidence Interval（平均預測區間）；PI = Prediction Interval（新觀測區間）")
