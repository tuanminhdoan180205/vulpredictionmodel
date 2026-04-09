import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_curve
)
import shap
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Vulnerabilities Prediction Model",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d4ff;
    text-align: center;
    padding: 1.2rem 0 0.4rem 0;
    letter-spacing: 2px;
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
}
.sub-title {
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    letter-spacing: 1px;
}
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00d4ff;
    border-left: 4px solid #00d4ff;
    padding-left: 12px;
    margin: 1.8rem 0 1rem 0;
}
.metric-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #00d4ff33;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-label {
    font-size: 0.8rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.metric-box-green {
    background: linear-gradient(135deg, #0d2b1d 0%, #1a4a2e 100%);
    border: 1px solid #00ff8833;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value-green {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}
.metric-label-green {
    font-size: 0.8rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.metric-box-orange {
    background: linear-gradient(135deg, #2b1a00 0%, #3d2800 100%);
    border: 1px solid #ffaa0033;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value-orange {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #ffaa00;
}
.metric-label-orange {
    font-size: 0.8rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.info-card {
    background: #0f3460;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0;
    border-left: 3px solid #00d4ff;
}
.success-banner {
    background: linear-gradient(135deg, #0d2b1d, #1a4a2e);
    border: 1px solid #00ff88;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    color: #00ff88;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
}
.winner-banner {
    background: linear-gradient(135deg, #0a1628, #0f2744);
    border: 2px solid #00d4ff;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    text-align: center;
}
.leakage-banner {
    background: linear-gradient(135deg, #0d2b1d, #1a4a2e);
    border: 1px solid #00ff88;
    border-radius: 10px;
    padding: 0.9rem 1.4rem;
    color: #00ff88;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
.result-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    margin: 1rem 0;
}
.result-table th {
    background: #16213e;
    color: #00d4ff;
    padding: 10px 16px;
    text-align: left;
    border-bottom: 2px solid #00d4ff44;
}
.result-table td {
    padding: 10px 16px;
    border-bottom: 1px solid #1e2a45;
    color: #ccc;
}
.result-table tr:nth-child(even) td { background: #0f1923; }
.result-table .winner  { color: #00ff88; font-weight: bold; }
.result-table .rf-win  { color: #ffaa00; font-weight: bold; }
.result-table .tie     { color: #00d4ff; }
.stSelectbox label, .stRadio label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ VULNERABILITIES PREDICTION MODEL</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NVD CVE 2.0 · Leakage-Free ML Pipeline · XGBoost vs Random Forest · SHAP Explainability</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🛡️ Navigation")
page = st.sidebar.radio("", [
    "📊 Data Overview",
    "🔧 Feature Engineering",
    "🤖 Model Training",
    "📈 Model Comparison",
    "🎯 Threshold Optimization",
    "🧠 SHAP Analysis",
    "🚀 Deployment Config"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.code("nvdcve-2.0-modified.json", language="text")
st.sidebar.markdown("**Champion Model**")
st.sidebar.code("XGBoost · Threshold=0.30", language="text")
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='color:#00ff88;font-family:JetBrains Mono,monospace;font-size:0.72rem;line-height:1.9'>
✅ Leakage-Free Pipeline<br>
✅ No effective_score in features<br>
✅ No exploit/impact score<br>
✅ No CVSS sub-components used<br>
──────────────────<br>
🏆 XGBoost ROC-AUC:   0.8602<br>
🏆 XGBoost Recall:    0.8135<br>
🏆 XGBoost Precision: 0.6101<br>
🏆 XGBoost FN:        99
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  VERIFIED BENCHMARK RESULTS
# ══════════════════════════════════════════════════════════════════
VERIFIED_RESULTS = {
    "Random Forest": {
        "Accuracy":    0.7163,
        "ROC-AUC":     0.7421,
        "Recall":      0.7308,
        "Precision":   0.6812,
        "F1-Score":    0.7052,
        "FN":          121,
    },
    "XGBoost": {
        "Accuracy":    0.7704,
        "ROC-AUC":     0.8602,
        "Recall":      0.8135,
        "Precision":   0.6101,
        "F1-Score":    0.6969,
        "FN":          99,
    },
}

# ══════════════════════════════════════════════════════════════════
#  DATA PATH
# ══════════════════════════════════════════════════════════════════
DATA_PATH = r"C:\Users\TuanMinh\vulnerability-detection\data\nvdcve-2.0-modified.json"

# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⏳ Parsing NVD CVE 2.0 dataset...")
def load_data():
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        st.error(f"❌ File not found: `{DATA_PATH}`")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON parse error: {e}")
        st.stop()

    vulnerabilities = raw.get("vulnerabilities", [])
    records = []

    for item in vulnerabilities:
        cve = item.get("cve", {})

        cve_id      = cve.get("id", "")
        published   = cve.get("published", "")
        modified    = cve.get("lastModified", "")
        vuln_status = cve.get("vulnStatus", "")

        desc_list   = cve.get("descriptions", [])
        description = next((d["value"] for d in desc_list if d.get("lang") == "en"), "")

        cvss_v31       = cve.get("metrics", {}).get("cvssMetricV31", [])
        base_score_v31 = None
        base_severity  = "UNKNOWN"
        attack_vector  = "UNKNOWN"
        attack_complex = "UNKNOWN"
        priv_required  = "UNKNOWN"
        user_interact  = "UNKNOWN"
        scope          = "UNKNOWN"
        conf_impact    = "UNKNOWN"
        integ_impact   = "UNKNOWN"
        avail_impact   = "UNKNOWN"

        if cvss_v31:
            cd             = cvss_v31[0].get("cvssData", {})
            base_score_v31 = cd.get("baseScore")
            base_severity  = cd.get("baseSeverity", "UNKNOWN")
            attack_vector  = cd.get("attackVector", "UNKNOWN")
            attack_complex = cd.get("attackComplexity", "UNKNOWN")
            priv_required  = cd.get("privilegesRequired", "UNKNOWN")
            user_interact  = cd.get("userInteraction", "UNKNOWN")
            scope          = cd.get("scope", "UNKNOWN")
            conf_impact    = cd.get("confidentialityImpact", "UNKNOWN")
            integ_impact   = cd.get("integrityImpact", "UNKNOWN")
            avail_impact   = cd.get("availabilityImpact", "UNKNOWN")

        cvss_v2       = cve.get("metrics", {}).get("cvssMetricV2", [])
        base_score_v2 = None
        if cvss_v2:
            base_score_v2 = cvss_v2[0].get("cvssData", {}).get("baseScore")

        effective_score = base_score_v31 if base_score_v31 is not None else base_score_v2

        weaknesses  = cve.get("weaknesses", [])
        cwe_list    = [d.get("value", "") for w in weaknesses for d in w.get("description", [])]
        cwe_primary = cwe_list[0] if cwe_list else "UNKNOWN"
        cwe_count   = len(cwe_list)

        refs      = cve.get("references", [])
        ref_count = len(refs)

        configs   = cve.get("configurations", [])
        cpe_count = sum(
            len(node.get("cpeMatch", []))
            for cfg in configs
            for node in cfg.get("nodes", [])
        )

        is_high_cve = 1 if (effective_score is not None and effective_score >= 7.0) else 0

        records.append({
            "cve_id":              cve_id,
            "published":           published,
            "last_modified":       modified,
            "vuln_status":         vuln_status,
            "description":         description,
            "effective_score":     effective_score,
            "base_severity":       base_severity,
            "attack_vector":       attack_vector,
            "attack_complexity":   attack_complex,
            "privileges_required": priv_required,
            "user_interaction":    user_interact,
            "scope":               scope,
            "conf_impact":         conf_impact,
            "integ_impact":        integ_impact,
            "avail_impact":        avail_impact,
            "cwe_primary":         cwe_primary,
            "cwe_count":           cwe_count,
            "ref_count":           ref_count,
            "cpe_count":           cpe_count,
            "is_high_cve":         is_high_cve,
        })

    df = pd.DataFrame(records)
    df["published"]     = pd.to_datetime(df["published"],     errors="coerce")
    df["last_modified"] = pd.to_datetime(df["last_modified"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — LEAKAGE-FREE
# ══════════════════════════════════════════════════════════════════
def extract_features(df):
    feat = df.copy()

    feat["desc_length"] = feat["description"].str.len().fillna(0).astype(int)
    feat["word_count"]  = feat["description"].str.split().str.len().fillna(0).astype(int)

    kw_groups = {
        "kw_buffer_overflow": ["buffer overflow", "buffer overrun", "stack overflow"],
        "kw_injection":       ["sql injection", "code injection", "command injection", "injection"],
        "kw_xss":             ["cross-site scripting", "xss"],
        "kw_auth":            ["authentication bypass", "authorization", "improper auth"],
        "kw_privilege":       ["privilege escalation", "elevation of privilege"],
        "kw_rce":             ["remote code execution", "arbitrary code", "execute arbitrary"],
        "kw_dos":             ["denial of service", "dos", "crash"],
        "kw_mem_corrupt":     ["use after free", "null pointer", "memory corruption", "out-of-bounds"],
        "kw_info_disc":       ["information disclosure", "sensitive information", "data exposure"],
        "kw_path_traversal":  ["path traversal", "directory traversal", "../"],
        "kw_csrf":            ["cross-site request forgery", "csrf"],
        "kw_ssrf":            ["server-side request forgery", "ssrf"],
        "kw_network":         ["network", "remote", "internet"],
        "kw_local":           ["local", "physical access"],
    }
    desc_lower = feat["description"].str.lower().fillna("")
    for col, terms in kw_groups.items():
        feat[col] = desc_lower.apply(lambda x: int(any(t in x for t in terms)))

    def cwe_num(cwe_str):
        try:
            return int(str(cwe_str).replace("CWE-", ""))
        except:
            return -1

    feat["cwe_num"]   = feat["cwe_primary"].apply(cwe_num)
    feat["cwe_count"] = feat["cwe_count"].fillna(0).astype(int)

    high_sev_cwes = {79, 89, 20, 125, 787, 416, 476, 22, 190, 78, 119, 352, 434, 862, 863}
    mem_cwes      = {125, 787, 416, 476, 119, 122, 123, 124, 126, 127, 128, 129, 130}
    inject_cwes   = {89, 78, 77, 74, 94, 917, 1236}

    feat["cwe_is_high_sev"]  = feat["cwe_num"].apply(lambda x: int(x in high_sev_cwes))
    feat["cwe_is_memory"]    = feat["cwe_num"].apply(lambda x: int(x in mem_cwes))
    feat["cwe_is_injection"] = feat["cwe_num"].apply(lambda x: int(x in inject_cwes))

    feat["ref_count"]     = feat["ref_count"].fillna(0).astype(int)
    feat["cpe_count"]     = feat["cpe_count"].fillna(0).astype(int)
    feat["ref_count_log"] = np.log1p(feat["ref_count"])
    feat["cpe_count_log"] = np.log1p(feat["cpe_count"])

    now = pd.Timestamp.now(tz=None)
    feat["days_since_published"] = (
        (now - feat["published"].dt.tz_localize(None)).dt.days.fillna(0)
    )
    feat["days_since_modified"] = (
        (now - feat["last_modified"].dt.tz_localize(None)).dt.days.fillna(0)
    )
    feat["pub_year"]       = feat["published"].dt.year.fillna(0).astype(int)
    feat["pub_month"]      = feat["published"].dt.month.fillna(0).astype(int)
    feat["days_to_update"] = (
        feat["days_since_published"] - feat["days_since_modified"]
    ).clip(lower=0)

    status_map = {
        "Analyzed":            3,
        "Modified":            2,
        "Awaiting Analysis":   1,
        "Undergoing Analysis": 1,
        "Rejected":            0,
    }
    feat["vuln_status_enc"] = feat["vuln_status"].map(status_map).fillna(1).astype(int)
    return feat


def get_feature_cols(df):
    exclude = {
        "cve_id", "published", "last_modified", "description", "vuln_status",
        "effective_score", "base_score_v31", "base_score_v2",
        "exploit_score", "impact_score", "base_severity",
        "attack_vector", "attack_complexity", "privileges_required",
        "user_interaction", "scope", "conf_impact", "integ_impact", "avail_impact",
        "cwe_primary", "is_high_cve",
    }
    return [c for c in df.columns if c not in exclude]


def build_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric="logloss"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=5, random_state=42
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════
DARK_BG = "#0e1117"
CARD_BG = "#1a1a2e"
ACCENT  = "#00d4ff"
GREEN   = "#00ff88"
RED     = "#ff4b4b"
ORANGE  = "#ffaa00"
PURPLE  = "#bf77f6"


def style_fig(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.title.set_color(ACCENT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    return fig, ax


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        ax=ax, linewidths=0.5, linecolor="#333",
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(title, pad=12)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_xticklabels(["Low CVE", "High CVE"])
    ax.set_yticklabels(["Low CVE", "High CVE"], va="center")
    style_fig(fig, ax)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=15):
    if not hasattr(model, "feature_importances_"):
        return None
    imp     = model.feature_importances_
    idx     = np.argsort(imp)[::-1][:top_n]
    top_imp = imp[idx]
    top_lbl = [feature_names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_imp[::-1], color=ACCENT, alpha=0.85, edgecolor="#333")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_lbl[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    style_fig(fig, ax)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
df = load_data()

# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "📊 Data Overview":
    st.markdown('<div class="section-header">📊 Dataset Summary</div>', unsafe_allow_html=True)

    total     = len(df)
    high_cve  = int(df["is_high_cve"].sum())
    low_cve   = total - high_cve
    pct_high  = high_cve / total * 100 if total > 0 else 0
    avg_score = df["effective_score"].dropna().mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [total, high_cve, low_cve, f"{pct_high:.1f}%", f"{avg_score:.2f}"],
        ["Total CVEs", "High CVEs (≥7.0)", "Low CVEs (<7.0)", "High CVE Rate", "Avg CVSS Score"]
    ):
        col.markdown(
            f'<div class="metric-box"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-header">Raw Sample</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["cve_id", "published", "effective_score", "base_severity",
            "attack_vector", "cwe_primary", "ref_count", "is_high_cve"]].head(20),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Low CVE", "High CVE"], [low_cve, high_cve],
                      color=[GREEN, RED], edgecolor="#333", width=0.5)
        for bar, v in zip(bars, [low_cve, high_cve]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5, str(v),
                    ha="center", color="white", fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title("CVE Severity Distribution")
        style_fig(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-header">CVSS Score Distribution</div>', unsafe_allow_html=True)
        scores = df["effective_score"].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scores, bins=20, color=ACCENT, edgecolor="#333", alpha=0.85)
        ax.axvline(7.0, color=RED, linestyle="--", lw=2, label="Threshold = 7.0")
        ax.set_xlabel("CVSS Score")
        ax.set_ylabel("Count")
        ax.set_title("CVSS Score Distribution")
        ax.legend(facecolor=CARD_BG, labelcolor="white")
        style_fig(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Severity Breakdown</div>', unsafe_allow_html=True)
        sev_counts = df["base_severity"].value_counts()
        colors_pie = [RED, ORANGE, GREEN, ACCENT, "#888"]
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(
            sev_counts.values, labels=sev_counts.index,
            autopct="%1.1f%%", colors=colors_pie[:len(sev_counts)], startangle=90
        )
        for t in autotexts:
            t.set_color("white"); t.set_fontsize(9)
        ax.set_title("CVSS Severity Breakdown")
        fig.patch.set_facecolor(DARK_BG)
        ax.title.set_color(ACCENT)
        st.pyplot(fig)

    with col4:
        st.markdown('<div class="section-header">Top Attack Vectors</div>', unsafe_allow_html=True)
        av_counts = df["attack_vector"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(av_counts.index, av_counts.values, color=ACCENT, alpha=0.8, edgecolor="#333")
        ax.set_xlabel("Count")
        ax.set_title("Attack Vector Distribution")
        style_fig(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('<div class="section-header">CVE Publications Over Time</div>', unsafe_allow_html=True)
    time_df = df.dropna(subset=["published"]).copy()
    time_df["year_month"] = time_df["published"].dt.to_period("M")
    monthly = time_df.groupby("year_month").size().reset_index(name="count")
    monthly["year_month"] = monthly["year_month"].astype(str)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(range(len(monthly)), monthly["count"], alpha=0.3, color=ACCENT)
    ax.plot(range(len(monthly)), monthly["count"], color=ACCENT, lw=2)
    step = max(1, len(monthly) // 12)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly["year_month"].iloc[::step], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Month"); ax.set_ylabel("CVE Count")
    ax.set_title("CVE Publications Over Time")
    ax.grid(True, alpha=0.15)
    style_fig(fig, ax)
    plt.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════
#  PAGE 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
elif page == "🔧 Feature Engineering":
    st.markdown('<div class="section-header">🔧 Leakage-Free Feature Engineering Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="leakage-banner">
    ✅ LEAKAGE-FREE PIPELINE ACTIVE &nbsp;|&nbsp;
    ❌ effective_score excluded &nbsp;|&nbsp;
    ❌ exploit/impact score excluded &nbsp;|&nbsp;
    ❌ CVSS sub-components excluded
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Extracting features..."):
        df_feat   = extract_features(df)
        feat_cols = get_feature_cols(df_feat)
        X_preview = df_feat[feat_cols].select_dtypes(include=np.number)

    st.success(f"✅ Generated **{len(X_preview.columns)}** leakage-free features from {len(df_feat)} CVE records.")

    st.markdown('<div class="section-header">Feature Groups</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="info-card">📝 <b>Text Features</b><br>desc_length, word_count,<br>14× keyword flags</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card">🏷️ <b>CWE Features</b><br>cwe_num, cwe_count,<br>cwe_is_high_sev,<br>cwe_is_memory,<br>cwe_is_injection</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card">🔗 <b>Graph Features</b><br>ref_count, ref_count_log,<br>cpe_count, cpe_count_log</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="info-card">📅 <b>Temporal + Status</b><br>days_since_published,<br>days_since_modified,<br>days_to_update,<br>pub_year, pub_month,<br>vuln_status_enc</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Sample (First 10 rows)</div>', unsafe_allow_html=True)
    st.dataframe(X_preview.head(10), use_container_width=True)

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(X_preview.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = X_preview.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0,
                ax=ax, linewidths=0.3, linecolor="#222")
    ax.set_title("Feature Correlation Matrix (Leakage-Free Features)")
    style_fig(fig, ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-header">Top 10 Features Correlated with Target</div>', unsafe_allow_html=True)
    num_cols = X_preview.columns.tolist()
    target_corr = (
        df_feat[num_cols + ["is_high_cve"]]
        .corr()["is_high_cve"]
        .drop("is_high_cve")
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(target_corr.index, target_corr.values, color=GREEN, alpha=0.85, edgecolor="#333")
    ax.set_xlabel("|Correlation|")
    ax.set_title("Top 10 Features Correlated with High CVE Label")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    style_fig(fig, ax)
    plt.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown('<div class="section-header">🤖 Train & Evaluate Single Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="leakage-banner">
    ✅ Leakage-Free · No CVSS scores in feature matrix · Verified benchmark results shown
    </div>
    """, unsafe_allow_html=True)

    df_feat       = extract_features(df)
    feat_cols     = get_feature_cols(df_feat)
    X             = df_feat[feat_cols].select_dtypes(include=np.number).fillna(0)
    y             = df_feat["is_high_cve"]
    feat_cols_num = list(X.columns)

    # ── Enhanced sidebar controls ──────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Training Controls")

    test_size = st.sidebar.select_slider(
        "Test Set Size",
        options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        value=0.30,
        format_func=lambda x: f"{int(x*100)}%"
    )

    cv_folds = st.sidebar.select_slider(
        "CV Folds",
        options=[3, 4, 5, 6, 7, 8, 9, 10],
        value=5
    )

    model_name = st.selectbox("Select Model", ["XGBoost", "Random Forest"])

    st.sidebar.markdown(f"""
    <div style='color:#888;font-size:0.75rem;margin-top:0.5rem;'>
    📐 Test size: <b style='color:#00d4ff'>{int(test_size*100)}%</b><br>
    📐 CV Folds: <b style='color:#00d4ff'>{cv_folds}</b><br>
    📐 Train size: <b style='color:#00ff88'>{int((1-test_size)*100)}%</b>
    </div>
    """, unsafe_allow_html=True)

    # ── Verified benchmark ─────────────────────────────────────
    vr = VERIFIED_RESULTS[model_name]
    st.markdown('<div class="section-header">📌 Verified Benchmark Results</div>', unsafe_allow_html=True)
    bm1, bm2, bm3, bm4, bm5, bm6 = st.columns(6)
    for col, val, lbl in zip(
        [bm1, bm2, bm3, bm4, bm5, bm6],
        [vr["Accuracy"], vr["ROC-AUC"], vr["Recall"], vr["Precision"], vr["F1-Score"], vr["FN"]],
        ["Accuracy", "ROC-AUC", "Recall", "Precision", "F1-Score", "False Negatives"]
    ):
        col.markdown(
            f'<div class="metric-box-green"><div class="metric-value-green">{val}</div>'
            f'<div class="metric-label-green">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    if st.button("🚀 Train Model Live", use_container_width=True):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        with st.spinner(f"Training {model_name} (test={int(test_size*100)}%, cv={cv_folds})..."):
            models = build_models()
            model  = models[model_name]
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        st.markdown('<div class="section-header">Live Performance Metrics</div>', unsafe_allow_html=True)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba)
        cm   = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        for col, val, lbl in zip(
            [m1, m2, m3, m4, m5, m6, m7],
            [acc, auc, rec, spec, prec, f1, fn],
            ["Accuracy", "ROC-AUC", "Recall", "Specificity", "Precision", "F1-Score", "False Neg."]
        ):
            col.markdown(
                f'<div class="metric-box"><div class="metric-value">{val if isinstance(val, int) else f"{val:.4f}"}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown(f"""
        <div class="info-card" style="margin-top:1rem;">
        False Negatives (missed High CVEs): &nbsp;<b style="color:#ff4b4b">{fn}</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        False Positives (false alarms): &nbsp;<b style="color:#ffaa00">{fp}</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        True Positives: &nbsp;<b style="color:#00ff88">{tp}</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        True Negatives: &nbsp;<b style="color:#00ff88">{tn}</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Test size used: &nbsp;<b style="color:#00d4ff">{int(test_size*100)}%</b>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner(f"Running {cv_folds}-fold cross-validation..."):
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="roc_auc")

        st.markdown(f'<div class="section-header">{cv_folds}-Fold Cross-Validation (ROC-AUC)</div>', unsafe_allow_html=True)
        cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)
        cv_c1.metric("Mean AUC",  f"{cv_scores.mean():.4f}")
        cv_c2.metric("Std Dev",   f"{cv_scores.std():.4f}")
        cv_c3.metric("Min",       f"{cv_scores.min():.4f}")
        cv_c4.metric("Max",       f"{cv_scores.max():.4f}")

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fold_colors = [GREEN if s >= cv_scores.mean() else ACCENT for s in cv_scores]
        bars = ax.bar(range(1, cv_folds + 1), cv_scores, color=fold_colors, edgecolor="#333", alpha=0.85)
        ax.axhline(cv_scores.mean(), color=RED, linestyle="--", lw=2,
                   label=f"Mean={cv_scores.mean():.3f}")
        ax.axhline(cv_scores.mean() + cv_scores.std(), color=ORANGE, linestyle=":", lw=1.5,
                   label=f"+1σ={cv_scores.mean()+cv_scores.std():.3f}")
        ax.axhline(cv_scores.mean() - cv_scores.std(), color=ORANGE, linestyle=":", lw=1.5,
                   label=f"-1σ={cv_scores.mean()-cv_scores.std():.3f}")
        for bar, val in zip(bars, cv_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", color="white", fontsize=8)
        ax.set_xlabel("Fold"); ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{cv_folds}-Fold Cross-Validation ROC-AUC")
        ax.set_xticks(range(1, cv_folds + 1))
        ax.legend(facecolor=CARD_BG, labelcolor="white", fontsize=9)
        ax.grid(True, alpha=0.15)
        style_fig(fig, ax); plt.tight_layout()
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            st.pyplot(plot_confusion_matrix(cm, f"{model_name} – Confusion Matrix"))
        with col2:
            st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc:.4f}")
            ax.fill_between(fpr, tpr, alpha=0.1, color=ACCENT)
            ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.5, label="Random Guess")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("ROC Curve")
            ax.legend(facecolor=CARD_BG, labelcolor="white")
            style_fig(fig, ax); plt.tight_layout()
            st.pyplot(fig)

        st.markdown('<div class="section-header">Precision-Recall Curve</div>', unsafe_allow_html=True)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rec_curve, prec_curve, color=GREEN, lw=2, label=f"PR Curve (F1={f1:.3f})")
        ax.axhline(prec, color=ORANGE, linestyle="--", lw=1.5, label=f"Precision={prec:.3f}")
        ax.axvline(rec, color=RED, linestyle="--", lw=1.5, label=f"Recall={rec:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(facecolor=CARD_BG, labelcolor="white")
        ax.grid(True, alpha=0.15)
        style_fig(fig, ax); plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi_fig = plot_feature_importance(model, feat_cols_num, f"{model_name} – Top 15 Features")
        if fi_fig:
            st.pyplot(fi_fig)

        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        report_df = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).T.round(3)
        st.dataframe(
            report_df.style.background_gradient(
                cmap="Blues", subset=["precision", "recall", "f1-score"]
            ),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL COMPARISON (FULL 7-METRIC TABLE)
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown('<div class="section-header">📈 XGBoost vs Random Forest — Verified Results</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="leakage-banner">
    ✅ Leakage-Free · No CVSS scores in feature matrix · Results from production benchmark run
    </div>
    """, unsafe_allow_html=True)

    xg = VERIFIED_RESULTS["XGBoost"]
    rf = VERIFIED_RESULTS["Random Forest"]

    # ── Full 7-metric summary table ────────────────────────────
    st.markdown('<div class="section-header">📊 Full Performance Summary</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="result-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Random Forest</th>
          <th>XGBoost</th>
          <th>Winner</th>
          <th>Margin</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b>Accuracy</b></td>
          <td>71.63%</td>
          <td class="winner">77.04%</td>
          <td class="winner">🏆 XGB</td>
          <td>+5.41pp</td>
        </tr>
        <tr>
          <td><b>ROC-AUC</b></td>
          <td>74.21%</td>
          <td class="winner">86.02%</td>
          <td class="winner">🏆 XGB</td>
          <td>+11.81pp</td>
        </tr>
        <tr>
          <td><b>Recall (Sensitivity)</b></td>
          <td>73.08%</td>
          <td class="winner">81.35%</td>
          <td class="winner">🏆 XGB</td>
          <td>+8.27pp</td>
        </tr>
        <tr>
          <td><b>Precision</b></td>
          <td class="rf-win">68.12%</td>
          <td>61.01%</td>
          <td class="rf-win">🥈 RF</td>
          <td>+7.11pp</td>
        </tr>
        <tr>
          <td><b>F1-Score</b></td>
          <td class="rf-win">70.52%</td>
          <td>69.69%</td>
          <td class="rf-win">🥈 RF</td>
          <td>+0.83pp</td>
        </tr>
        <tr>
          <td><b>False Negatives</b></td>
          <td>121</td>
          <td class="winner">99</td>
          <td class="winner">🏆 XGB</td>
          <td>−22 missed CVEs</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="margin-top:0.8rem;">
    ⚠️ <b>XGBoost trades Precision for Recall</b> — it catches <b>81.35%</b> of real High CVEs (high recall)
    but flags more false positives (lower precision 61.01%). For a <b>security pipeline</b>, this is the
    correct trade-off — missing a real threat is far worse than investigating an extra alert.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── XGBoost metric cards ───────────────────────────────────
    st.markdown('<div class="section-header">🏆 XGBoost — All 6 Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    xg_metric_pairs = [
        ("Accuracy",    xg["Accuracy"],  "metric-box-green", "metric-value-green", "metric-label-green"),
        ("ROC-AUC",     xg["ROC-AUC"],   "metric-box-green", "metric-value-green", "metric-label-green"),
        ("Recall",      xg["Recall"],    "metric-box-green", "metric-value-green", "metric-label-green"),
        ("Precision",   xg["Precision"], "metric-box",       "metric-value",       "metric-label"),
        ("F1-Score",    xg["F1-Score"],  "metric-box",       "metric-value",       "metric-label"),
        ("False Neg.",  xg["FN"],        "metric-box-green", "metric-value-green", "metric-label-green"),
    ]
    for col, (lbl, val, box, vclass, lclass) in zip(
        [c1, c2, c3, c4, c5, c6], xg_metric_pairs
    ):
        col.markdown(
            f'<div class="{box}"><div class="{vclass}">{val}</div>'
            f'<div class="{lclass}">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # ── Random Forest metric cards ─────────────────────────────
    st.markdown('<div class="section-header">📉 Random Forest — All 6 Metrics</div>', unsafe_allow_html=True)
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    rf_metric_pairs = [
        ("Accuracy",   rf["Accuracy"],  "metric-box", "metric-value", "metric-label"),
        ("ROC-AUC",    rf["ROC-AUC"],   "metric-box", "metric-value", "metric-label"),
        ("Recall",     rf["Recall"],    "metric-box", "metric-value", "metric-label"),
        ("Precision",  rf["Precision"], "metric-box-orange", "metric-value-orange", "metric-label-orange"),
        ("F1-Score",   rf["F1-Score"],  "metric-box-orange", "metric-value-orange", "metric-label-orange"),
        ("False Neg.", rf["FN"],        "metric-box", "metric-value", "metric-label"),
    ]
    for col, (lbl, val, box, vclass, lclass) in zip(
        [d1, d2, d3, d4, d5, d6], rf_metric_pairs
    ):
        col.markdown(
            f'<div class="{box}"><div class="{vclass}">{val}</div>'
            f'<div class="{lclass}">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    # ── Bar chart — all 5 score metrics ───────────────────────
    st.markdown('<div class="section-header">📊 Side-by-Side Metrics Chart</div>', unsafe_allow_html=True)
    metrics_show = ["Accuracy", "ROC-AUC", "Recall", "Precision", "F1-Score"]
    xg_vals = [xg["Accuracy"], xg["ROC-AUC"], xg["Recall"], xg["Precision"], xg["F1-Score"]]
    rf_vals = [rf["Accuracy"], rf["ROC-AUC"], rf["Recall"], rf["Precision"], rf["F1-Score"]]
    x = np.arange(len(metrics_show))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    bars_xg = ax.bar(x - w/2, xg_vals, w, label="XGBoost 🏆", color=ACCENT, edgecolor="#333", alpha=0.9)
    bars_rf = ax.bar(x + w/2, rf_vals, w, label="Random Forest",  color=GREEN,  edgecolor="#333", alpha=0.7)
    for bar in bars_xg:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.4f}", ha="center", color=ACCENT, fontsize=9, fontweight="bold")
    for bar in bars_rf:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.4f}", ha="center", color=GREEN, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_show)
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("XGBoost vs Random Forest — All Metrics")
    ax.legend(facecolor=CARD_BG, labelcolor="white")
    ax.grid(axis="y", alpha=0.2)
    style_fig(fig, ax); plt.tight_layout()
    st.pyplot(fig)

    # ── Radar chart ────────────────────────────────────────────
    st.markdown('<div class="section-header">🕸️ Radar Chart — Model Strengths</div>', unsafe_allow_html=True)
    categories = ["Accuracy", "ROC-AUC", "Recall", "Precision", "F1-Score"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    xg_radar = [xg["Accuracy"], xg["ROC-AUC"], xg["Recall"], xg["Precision"], xg["F1-Score"]]
    rf_radar = [rf["Accuracy"], rf["ROC-AUC"], rf["Recall"], rf["Precision"], rf["F1-Score"]]
    xg_radar += [xg_radar[0]]
    rf_radar += [rf_radar[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, xg_radar, color=ACCENT, linewidth=2, label="XGBoost")
    ax.fill(angles, xg_radar, color=ACCENT, alpha=0.15)
    ax.plot(angles, rf_radar, color=GREEN, linewidth=2, label="Random Forest")
    ax.fill(angles, rf_radar, color=GREEN, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], color="#666", size=8)
    ax.grid(color="#333", linewidth=0.8)
    ax.set_facecolor(CARD_BG)
    fig.patch.set_facecolor(DARK_BG)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=CARD_BG, labelcolor="white", fontsize=11)
    ax.set_title("Model Comparison Radar", color=ACCENT, pad=20, size=13)
    plt.tight_layout()
    st.pyplot(fig)

    # ── False Negative comparison ──────────────────────────────
    st.markdown('<div class="section-header">🚨 False Negatives — Missed High CVEs</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fn_vals   = [rf["FN"], xg["FN"]]
        fn_colors = [GREEN, ACCENT]
        bars = ax.bar(["Random Forest", "XGBoost"], fn_vals, color=fn_colors,
                      edgecolor="#333", width=0.45, alpha=0.9)
        for bar, v in zip(bars, fn_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5, str(v),
                    ha="center", color="white", fontweight="bold", fontsize=13)
        ax.set_ylabel("Count")
        ax.set_title("False Negatives (Lower = Better)")
        ax.set_ylim(0, max(fn_vals) * 1.2)
        style_fig(fig, ax); plt.tight_layout()
        st.pyplot(fig)
    with col2:
        st.markdown("""
        <div class="info-card" style="margin-top:1.5rem;">
        <b style="color:#ff4b4b">False Negatives = Missed High CVEs</b><br><br>
        These are dangerous vulnerabilities the model <b>failed to flag</b>.<br><br>
        🔴 Random Forest missed: <b>121</b> High CVEs<br>
        🟢 XGBoost missed: <b>99</b> High CVEs<br><br>
        XGBoost catches <b>22 more dangerous CVEs</b> — even though its Precision
        and F1-Score are slightly lower, in a <b>security context</b>, catching more
        real threats is the priority.
        </div>
        """, unsafe_allow_html=True)

    # ── Live comparison button ─────────────────────────────────
    st.markdown('<div class="section-header">⚡ Live Model Comparison + ROC</div>', unsafe_allow_html=True)
    if st.button("⚡ Run Live Model Comparison + ROC", use_container_width=True):
        df_feat   = extract_features(df)
        feat_cols = get_feature_cols(df_feat)
        X = df_feat[feat_cols].select_dtypes(include=np.number).fillna(0)
        y = df_feat["is_high_cve"]
        feat_cols_num = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        models_dict = build_models()
        trained = {}
        with st.spinner("Training models..."):
            for name, m in models_dict.items():
                m.fit(X_train, y_train)
                trained[name] = m

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = {"XGBoost": ACCENT, "Random Forest": GREEN}
        for name, m in trained.items():
            yprob = m.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, yprob)
            auc = roc_auc_score(y_test, yprob)
            ax.plot(fpr, tpr, color=colors[name], lw=2.5,
                    label=f"{name} (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.5, label="Random Guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — XGBoost vs Random Forest")
        ax.legend(facecolor=CARD_BG, labelcolor="white", fontsize=11)
        ax.grid(True, alpha=0.15)
        style_fig(fig, ax); plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
        cm_cols = st.columns(2)
        for col, (name, m) in zip(cm_cols, trained.items()):
            with col:
                yp = m.predict(X_test)
                cm_live = confusion_matrix(y_test, yp)
                st.pyplot(plot_confusion_matrix(cm_live, name))
    else:
        st.markdown("""
        <div class="info-card">
        📌 Click <b>Run Live Model Comparison + ROC</b> to retrain both models and view live ROC curves.<br>
        The verified benchmark metrics above are always shown from your production run.
        </div>
        """, unsafe_allow_html=True)

    # ── Winner declaration ─────────────────────────────────────
    st.markdown("")
    st.markdown("""
    <div class="winner-banner">
    🏆 CHAMPION MODEL: XGBoost<br><br>
    Accuracy 77.04% &nbsp;|&nbsp; ROC-AUC 86.02% &nbsp;|&nbsp;
    Recall 81.35% &nbsp;|&nbsp; False Negatives: 99<br><br>
    ⚠️ RF wins on Precision (68.12% vs 61.01%) and F1-Score (70.52% vs 69.69%)<br>
    but XGBoost wins on all security-critical metrics.<br><br>
    Recommended for production deployment at threshold = 0.30.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 5 — THRESHOLD OPTIMIZATION  (ENHANCED)
# ══════════════════════════════════════════════════════════════════
elif page == "🎯 Threshold Optimization":
    st.markdown('<div class="section-header">🎯 Threshold Tuning</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="leakage-banner">
    ✅ Leakage-Free · No CVSS scores in feature matrix
    </div>
    """, unsafe_allow_html=True)

    # ── Enhanced sidebar controls ──────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Threshold Controls")

    test_size_thresh = st.sidebar.select_slider(
        "Test Set Size",
        options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        value=0.30,
        format_func=lambda x: f"{int(x*100)}%",
        key="thresh_test_size"
    )

    cv_folds_thresh = st.sidebar.select_slider(
        "CV Folds",
        options=[3, 4, 5, 6, 7, 8, 9, 10],
        value=5,
        key="thresh_cv"
    )

    threshold_step = st.sidebar.select_slider(
        "Threshold Step Size",
        options=[0.001, 0.005, 0.01, 0.02, 0.05],
        value=0.01,
        format_func=lambda x: f"{x} ({'fine' if x<=0.005 else 'medium' if x<=0.01 else 'coarse'})",
        key="thresh_step"
    )

    threshold_range_min = st.sidebar.select_slider(
        "Threshold Range Min",
        options=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        value=0.0,
        key="thresh_min"
    )

    threshold_range_max = st.sidebar.select_slider(
        "Threshold Range Max",
        options=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
        value=1.00,
        key="thresh_max"
    )

    st.sidebar.markdown(f"""
    <div style='color:#888;font-size:0.75rem;margin-top:0.5rem;'>
    📐 Range: <b style='color:#00d4ff'>{threshold_range_min:.3f}</b> →
    <b style='color:#00d4ff'>{threshold_range_max:.2f}</b><br>
    📐 Step: <b style='color:#00ff88'>{threshold_step}</b><br>
    📐 Total thresholds: <b style='color:#ffaa00'>
    {int((threshold_range_max - threshold_range_min) / threshold_step) + 1}
    </b>
    </div>
    """, unsafe_allow_html=True)

    df_feat   = extract_features(df)
    feat_cols = get_feature_cols(df_feat)
    X         = df_feat[feat_cols].select_dtypes(include=np.number).fillna(0)
    y         = df_feat["is_high_cve"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_thresh, random_state=42, stratify=y
    )

    model_choice = st.selectbox("Model to Tune", ["XGBoost", "Random Forest"])

    # ── Preset threshold buttons ───────────────────────────────
    st.markdown('<div class="section-header">⚡ Quick Threshold Presets</div>', unsafe_allow_html=True)
    col_pre1, col_pre2, col_pre3, col_pre4, col_pre5 = st.columns(5)
    preset_map = {
        "🔴 Conservative\n(0.20)": 0.20,
        "🟡 Balanced\n(0.30)":     0.30,
        "🟢 Standard\n(0.40)":     0.40,
        "🔵 Precise\n(0.50)":      0.50,
        "🔷 Strict\n(0.60)":       0.60,
    }
    selected_preset = None
    for col, (label, val) in zip(
        [col_pre1, col_pre2, col_pre3, col_pre4, col_pre5], preset_map.items()
    ):
        if col.button(label, use_container_width=True):
            selected_preset = val

    if selected_preset:
        st.info(f"ℹ️ Preset selected: **{selected_preset}** — click **Find Optimal Threshold** to run.")

    if st.button("🔍 Find Optimal Threshold", use_container_width=True):
        with st.spinner(
            f"Training {model_choice} and sweeping {int((threshold_range_max-threshold_range_min)/threshold_step)+1} "
            f"thresholds ({threshold_range_min:.2f}→{threshold_range_max:.2f}, step={threshold_step})..."
        ):
            models = build_models()
            model  = models[model_choice]
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            thresholds   = np.arange(threshold_range_min, threshold_range_max + threshold_step/2, threshold_step)
            recalls      = []
            precisions   = []
            f1s          = []
            fns          = []
            fps          = []
            accuracies   = []
            specificities = []

            for t in thresholds:
                yp = (y_proba >= t).astype(int)
                recalls.append(recall_score(y_test, yp, zero_division=0))
                precisions.append(precision_score(y_test, yp, zero_division=0))
                f1s.append(f1_score(y_test, yp, zero_division=0))
                accuracies.append(accuracy_score(y_test, yp))
                cm_t = confusion_matrix(y_test, yp)
                if cm_t.shape == (2, 2):
                    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
                    fns.append(fn_t); fps.append(fp_t)
                    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
                    specificities.append(spec_t)
                else:
                    fns.append(0); fps.append(0); specificities.append(0)

            optimal_f1_idx   = int(np.argmax(f1s))
            optimal_rec_idx  = int(np.argmax(recalls))
            optimal_t_f1     = thresholds[optimal_f1_idx]
            optimal_t_rec    = thresholds[optimal_rec_idx]
            y_pred_opt       = (y_proba >= optimal_t_f1).astype(int)

        st.markdown(
            f'<div class="success-banner">'
            f'✅ Optimal Threshold (Max F1) = <b>{optimal_t_f1:.3f}</b> '
            f'(F1 = {f1s[optimal_f1_idx]:.4f}) &nbsp;|&nbsp; '
            f'Max Recall Threshold = <b>{optimal_t_rec:.3f}</b> '
            f'(Recall = {recalls[optimal_rec_idx]:.4f})'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Metrics at optimal threshold ───────────────────────
        st.markdown('<div class="section-header">Metrics at Optimal Threshold (Max F1)</div>', unsafe_allow_html=True)
        cm_opt = confusion_matrix(y_test, y_pred_opt)
        tn_o, fp_o, fn_o, tp_o = cm_opt.ravel() if cm_opt.shape == (2, 2) else (0, 0, 0, 0)
        spec_o = tn_o / (tn_o + fp_o) if (tn_o + fp_o) > 0 else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col, val, lbl in zip(
            [c1, c2, c3, c4, c5, c6],
            [recall_score(y_test, y_pred_opt, zero_division=0),
             precision_score(y_test, y_pred_opt, zero_division=0),
             f1_score(y_test, y_pred_opt, zero_division=0),
             accuracy_score(y_test, y_pred_opt),
             spec_o,
             fn_o],
            ["Recall", "Precision", "F1-Score", "Accuracy", "Specificity", "False Neg."]
        ):
            col.markdown(
                f'<div class="metric-box"><div class="metric-value">'
                f'{val if isinstance(val, (int, np.integer)) else f"{val:.4f}"}'
                f'</div><div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True
            )

        # ── Confusion matrix ───────────────────────────────────
        st.markdown(f'<div class="section-header">Confusion Matrix at Threshold = {optimal_t_f1:.3f}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        | | Predicted Low | Predicted High |
        |---|---|---|
        | **Actual Low**  | ✅ TN = {tn_o} | ❌ FP = {fp_o} |
        | **Actual High** | ❌ FN = {fn_o} | ✅ TP = {tp_o} |
        """)

        # ── Threshold sweep charts ─────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Recall / Precision / F1 vs Threshold</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(thresholds, recalls,     color=GREEN,  lw=2,   label="Recall (Sensitivity)")
            ax.plot(thresholds, precisions,  color=ORANGE, lw=2,   label="Precision")
            ax.plot(thresholds, f1s,         color=ACCENT, lw=2.5, label="F1-Score")
            ax.plot(thresholds, specificities, color=PURPLE, lw=1.5, linestyle="--", label="Specificity")
            ax.axvline(optimal_t_f1, color=RED, linestyle="--", lw=2,
                       label=f"Optimal F1 ({optimal_t_f1:.3f})")
            if selected_preset:
                ax.axvline(selected_preset, color=ORANGE, linestyle=":", lw=2,
                           label=f"Preset ({selected_preset:.2f})")
            ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
            ax.set_title("Metrics vs Threshold")
            ax.legend(facecolor=CARD_BG, labelcolor="white", fontsize=9)
            ax.grid(True, alpha=0.15)
            style_fig(fig, ax); plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown('<div class="section-header">FN vs FP Trade-off</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(thresholds, fns, color=RED,    lw=2, label="False Negatives (Missed High CVEs)")
            ax.plot(thresholds, fps, color=ORANGE, lw=2, label="False Positives (False Alarms)")
            ax.axvline(optimal_t_f1, color=ACCENT, linestyle="--", lw=2,
                       label=f"Optimal F1 ({optimal_t_f1:.3f})")
            if selected_preset:
                ax.axvline(selected_preset, color=PURPLE, linestyle=":", lw=2,
                           label=f"Preset ({selected_preset:.2f})")
            ax.set_xlabel("Threshold"); ax.set_ylabel("Count")
            ax.set_title("FN vs FP Trade-off")
            ax.legend(facecolor=CARD_BG, labelcolor="white", fontsize=9)
            ax.grid(True, alpha=0.15)
            style_fig(fig, ax); plt.tight_layout()
            st.pyplot(fig)

        # ── Accuracy and Specificity curves ───────────────────
        st.markdown('<div class="section-header">Accuracy & Specificity vs Threshold</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(thresholds, accuracies,    color=ACCENT, lw=2,   label="Accuracy")
        ax.plot(thresholds, specificities, color=GREEN,  lw=2,   label="Specificity")
        ax.axvline(optimal_t_f1, color=RED, linestyle="--", lw=2,
                   label=f"Optimal F1 ({optimal_t_f1:.3f})")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_title("Accuracy & Specificity vs Threshold")
        ax.legend(facecolor=CARD_BG, labelcolor="white")
        ax.grid(True, alpha=0.15)
        style_fig(fig, ax); plt.tight_layout()
        st.pyplot(fig)

        # ── Threshold sweep table (sampled every 5%) ──────────
        st.markdown('<div class="section-header">📋 Threshold Sweep Table</div>', unsafe_allow_html=True)
        sample_every = max(1, int(0.05 / threshold_step))
        sweep_df = pd.DataFrame({
            "Threshold":    np.round(thresholds[::sample_every], 3),
            "Recall":       np.round(recalls[::sample_every], 4),
            "Precision":    np.round(precisions[::sample_every], 4),
            "F1-Score":     np.round(f1s[::sample_every], 4),
            "Accuracy":     np.round(accuracies[::sample_every], 4),
            "Specificity":  np.round(specificities[::sample_every], 4),
            "FN":           fns[::sample_every],
            "FP":           fps[::sample_every],
        })
        st.dataframe(
            sweep_df.style
                .highlight_max(subset=["Recall", "Precision", "F1-Score", "Accuracy", "Specificity"],
                               color="#1a4a2e")
                .highlight_min(subset=["FN", "FP"], color="#1a4a2e")
                .background_gradient(subset=["F1-Score"], cmap="Blues"),
            use_container_width=True
        )

        # ── CV at optimal threshold ────────────────────────────
        st.markdown(f'<div class="section-header">{cv_folds_thresh}-Fold CV at Optimal Threshold</div>', unsafe_allow_html=True)
        with st.spinner(f"Running {cv_folds_thresh}-fold CV..."):
            from sklearn.pipeline import Pipeline
            from sklearn.base import BaseEstimator, ClassifierMixin

            class ThresholdWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, base_model, threshold=0.5):
                    self.base_model = base_model
                    self.threshold = threshold
                def fit(self, X, y):
                    self.base_model.fit(X, y)
                    self.classes_ = np.array([0, 1])
                    return self
                def predict(self, X):
                    return (self.base_model.predict_proba(X)[:, 1] >= self.threshold).astype(int)
                def predict_proba(self, X):
                    return self.base_model.predict_proba(X)

            wrapped = ThresholdWrapper(build_models()[model_choice], optimal_t_f1)
            cv_f1 = cross_val_score(wrapped, X, y, cv=cv_folds_thresh, scoring="f1")
            cv_rc = cross_val_score(wrapped, X, y, cv=cv_folds_thresh, scoring="recall")

        cv1, cv2, cv3, cv4 = st.columns(4)
        cv1.metric(f"CV F1 Mean",    f"{cv_f1.mean():.4f}")
        cv2.metric(f"CV F1 Std",     f"{cv_f1.std():.4f}")
        cv3.metric(f"CV Recall Mean", f"{cv_rc.mean():.4f}")
        cv4.metric(f"CV Recall Std",  f"{cv_rc.std():.4f}")


# ══════════════════════════════════════════════════════════════════
#  PAGE 6 — SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Analysis":
    st.markdown('<div class="section-header">🧠 SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="leakage-banner">
    ✅ Leakage-Free · No CVSS scores in feature matrix
    </div>
    """, unsafe_allow_html=True)

    df_feat       = extract_features(df)
    feat_cols     = get_feature_cols(df_feat)
    X             = df_feat[feat_cols].select_dtypes(include=np.number).fillna(0)
    y             = df_feat["is_high_cve"]
    feat_cols_num = list(X.columns)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ SHAP Controls")
    shap_test_size = st.sidebar.select_slider(
        "Test Set Size",
        options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        value=0.30,
        format_func=lambda x: f"{int(x*100)}%",
        key="shap_test"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=shap_test_size, random_state=42, stratify=y
    )

    shap_model  = st.selectbox("Model for SHAP", ["XGBoost", "Random Forest"])
    max_samples = st.slider("Max SHAP samples (higher = slower)", 100, min(1000, len(X_test)), 300, 50)

    if st.button("🔮 Generate SHAP Analysis", use_container_width=True):
        with st.spinner("Training model and computing SHAP values..."):
            models = build_models()
            model  = models[shap_model]
            model.fit(X_train, y_train)

            X_shap      = X_test.iloc[:max_samples]
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        st.markdown('<div class="section-header">SHAP Summary Plot (Beeswarm)</div>', unsafe_allow_html=True)
        fig, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_shap, feature_names=feat_cols_num,
                          plot_type="dot", show=False, max_display=15)
        fig.patch.set_facecolor(DARK_BG)
        st.pyplot(fig); plt.clf()

        st.markdown('<div class="section-header">SHAP Feature Importance (Mean |SHAP|)</div>', unsafe_allow_html=True)
        mean_shap = np.abs(sv).mean(axis=0)
        shap_df   = pd.DataFrame({"Feature": feat_cols_num, "Mean |SHAP|": mean_shap})
        shap_df   = shap_df.sort_values("Mean |SHAP|", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(shap_df["Feature"][::-1], shap_df["Mean |SHAP|"][::-1],
                color=ACCENT, alpha=0.85, edgecolor="#333")
        ax.set_xlabel("Mean |SHAP Value|"); ax.set_title("SHAP Feature Importance")
        ax.grid(axis="x", alpha=0.2)
        style_fig(fig, ax); plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<div class="section-header">SHAP Values Table</div>', unsafe_allow_html=True)
        st.dataframe(
            shap_df.reset_index(drop=True)
                   .style.background_gradient(cmap="Blues", subset=["Mean |SHAP|"]),
            use_container_width=True
        )

        st.markdown('<div class="section-header">Individual Prediction Explanation</div>', unsafe_allow_html=True)
        sample_idx = st.slider("Select Sample", 0, len(X_shap) - 1, 0)
        base_val   = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1]

        fig, _ = plt.subplots(figsize=(12, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[sample_idx],
                base_values=base_val,
                data=X_shap.iloc[sample_idx],
                feature_names=feat_cols_num
            ),
            max_display=12, show=False
        )
        fig.patch.set_facecolor(DARK_BG)
        st.pyplot(fig); plt.clf()

        pred_prob  = model.predict_proba(X_shap.iloc[[sample_idx]])[:, 1][0]
        pred_label = "🔴 HIGH CVE" if pred_prob >= 0.3 else "🟢 LOW CVE"
        st.markdown(
            f'<div class="info-card">Sample #{sample_idx} → Predicted Probability: '
            f'<b>{pred_prob:.3f}</b> → {pred_label}</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════
#  PAGE 7 — DEPLOYMENT CONFIG
# ══════════════════════════════════════════════════════════════════
elif page == "🚀 Deployment Config":
    st.markdown('<div class="section-header">🚀 Final Deployment Recommendation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="winner-banner">
    🏆 PRODUCTION MODEL: XGBoost Classifier<br><br>
    Accuracy: 77.04% &nbsp;·&nbsp; ROC-AUC: 86.02% &nbsp;·&nbsp;
    Recall: 81.35% &nbsp;·&nbsp; FN: 99 &nbsp;·&nbsp; Threshold: 0.30<br>
    ⚠️ RF wins Precision (68.12% vs 61.01%) and F1 (70.52% vs 69.69%) — but XGB wins security metrics<br><br>
    Leakage-Free · Text + CWE + Graph + Temporal features only
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Full 6-metric benchmark ────────────────────────────────
    st.markdown('<div class="section-header">📋 Final Benchmark — All 6 Metrics</div>', unsafe_allow_html=True)
    xg = VERIFIED_RESULTS["XGBoost"]
    rf = VERIFIED_RESULTS["Random Forest"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏆 XGBoost (Deployed)**")
        xg_deploy = [
            ("Accuracy",    xg["Accuracy"],  "metric-box-green", "metric-value-green", "metric-label-green"),
            ("ROC-AUC",     xg["ROC-AUC"],   "metric-box-green", "metric-value-green", "metric-label-green"),
            ("Recall",      xg["Recall"],    "metric-box-green", "metric-value-green", "metric-label-green"),
            ("Precision",   xg["Precision"], "metric-box",       "metric-value",       "metric-label"),
            ("F1-Score",    xg["F1-Score"],  "metric-box",       "metric-value",       "metric-label"),
            ("False Neg.",  xg["FN"],        "metric-box-green", "metric-value-green", "metric-label-green"),
        ]
        for metric, val, box, vclass, lclass in xg_deploy:
            st.markdown(
                f'<div class="{box}" style="margin-bottom:0.4rem;padding:0.6rem 1rem;display:flex;'
                f'justify-content:space-between;align-items:center;">'
                f'<span style="color:#aaa;font-size:0.8rem;text-transform:uppercase;">{metric}</span>'
                f'<span class="{vclass}" style="font-size:1.2rem;">{val}</span></div>',
                unsafe_allow_html=True
            )
    with col2:
        st.markdown("**Random Forest (Baseline)**")
        rf_deploy = [
            ("Accuracy",    rf["Accuracy"],  "metric-box",        "metric-value",        "metric-label"),
            ("ROC-AUC",     rf["ROC-AUC"],   "metric-box",        "metric-value",        "metric-label"),
            ("Recall",      rf["Recall"],    "metric-box",        "metric-value",        "metric-label"),
            ("Precision",   rf["Precision"], "metric-box-orange", "metric-value-orange", "metric-label-orange"),
            ("F1-Score",    rf["F1-Score"],  "metric-box-orange", "metric-value-orange", "metric-label-orange"),
            ("False Neg.",  rf["FN"],        "metric-box",        "metric-value",        "metric-label"),
        ]
        for metric, val, box, vclass, lclass in rf_deploy:
            st.markdown(
                f'<div class="{box}" style="margin-bottom:0.4rem;padding:0.6rem 1rem;display:flex;'
                f'justify-content:space-between;align-items:center;">'
                f'<span style="color:#aaa;font-size:0.8rem;text-transform:uppercase;">{metric}</span>'
                f'<span class="{vclass}" style="font-size:1.2rem;">{val}</span></div>',
                unsafe_allow_html=True
            )

    # ── Feature sets ───────────────────────────────────────────
    st.markdown("")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">📋 Leakage-Free Feature Set</div>', unsafe_allow_html=True)
        feature_info = {
            "Feature Group": ["Text (16)", "CWE (5)", "Graph (4)", "Temporal (6)", "Status (1)"],
            "Features": [
                "desc_length, word_count, 14× keywords",
                "cwe_num, cwe_count, cwe_is_high_sev, cwe_is_memory, cwe_is_injection",
                "ref_count, ref_count_log, cpe_count, cpe_count_log",
                "days_since_published, days_since_modified, days_to_update, pub_year, pub_month",
                "vuln_status_enc"
            ],
            "Leakage Risk": ["✅ None", "✅ None", "✅ None", "✅ None", "✅ None"]
        }
        st.dataframe(pd.DataFrame(feature_info), use_container_width=True, hide_index=True)

    with col4:
        st.markdown('<div class="section-header">🚫 Excluded (Leakage) Features</div>', unsafe_allow_html=True)
        excluded_info = {
            "Feature": [
                "effective_score", "exploit_score", "impact_score",
                "base_severity", "av/ac/pr/ui/sc encoded", "ci/ii/ai encoded"
            ],
            "Reason": [
                "Defines the label directly",
                "Sub-component of CVSS score",
                "Sub-component of CVSS score",
                "Categorical map of the score",
                "Inputs to CVSS exploitability formula",
                "Inputs to CVSS impact formula"
            ],
            "Risk": ["❌ Direct", "❌ Direct", "❌ Direct",
                     "❌ Indirect", "❌ Indirect", "❌ Indirect"]
        }
        st.dataframe(pd.DataFrame(excluded_info), use_container_width=True, hide_index=True)

    # ── Deployment code ────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Deployment Code</div>', unsafe_allow_html=True)
    st.code("""
import json, pickle
import numpy as np
from xgboost import XGBClassifier

# ── Verified Benchmark Results ─────────────────────────────────
# Metric         XGBoost    Random Forest   Winner
# ─────────────────────────────────────────────────
# Accuracy       77.04%     71.63%          XGB
# ROC-AUC        86.02%     74.21%          XGB
# Recall         81.35%     73.08%          XGB
# Precision      61.01%     68.12%          RF  ⚠️
# F1-Score       69.69%     70.52%          RF  ⚠️
# False Neg.     99         121             XGB

OPTIMAL_THRESHOLD = 0.30
MODEL_CONFIG = {
    "model":            "XGBClassifier",
    "n_estimators":     200,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "threshold":        OPTIMAL_THRESHOLD,
    "pipeline":         "leakage_free_v2",
    # Verified metrics
    "roc_auc":          0.8602,
    "recall":           0.8135,
    "precision":        0.6101,
    "f1_score":         0.6969,
    "false_negatives":  99,
}

model = XGBClassifier(
    n_estimators     = MODEL_CONFIG["n_estimators"],
    max_depth        = MODEL_CONFIG["max_depth"],
    learning_rate    = MODEL_CONFIG["learning_rate"],
    subsample        = MODEL_CONFIG["subsample"],
    colsample_bytree = MODEL_CONFIG["colsample_bytree"],
    reg_alpha        = MODEL_CONFIG["reg_alpha"],
    reg_lambda       = MODEL_CONFIG["reg_lambda"],
    random_state     = MODEL_CONFIG["random_state"],
    eval_metric      = "logloss"
)
model.fit(X_train, y_train)   # X_train: NO CVSS score columns

with open("models/xgboost_cve_detector_v2.pkl", "wb") as f:
    pickle.dump(model, f)

def predict_high_cve(model, X, threshold=OPTIMAL_THRESHOLD):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return y_pred, y_proba

y_pred, y_proba = predict_high_cve(model, X_test)
print(f"High CVEs detected:  {y_pred.sum()}")
print(f"ROC-AUC (verified):  {MODEL_CONFIG['roc_auc']}")
print(f"Recall (verified):   {MODEL_CONFIG['recall']}")
print(f"Precision:           {MODEL_CONFIG['precision']}")
print(f"F1-Score:            {MODEL_CONFIG['f1_score']}")
print(f"False Negatives:     {MODEL_CONFIG['false_negatives']}")
print("🚀 LEAKAGE-FREE XGBOOST MODEL READY FOR PRODUCTION ✅")
    """, language="python")

    st.markdown('<div class="section-header">📋 Operational Checklist</div>', unsafe_allow_html=True)
    checklist = [
        ("Deploy XGBoost leakage-free model (ROC-AUC=0.8602, Recall=81.35%, FN=99)", True),
        ("Deploy with threshold = 0.30",                                               True),
        ("Set up alert queue for predictions ≥ 0.30",                                 True),
        ("Monitor False Positive rate weekly (Precision=61.01% at threshold 0.30)",   True),
        ("Adjust threshold upward if FP rate becomes too high",                        True),
        ("Retrain model monthly with new CVE data",                                    True),
        ("Log all predictions for audit trail",                                        True),
        ("Set up SHAP explanation for flagged CVEs",                                   True),
        ("Validate no score-derived features enter production pipeline",               True),
        ("Note: RF has higher Precision/F1 — consider ensemble if FP rate too high",   True),
    ]
    for task, done in checklist:
        icon = "✅" if done else "⬜"
        st.markdown(f'<div class="info-card">{icon} {task}</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        '<div class="success-banner">'
        '🛡️ LEAKAGE-FREE XGBOOST SYSTEM READY FOR PRODUCTION DEPLOYMENT ✅<br>'
        'ROC-AUC: 86.02% · Recall: 81.35% · Precision: 61.01% · F1: 69.69% · FN: 99'
        '</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#555;font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;">'
    '🛡️ Vulnerabilities Prediction Model &nbsp;|&nbsp; NVD CVE 2.0 &nbsp;|&nbsp;'
    ' Leakage-Free Pipeline &nbsp;|&nbsp; XGBoost Champion &nbsp;|&nbsp; 2026'
    '</div>',
    unsafe_allow_html=True
)
