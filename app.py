"""
SkinSeg AI — Streamlit Dashboard
Few-Shot Skin Lesion Segmentation using MobileNetV3-Small + U-Net

Run:  streamlit run app.py
"""

import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import torch

from src.utils import (
    TRAIN_SIZES, EXPERIMENT_CONFIG, MODELS_DIR,
    discover_models, model_path, model_exists,
    load_experiment_results, discover_test_pairs,
    fmt3, fmt_pct, status_summary, get_device,
)
from src.preprocessing import load_image_rgb, load_mask_binary, INPUT_SIZE
from src.inference import (
    load_model, predict_mask,
    prob_map_to_heatmap_rgb, overlay_mask_on_image, binary_mask_to_rgb,
)
from src.metrics import compute_all_metrics


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SkinSeg AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Medical-AI Research aesthetic
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ───────────────────────────────────────────────────────────────── */
:root {
    --bg-primary:   #0e1117;
    --bg-card:      #161b27;
    --bg-card2:     #1c2333;
    --accent:       #00d4aa;
    --accent2:      #4f8ef7;
    --accent3:      #f7a24f;
    --danger:       #f74f4f;
    --text-primary: #e8eaf0;
    --text-muted:   #8892a4;
    --border:       rgba(255,255,255,0.07);
    --radius:       12px;
    --radius-sm:    8px;
}
.stApp { background-color: var(--bg-primary); }

/* ── Cards ─────────────────────────────────────────────────────────────── */
.sk-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 16px;
}
.sk-card-accent {
    background: linear-gradient(135deg, #0e2340 0%, #0a1a2e 100%);
    border: 1px solid var(--accent2);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* ── KPI Cards ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.1;
    letter-spacing: -1px;
}
.kpi-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 6px;
}

/* ── Section headers ────────────────────────────────────────────────────── */
.sk-section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.3px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
.metric-card {
    background: var(--bg-card2);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    text-align: center;
    border: 1px solid var(--border);
}
.metric-val {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Status badge ───────────────────────────────────────────────────────── */
.badge-ready {
    display: inline-block;
    background: rgba(0,212,170,0.15);
    color: #00d4aa;
    border: 1px solid rgba(0,212,170,0.35);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-wait {
    display: inline-block;
    background: rgba(247,162,79,0.15);
    color: #f7a24f;
    border: 1px solid rgba(247,162,79,0.35);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-error {
    display: inline-block;
    background: rgba(247,79,79,0.15);
    color: #f74f4f;
    border: 1px solid rgba(247,79,79,0.35);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Pipeline step ──────────────────────────────────────────────────────── */
.pipe-step {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.85rem;
    color: var(--text-primary);
}
.pipe-arrow {
    color: var(--accent);
    text-align: center;
    font-size: 1rem;
    margin: 2px 0;
}

/* ── Config row ─────────────────────────────────────────────────────────── */
.cfg-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.cfg-key { color: var(--text-muted); }
.cfg-val { color: var(--text-primary); font-weight: 500; }

/* ── Info / warning boxes ───────────────────────────────────────────────── */
.sk-info {
    background: rgba(79,142,247,0.10);
    border-left: 3px solid var(--accent2);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: var(--text-primary);
    margin: 10px 0;
}
.sk-warn {
    background: rgba(247,162,79,0.10);
    border-left: 3px solid var(--accent3);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: var(--text-primary);
    margin: 10px 0;
}
.sk-danger {
    background: rgba(247,79,79,0.10);
    border-left: 3px solid var(--danger);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: var(--text-primary);
    margin: 10px 0;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0b0f1a !important;
    border-right: 1px solid var(--border);
}

/* ── Streamlit overrides ────────────────────────────────────────────────── */
h1, h2, h3 { color: var(--text-primary) !important; }
.stRadio label { color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,39,1)",
    font=dict(color="#8892a4", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

ACCENT_COLORS = ["#00d4aa", "#4f8ef7", "#f7a24f", "#d44fd4", "#f74f4f"]


def plotly_card(fig):
    """Render a Plotly figure inside a dark card."""
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_resource
def cached_load_model(model_path_str: str):
    """Load and cache a model (avoids reloading on every interaction)."""
    device = get_device()
    return load_model(model_path_str, device), device


@st.cache_data(ttl=30)
def cached_load_results():
    return load_experiment_results()


@st.cache_data(ttl=30)
def cached_discover_models():
    return discover_models()


@st.cache_data(ttl=30)
def cached_status():
    return status_summary()


def section_header(title: str, icon: str = ""):
    st.markdown(
        f'<div class="sk-section-title">{icon + " " if icon else ""}{title}</div>',
        unsafe_allow_html=True,
    )


def metric_card_html(label: str, value: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-val">{value}</div>'
        f'<div class="metric-lbl">{label}</div>'
        f'</div>'
    )


def kpi_card_html(value: str, label: str) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
    )


def show_model_missing(train_size: int):
    expected = model_path(train_size)
    st.markdown(
        f'<div class="sk-warn">'
        f'⚠️ <strong>{train_size}-image model is not available yet.</strong><br>'
        f'Expected file:<br>'
        f'<code>models/mobilenetv3_unet_{train_size}_best.pth</code>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="padding:16px 0 8px;">'
        '<span style="font-size:1.8rem;">🧬</span>'
        '<span style="font-size:1.2rem;font-weight:700;color:#e8eaf0;margin-left:8px;">SkinSeg AI</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    status = cached_status()
    n_avail  = status["available_models"]
    n_total  = status["total_models"]
    res_rdy  = status["results_ready"]

    if n_avail == n_total:
        st.markdown('<span class="badge-ready">● Models Ready</span>', unsafe_allow_html=True)
    elif n_avail > 0:
        st.markdown(f'<span class="badge-wait">● {n_avail}/{n_total} Models Available</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-wait">● Awaiting Training Results</span>', unsafe_allow_html=True)

    st.markdown(
        f'<p style="font-size:0.72rem;color:#8892a4;margin:6px 0 16px;">'
        f'Available Models: {n_avail} / {n_total}</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:0 0 12px;">', unsafe_allow_html=True)

    pages = [
        ("🏠", "Dashboard"),
        ("🔬", "Experiment Lab"),
        ("🎯", "Segmentation Studio"),
        ("📊", "Model Comparison"),
        ("🧪", "Test Evaluation"),
        ("ℹ️", "About Project"),
    ]

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

    for icon, name in pages:
        is_active = st.session_state.current_page == name
        border = f"border-left:3px solid #00d4aa;background:rgba(0,212,170,0.08);" if is_active else ""
        if st.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            use_container_width=True,
        ):
            st.session_state.current_page = name
            st.rerun()

    st.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:16px 0 10px;">', unsafe_allow_html=True)

    device = get_device()
    device_label = "🟢 CUDA (GPU)" if device.type == "cuda" else "🔵 CPU"
    st.markdown(
        f'<p style="font-size:0.72rem;color:#8892a4;">Device: {device_label}</p>',
        unsafe_allow_html=True,
    )
    if device.type == "cpu":
        st.info("Running inference on CPU", icon="ℹ️")

PAGE = st.session_state.current_page


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD / OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

if PAGE == "Dashboard":

    st.markdown(
        '<h1 style="font-size:2.2rem;font-weight:800;letter-spacing:-1px;margin-bottom:4px;">🧬 SkinSeg AI</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:1.05rem;color:#8892a4;margin-bottom:4px;font-weight:500;">'
        'Few-Shot Skin Lesion Segmentation</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.88rem;color:#616a7a;margin-bottom:28px;">'
        'Exploring how limited labeled medical data affects segmentation performance '
        'using MobileNetV3-Small + U-Net.</p>',
        unsafe_allow_html=True,
    )

    # ── KPI Cards ───────────────────────────────────────────────────────────
    cols = st.columns(4)
    kpis = [
        ("4", "Training Configurations"),
        ("500", "Maximum Training Images"),
        ("224×224", "Input Resolution"),
        ("MobileNetV3", "Lightweight Encoder"),
    ]
    for col, (val, lbl) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card_html(val, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Experiment Progression ───────────────────────────────────────────────
    section_header("Experiment Progression", "🔭")
    avail = cached_discover_models()

    prog_cols = st.columns(7)
    sizes_str = ["50", "→", "100", "→", "250", "→", "500"]
    for i, (col, label) in enumerate(zip(prog_cols, sizes_str)):
        with col:
            if label == "→":
                st.markdown(
                    '<p style="text-align:center;color:#4f8ef7;font-size:1.2rem;margin-top:12px;">→</p>',
                    unsafe_allow_html=True,
                )
            else:
                size = int(label)
                color = "#00d4aa" if size in avail else "#f7a24f"
                status_dot = "✓" if size in avail else "○"
                st.markdown(
                    f'<div style="text-align:center;background:rgba(22,27,39,1);'
                    f'border:1px solid {color}33;border-radius:8px;padding:10px 4px;">'
                    f'<div style="font-size:1.15rem;font-weight:700;color:{color};">{label}</div>'
                    f'<div style="font-size:0.65rem;color:{color};margin-top:2px;">{status_dot}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Experiment Summary Table ─────────────────────────────────────────────
    section_header("Experiment Results Summary", "📋")
    df = cached_load_results()

    if df is None:
        st.markdown(
            '<div class="sk-warn">'
            '📊 <strong>Experiment results are not available yet.</strong><br>'
            'Run the training pipeline and place <code>few_shot_experiment_results.csv</code> '
            'in <code>results/</code>.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        col_map = {
            "train_size":     "Training Images",
            "best_epoch":     "Best Epoch",
            "best_val_dice":  "Val Dice",
            "best_val_iou":   "Val IoU",
            "test_dice":      "Test Dice",
            "test_iou":       "Test IoU",
            "test_precision": "Precision",
            "test_recall":    "Recall",
        }
        display_cols = {k: v for k, v in col_map.items() if k in df.columns}
        display_df = df[list(display_cols.keys())].rename(columns=display_cols)

        # Format numeric columns
        float_cols = [c for c in display_df.columns if c not in ("Training Images", "Best Epoch")]
        for c in float_cols:
            display_df[c] = display_df[c].apply(lambda x: f"{float(x):.3f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Highlight best available model
        best_col = "best_val_dice" if "best_val_dice" in df.columns else (
                   "test_dice"     if "test_dice"     in df.columns else None)
        if best_col:
            best_row = df.loc[df[best_col].idxmax()]
            n_avail_models = len(cached_discover_models())
            st.markdown(
                f'<div class="sk-info" style="margin-top:10px;">'
                f'<strong>{int(best_row["train_size"])}-image configuration</strong> currently records '
                f'the highest validation Dice of <strong>{best_row[best_col]:.3f}</strong> '
                f'among the {n_avail_models} available trained models. '
                f'{"Test-set metrics will appear here once the full evaluation CSV is added." if "test_dice" not in df.columns else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Live Validation Scores from Checkpoints ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Live Validation Scores — From Checkpoints", "🎯")

    avail = cached_discover_models()
    df_live = cached_load_results()

    if not avail:
        st.markdown(
            '<div class="sk-warn">No trained models detected yet. '
            'Place <code>.pth</code> files in <code>models/</code> or <code>results/</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Per-model metric cards with best_val_dice + best_val_iou from CSV
        model_cols = st.columns(len(avail))
        for col, (sz, _) in zip(model_cols, avail.items()):
            val_dice = val_iou = epoch = None
            if df_live is not None and sz in df_live["train_size"].values:
                row = df_live[df_live["train_size"] == sz].iloc[0]
                val_dice = row.get("best_val_dice")
                val_iou  = row.get("best_val_iou")
                epoch    = row.get("best_epoch")

            dice_str  = f"{float(val_dice):.4f}" if val_dice is not None else "—"
            iou_str   = f"{float(val_iou):.4f}"  if val_iou  is not None else "—"
            epoch_str = f"Epoch {int(epoch)}"     if epoch    is not None else "—"

            # Colour gradient: low=orange, high=teal
            try:
                dice_val = float(val_dice) if val_dice is not None else 0.0
            except Exception:
                dice_val = 0.0
            bar_pct = int(dice_val * 100)
            bar_color = "#00d4aa" if dice_val >= 0.7 else ("#f7a24f" if dice_val >= 0.5 else "#f74f4f")

            with col:
                st.markdown(
                    f'<div class="sk-card" style="text-align:center;padding:18px 12px;">'
                    f'<div style="font-size:0.72rem;color:#8892a4;text-transform:uppercase;'
                    f'letter-spacing:1.2px;margin-bottom:8px;">{sz}-Image Model</div>'
                    f'<div style="font-size:2rem;font-weight:800;color:{bar_color};'
                    f'letter-spacing:-1px;line-height:1;">{dice_str}</div>'
                    f'<div style="font-size:0.68rem;color:#8892a4;margin:2px 0 10px;">Val Dice</div>'
                    f'<div style="background:rgba(255,255,255,0.06);border-radius:4px;'
                    f'height:5px;margin-bottom:10px;">'
                    f'<div style="width:{bar_pct}%;background:{bar_color};height:5px;'
                    f'border-radius:4px;transition:width 0.4s;"></div></div>'
                    f'<div style="font-size:0.78rem;color:#e8eaf0;">IoU&nbsp;&nbsp;<strong>{iou_str}</strong></div>'
                    f'<div style="font-size:0.68rem;color:#616a7a;margin-top:4px;">{epoch_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Dice coefficient progress bar comparison
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Dice Coefficient — Model Comparison", "📊")

        if df_live is not None and "best_val_dice" in df_live.columns:
            avail_df = df_live[df_live["train_size"].isin(list(avail.keys()))].sort_values("train_size")
            for _, row in avail_df.iterrows():
                sz_r   = int(row["train_size"])
                dv     = float(row["best_val_dice"])
                iv     = float(row.get("best_val_iou", 0))
                color  = "#00d4aa" if dv >= 0.7 else ("#f7a24f" if dv >= 0.5 else "#f74f4f")
                pct    = int(dv * 100)
                bar_label = f"{sz_r}-image"
                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.8rem;margin-bottom:4px;">'
                    f'<span style="color:#e8eaf0;font-weight:600;">{bar_label}</span>'
                    f'<span style="color:{color};font-weight:700;">Dice {dv:.4f} &nbsp;|&nbsp; IoU {iv:.4f}</span>'
                    f'</div>'
                    f'<div style="background:rgba(255,255,255,0.06);border-radius:6px;height:10px;">'
                    f'<div style="width:{pct}%;background:linear-gradient(90deg,{color}99,{color});'
                    f'height:10px;border-radius:6px;"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="sk-info">Place the results CSV to see the Dice progress bars.</div>',
                unsafe_allow_html=True,
            )

    # ── Quick Architecture Overview ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Architecture at a Glance", "⚙️")
    left, right = st.columns([1, 1])


    with left:
        st.markdown(
            '<div class="sk-card">'
            '<div class="cfg-row"><span class="cfg-key">Encoder</span><span class="cfg-val">MobileNetV3-Small (ImageNet)</span></div>'
            '<div class="cfg-row"><span class="cfg-key">Decoder</span><span class="cfg-val">U-Net</span></div>'
            '<div class="cfg-row"><span class="cfg-key">Skip Connections</span><span class="cfg-val">Yes (4 stages)</span></div>'
            '<div class="cfg-row"><span class="cfg-key">Output Channels</span><span class="cfg-val">1 (binary segmentation)</span></div>'
            '<div class="cfg-row"><span class="cfg-key">Input Resolution</span><span class="cfg-val">224 × 224</span></div>'
            '<div class="cfg-row"><span class="cfg-key">Loss</span><span class="cfg-val">Dice + BCEWithLogitsLoss</span></div>'
            '<div class="cfg-row"><span class="cfg-key">Threshold</span><span class="cfg-val">0.5</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="sk-card">'
            '<div class="pipe-step">📥 Input Image  224×224×3</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">🔷 MobileNetV3-Small Encoder</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">🔀 Skip Connections (4 levels)</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">🔶 U-Net Decoder</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">σ Sigmoid → Threshold 0.5</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">📤 Lesion Mask  224×224×1</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EXPERIMENT LAB
# ═══════════════════════════════════════════════════════════════════════════

elif PAGE == "Experiment Lab":

    st.markdown('<h1 style="font-size:2rem;font-weight:800;margin-bottom:4px;">🔬 Experiment Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892a4;margin-bottom:24px;">Explore training configurations and performance across few-shot experiments.</p>', unsafe_allow_html=True)

    # Experiment selector
    if "selected_exp" not in st.session_state:
        st.session_state.selected_exp = 250

    sel_cols = st.columns(len(TRAIN_SIZES))
    for i, sz in enumerate(TRAIN_SIZES):
        with sel_cols[i]:
            label = f"{'✓ ' if model_exists(sz) else ''}{sz} Images"
            if st.button(label, key=f"exp_btn_{sz}", use_container_width=True):
                st.session_state.selected_exp = sz
                st.rerun()

    sel = st.session_state.selected_exp
    cfg = EXPERIMENT_CONFIG[sel]

    st.markdown("<br>", unsafe_allow_html=True)
    section_header(f"Training Configuration — {sel} Images", "⚙️")

    cfg_cols = st.columns(2)
    with cfg_cols[0]:
        st.markdown(
            f'<div class="sk-card">'
            f'<div class="cfg-row"><span class="cfg-key">Training Images</span><span class="cfg-val">{sel}</span></div>'
            f'<div class="cfg-row"><span class="cfg-key">Encoder</span><span class="cfg-val">{cfg["encoder"]}</span></div>'
            f'<div class="cfg-row"><span class="cfg-key">Decoder</span><span class="cfg-val">{cfg["decoder"]}</span></div>'
            f'<div class="cfg-row"><span class="cfg-key">Input Size</span><span class="cfg-val">{cfg["input"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cfg_cols[1]:
        st.markdown(
            f'<div class="sk-card">'
            f'<div class="cfg-row"><span class="cfg-key">Loss Function</span><span class="cfg-val">{cfg["loss"]}</span></div>'
            f'<div class="cfg-row"><span class="cfg-key">Max Epochs</span><span class="cfg-val">{cfg["max_epochs"]}</span></div>'
            f'<div class="cfg-row"><span class="cfg-key">Optimizer</span><span class="cfg-val">{cfg["optimizer"]}</span></div>'
            f'<div class="cfg-row"><span class="cfg-key">Early Stopping</span><span class="cfg-val">{cfg["early_stopping"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Performance Charts ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Performance Visualization", "📈")

    df = cached_load_results()

    if df is None:
        st.markdown(
            '<div class="sk-warn">📊 Results CSV not found. Charts will appear once '
            '<code>results/few_shot_experiment_results.csv</code> is added.</div>',
            unsafe_allow_html=True,
        )
    else:
        x = df["train_size"].tolist()
        chart_cols = st.columns(2)

        # Chart 1 — Dice (val + test if available)
        with chart_cols[0]:
            has_val_dice  = "best_val_dice" in df.columns
            has_test_dice = "test_dice"     in df.columns
            if has_val_dice or has_test_dice:
                fig = go.Figure()
                if has_val_dice:
                    fig.add_trace(go.Scatter(
                        x=x, y=df["best_val_dice"].tolist(),
                        mode="lines+markers",
                        name="Val Dice",
                        line=dict(color="#00d4aa", width=2.5),
                        marker=dict(size=9, color="#00d4aa",
                                    line=dict(width=2, color="#ffffff")),
                        hovertemplate="<b>%{x} images</b><br>Val Dice: %{y:.3f}<extra></extra>",
                    ))
                if has_test_dice:
                    fig.add_trace(go.Scatter(
                        x=x, y=df["test_dice"].tolist(),
                        mode="lines+markers",
                        name="Test Dice",
                        line=dict(color="#4f8ef7", width=2, dash="dot"),
                        marker=dict(size=7, color="#4f8ef7"),
                        hovertemplate="<b>%{x} images</b><br>Test Dice: %{y:.3f}<extra></extra>",
                    ))
                layout = dict(PLOTLY_LAYOUT)
                layout.update(title="Training Data Size vs Dice Score",
                              xaxis_title="Training Images",
                              yaxis_title="Dice Score",
                              yaxis=dict(range=[0, 1],
                                         gridcolor="rgba(255,255,255,0.05)"))
                fig.update_layout(**layout)
                plotly_card(fig)

        # Chart 2 — IoU (val + test if available)
        with chart_cols[1]:
            has_val_iou  = "best_val_iou" in df.columns
            has_test_iou = "test_iou"     in df.columns
            if has_val_iou or has_test_iou:
                fig = go.Figure()
                if has_val_iou:
                    fig.add_trace(go.Scatter(
                        x=x, y=df["best_val_iou"].tolist(),
                        mode="lines+markers",
                        name="Val IoU",
                        line=dict(color="#f7a24f", width=2.5),
                        marker=dict(size=9, color="#f7a24f",
                                    line=dict(width=2, color="#ffffff")),
                        hovertemplate="<b>%{x} images</b><br>Val IoU: %{y:.3f}<extra></extra>",
                    ))
                if has_test_iou:
                    fig.add_trace(go.Scatter(
                        x=x, y=df["test_iou"].tolist(),
                        mode="lines+markers",
                        name="Test IoU",
                        line=dict(color="#d44fd4", width=2, dash="dot"),
                        marker=dict(size=7, color="#d44fd4"),
                        hovertemplate="<b>%{x} images</b><br>Test IoU: %{y:.3f}<extra></extra>",
                    ))
                layout = dict(PLOTLY_LAYOUT)
                layout.update(title="Training Data Size vs IoU",
                              xaxis_title="Training Images",
                              yaxis_title="IoU Score",
                              yaxis=dict(range=[0, 1],
                                         gridcolor="rgba(255,255,255,0.05)"))
                fig.update_layout(**layout)
                plotly_card(fig)

        # Chart 3 — Precision vs Recall (only when test metrics available)
        if "test_precision" in df.columns and "test_recall" in df.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Precision vs Recall by Configuration", "🎯")
            fig = go.Figure()
            for i, row in df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["test_recall"]], y=[row["test_precision"]],
                    mode="markers+text",
                    text=[f'{int(row["train_size"])}'],
                    textposition="top center",
                    marker=dict(size=14, color=ACCENT_COLORS[i % len(ACCENT_COLORS)]),
                    name=f'{int(row["train_size"])}-image',
                    hovertemplate=(
                        f'<b>{int(row["train_size"])}-image config</b><br>'
                        "Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
                    ),
                ))
            layout = dict(PLOTLY_LAYOUT)
            layout.update(title="Precision vs Recall (per configuration)",
                          xaxis_title="Recall", yaxis_title="Precision",
                          xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
                          yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)"))
            fig.update_layout(**layout)
            plotly_card(fig)

        if "test_dice" not in df.columns:
            st.markdown(
                '<div class="sk-info" style="font-size:0.8rem;">📌 Showing <strong>validation</strong> '
                'metrics from checkpoint metadata. Test-set Dice/IoU/Precision/Recall will appear '
                'automatically when added to <code>few_shot_experiment_results.csv</code>.</div>',
                unsafe_allow_html=True,
            )

    # ── Research Insight ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Experiment Insight", "💡")

    if df is None:
        st.markdown(
            '<div class="sk-info">Insights will be generated automatically once experiment results are available.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Auto-generate factual insight from actual numbers
        insight_lines = [
            "The dashboard compares segmentation performance as the number of labeled "
            f"training images increases from {df['train_size'].min()} to {df['train_size'].max()}. "
            "The results below reflect how model performance changes under different data availability conditions."
        ]

        if "test_dice" in df.columns:
            max_row = df.loc[df["test_dice"].idxmax()]
            min_row = df.loc[df["test_dice"].idxmin()]
            insight_lines.append(
                f"The highest Test Dice ({max_row['test_dice']:.3f}) is recorded at "
                f"{int(max_row['train_size'])} training images, while the lowest "
                f"({min_row['test_dice']:.3f}) is at {int(min_row['train_size'])} images."
            )

            # Check monotonicity
            dice_vals = df["test_dice"].tolist()
            is_monotone = all(a <= b for a, b in zip(dice_vals, dice_vals[1:]))
            if not is_monotone:
                # Find non-monotone adjacent pair
                for j in range(len(dice_vals) - 1):
                    if dice_vals[j + 1] < dice_vals[j]:
                        sz_a = int(df.iloc[j]["train_size"])
                        sz_b = int(df.iloc[j + 1]["train_size"])
                        insight_lines.append(
                            f"Notably, the {sz_b}-image experiment currently records a lower "
                            f"test score than the {sz_a}-image experiment. This indicates that "
                            "additional training samples did not translate into higher measured "
                            "performance in this experiment and may warrant further investigation."
                        )
                        break

        for line in insight_lines:
            st.markdown(f'<div class="sk-info">{line}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — SEGMENTATION STUDIO
# ═══════════════════════════════════════════════════════════════════════════

elif PAGE == "Segmentation Studio":

    st.markdown('<h1 style="font-size:2rem;font-weight:800;margin-bottom:4px;">🎯 Segmentation Studio</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8892a4;margin-bottom:4px;">Upload a dermoscopic image and visualize the predicted lesion boundary.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sk-warn" style="font-size:0.78rem;">⚠️ <strong>Research prototype.</strong> '
        'Predicted masks are for research purposes only and do not constitute a medical diagnosis.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    avail = cached_discover_models()

    left_col, right_col = st.columns([1, 2])

    with left_col:
        section_header("Image Input", "🖼️")
        input_mode = st.radio(
            "Select Input Source",
            ["Sample ISIC Images", "Upload Custom Image"],
            horizontal=True,
            label_visibility="collapsed",
        )

        test_pairs = discover_test_pairs()
        img = None

        if input_mode == "Sample ISIC Images":
            if test_pairs:
                pair_names = [p["id"] for p in test_pairs]
                sel_pair_id = st.selectbox(
                    "Choose Sample Image",
                    pair_names,
                    format_func=lambda x: f"ISIC Sample: {x}",
                    label_visibility="collapsed",
                )
                chosen_pair = next(p for p in test_pairs if p["id"] == sel_pair_id)
                img = Image.open(chosen_pair["image"]).convert("RGB")
            else:
                st.info("No sample images found. Upload your own image below.")
                input_mode = "Upload Custom Image"

        if input_mode == "Upload Custom Image":
            uploaded = st.file_uploader(
                "Accept: PNG, JPG, JPEG",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            if uploaded:
                try:
                    img = load_image_rgb(uploaded)
                except Exception:
                    st.error("Invalid image file. Please upload a valid JPG or PNG.")

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Model & Settings", "🤖")

        if not avail:
            st.markdown(
                '<div class="sk-warn">No trained models found.<br>'
                'Place <code>*.pth</code> files in <code>models/</code>.</div>',
                unsafe_allow_html=True,
            )
            sel_model_size = None
        else:
            if "studio_model_size" not in st.session_state or st.session_state.studio_model_size not in avail:
                st.session_state.studio_model_size = sorted(avail.keys())[-1]

            sel_model_size = st.radio(
                "Choose model configuration:",
                options=sorted(avail.keys()),
                format_func=lambda x: f"{x}-image configuration",
                index=list(sorted(avail.keys())).index(st.session_state.studio_model_size),
                key="studio_model_radio",
                label_visibility="collapsed",
            )
            st.session_state.studio_model_size = sel_model_size

            # Interactive confidence threshold slider
            conf_thresh = st.slider(
                "Confidence Threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Adjust sensitivity for boundary classification. Lower values detect more lesion pixels; higher values ensure higher certainty.",
            )

            st.markdown(
                f'<div class="sk-card" style="margin-top:8px;">'
                f'<div class="cfg-row"><span class="cfg-key">Training Data</span><span class="cfg-val">{sel_model_size} images</span></div>'
                f'<div class="cfg-row"><span class="cfg-key">Architecture</span><span class="cfg-val">MobileNetV3-Small U-Net</span></div>'
                f'<div class="cfg-row"><span class="cfg-key">Threshold</span><span class="cfg-val">{conf_thresh:.2f}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        if img is None:
            st.markdown(
                '<div class="sk-card" style="min-height:400px;display:flex;align-items:center;'
                'justify-content:center;text-align:center;">'
                '<div>'
                '<div style="font-size:3rem;margin-bottom:12px;">🩺</div>'
                '<div style="color:#8892a4;font-size:0.9rem;">Choose a sample image from the library<br>or upload your own dermoscopic photo.</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif sel_model_size is None:
            st.markdown(
                '<div class="sk-warn">Please add trained model files to run inference.</div>',
                unsafe_allow_html=True,
            )
        else:
            # Load model
            mp = str(model_path(sel_model_size))
            with st.spinner("Loading model…"):
                try:
                    result, device = cached_load_model(mp)
                    model_obj = result["model"]
                    model_meta = result["meta"]
                except Exception:
                    show_model_missing(sel_model_size)
                    st.stop()

            # Run inference with dynamic threshold
            with st.spinner("Running segmentation…"):
                pred = predict_mask(model_obj, img, device, threshold=conf_thresh)

            prob_map    = pred["prob_map"]
            binary_mask = pred["binary_mask"]
            inf_ms      = pred["inference_ms"]
            lesion_pct  = pred["lesion_pct"]

            # Build four visualization panels
            img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            heatmap_rgb = prob_map_to_heatmap_rgb(prob_map)
            mask_rgb    = binary_mask_to_rgb(binary_mask)
            overlay_rgb = overlay_mask_on_image(img, binary_mask)

            section_header("Segmentation Results", "📊")

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.image(img_resized, caption="Original Image", use_container_width=True)
            with p2:
                st.image(heatmap_rgb, caption="Probability Heatmap", use_container_width=True)
            with p3:
                st.image(mask_rgb, caption=f"Binary Mask (θ={conf_thresh:.2f})", use_container_width=True)
            with p4:
                st.image(overlay_rgb, caption="Lesion Overlay", use_container_width=True)

            # Export download button
            buf = io.BytesIO()
            mask_rgb.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Predicted Mask (PNG)",
                data=buf.getvalue(),
                file_name=f"predicted_mask_{sel_model_size}imgs_thresh{int(conf_thresh*100)}.png",
                mime="image/png",
                use_container_width=True,
            )

            # Stats
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Prediction Statistics", "📉")

            stat_cols = st.columns(5)
            stats = [
                ("Predicted Lesion Area", fmt_pct(lesion_pct)),
                ("Inference Time", f"{inf_ms:.0f} ms"),
                ("Model", f"{sel_model_size}-img"),
                ("Resolution", f"{INPUT_SIZE}×{INPUT_SIZE}"),
                ("Threshold", f"{conf_thresh:.2f}"),
            ]
            for col, (lbl, val) in zip(stat_cols, stats):
                with col:
                    st.markdown(metric_card_html(lbl, val), unsafe_allow_html=True)

            # Stored metadata from checkpoint
            if model_meta:
                st.markdown("<br>", unsafe_allow_html=True)
                meta_parts = []
                if "best_val_dice" in model_meta:
                    meta_parts.append(f"Best Val Dice: <strong>{model_meta['best_val_dice']:.4f}</strong>")
                if "best_val_iou" in model_meta:
                    meta_parts.append(f"Best Val IoU: <strong>{model_meta['best_val_iou']:.4f}</strong>")
                if "best_epoch" in model_meta:
                    meta_parts.append(f"Best Epoch: <strong>{model_meta['best_epoch']}</strong>")
                if meta_parts:
                    st.markdown(
                        f'<div class="sk-info">Checkpoint info — {" · ".join(meta_parts)}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Comparative Visual Evaluation Across All Available Models ────────
            if len(avail) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔬 Compare All Available Models on This Image", expanded=True):
                    comp_cols = st.columns(len(avail))
                    for c_col, (m_sz, m_p) in zip(comp_cols, avail.items()):
                        with c_col:
                            try:
                                m_res, m_dev = cached_load_model(str(m_p))
                                m_pred = predict_mask(m_res["model"], img, m_dev, threshold=conf_thresh)
                                m_overlay = overlay_mask_on_image(img, m_pred["binary_mask"])
                                st.image(m_overlay, caption=f"{m_sz} Images (Area: {m_pred['lesion_pct']:.1f}%)", use_container_width=True)
                            except Exception as e:
                                st.caption(f"{m_sz}-img: Error")

            st.markdown(
                '<div class="sk-info" style="font-size:0.75rem;margin-top:10px;">'
                '🔬 <em>Predicted lesion pixel percentage — not a clinical diagnosis. '
                'This is a research prototype.</em></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

elif PAGE == "Model Comparison":

    st.markdown('<h1 style="font-size:2rem;font-weight:800;margin-bottom:4px;">📊 Few-Shot Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892a4;margin-bottom:8px;">Validation scores extracted directly from checkpoints · test metrics appear when CSV is updated.</p>', unsafe_allow_html=True)

    df    = cached_load_results()
    avail = cached_discover_models()

    # ── Section 1: Per-model score cards ────────────────────────────────────
    section_header("Validation Score Cards — All Available Models", "🃏")

    if not avail:
        st.markdown('<div class="sk-warn">No trained models found yet.</div>', unsafe_allow_html=True)
    else:
        card_cols = st.columns(len(avail))
        for col, (sz, _) in zip(card_cols, avail.items()):
            val_dice = val_iou = epoch = None
            if df is not None and sz in df["train_size"].values:
                row = df[df["train_size"] == sz].iloc[0]
                val_dice = row.get("best_val_dice")
                val_iou  = row.get("best_val_iou")
                epoch    = row.get("best_epoch")

            dv = float(val_dice) if val_dice is not None else 0.0
            iv = float(val_iou)  if val_iou  is not None else 0.0
            color = "#00d4aa" if dv >= 0.7 else ("#f7a24f" if dv >= 0.5 else "#f74f4f")
            ep_str = f"Epoch {int(epoch)}" if epoch is not None else "—"

            with col:
                st.markdown(
                    f'<div class="sk-card" style="text-align:center;padding:20px 10px;">'
                    f'<div style="font-size:0.7rem;color:#8892a4;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">{sz}-Image</div>'
                    f'<div style="font-size:2.2rem;font-weight:800;color:{color};line-height:1;">{dv:.4f}</div>'
                    f'<div style="font-size:0.65rem;color:#8892a4;margin:3px 0 8px;">Val Dice</div>'
                    f'<div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;margin-bottom:10px;">'
                    f'<div style="width:{int(dv*100)}%;background:{color};height:6px;border-radius:4px;"></div></div>'
                    f'<div style="font-size:0.82rem;color:#e8eaf0;margin-bottom:4px;">IoU &nbsp;<strong style="color:{color};">{iv:.4f}</strong></div>'
                    f'<div style="font-size:0.68rem;color:#616a7a;">{ep_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Dice Coefficient Matrix Heatmap ──────────────────────────
    section_header("Dice Coefficient Matrix", "🔥")

    if df is not None and "best_val_dice" in df.columns:
        avail_df = df[df["train_size"].isin(list(avail.keys()))].sort_values("train_size")

        sizes     = [int(s) for s in avail_df["train_size"].tolist()]
        val_dices = [round(float(v), 4) for v in avail_df["best_val_dice"].tolist()]
        val_ious  = [round(float(v), 4) for v in avail_df["best_val_iou"].tolist()] if "best_val_iou" in avail_df.columns else [0.0] * len(sizes)

        # Test metrics if available
        test_dices = [round(float(v), 4) for v in avail_df["test_dice"].tolist()] if "test_dice" in avail_df.columns else None
        test_ious  = [round(float(v), 4) for v in avail_df["test_iou"].tolist()]  if "test_iou"  in avail_df.columns else None

        # Build heatmap z-matrix: rows = metrics, cols = model sizes
        row_labels  = ["Val Dice", "Val IoU"]
        z_values    = [val_dices, val_ious]
        hover_texts = [
            [f"{sz}-image<br>Val Dice: {v:.4f}" for sz, v in zip(sizes, val_dices)],
            [f"{sz}-image<br>Val IoU: {v:.4f}"  for sz, v in zip(sizes, val_ious)],
        ]
        if test_dices:
            row_labels.append("Test Dice")
            z_values.append(test_dices)
            hover_texts.append([f"{sz}-image<br>Test Dice: {v:.4f}" for sz, v in zip(sizes, test_dices)])
        if test_ious:
            row_labels.append("Test IoU")
            z_values.append(test_ious)
            hover_texts.append([f"{sz}-image<br>Test IoU: {v:.4f}" for sz, v in zip(sizes, test_ious)])

        x_labels = [f"{s} imgs" for s in sizes]

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=row_labels,
            text=[[f"{v:.4f}" for v in row] for row in z_values],
            texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            colorscale=[
                [0.0,  "#1a0a0a"],
                [0.3,  "#7a1a1a"],
                [0.5,  "#c45c00"],
                [0.7,  "#00897b"],
                [1.0,  "#00e5c3"],
            ],
            zmin=0.0,
            zmax=1.0,
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title=dict(text="Score", font=dict(color="#8892a4")),
                tickfont=dict(color="#8892a4"),
                len=0.8,
            ),
        ))
        heatmap_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22,27,39,1)",
            font=dict(color="#8892a4", size=12),
            margin=dict(l=100, r=40, t=40, b=60),
            title="Dice Coefficient Matrix — Validation (+ Test when available)",
            xaxis=dict(side="bottom", tickfont=dict(color="#e8eaf0", size=11)),
            yaxis=dict(tickfont=dict(color="#e8eaf0", size=11), autorange="reversed"),
            height=260 + len(row_labels) * 40,
        )
        plotly_card(heatmap_fig)
    else:
        st.markdown(
            '<div class="sk-warn">Heatmap requires the results CSV with '
            '<code>best_val_dice</code> column.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Grouped Bar Chart ─────────────────────────────────────────
    section_header("All Validation Metrics — Side-by-Side Bar Chart", "📊")

    if df is not None:
        avail_df = df[df["train_size"].isin(list(avail.keys()))].sort_values("train_size")
        x_bar    = [f"{int(s)}-image" for s in avail_df["train_size"]]

        bar_metrics = {}
        if "best_val_dice" in avail_df.columns:
            bar_metrics["Val Dice"] = ("#00d4aa", avail_df["best_val_dice"].tolist())
        if "best_val_iou"  in avail_df.columns:
            bar_metrics["Val IoU"]  = ("#4f8ef7", avail_df["best_val_iou"].tolist())
        if "test_dice"      in avail_df.columns:
            bar_metrics["Test Dice"]      = ("#f7a24f", avail_df["test_dice"].tolist())
        if "test_iou"       in avail_df.columns:
            bar_metrics["Test IoU"]       = ("#d44fd4", avail_df["test_iou"].tolist())
        if "test_precision" in avail_df.columns:
            bar_metrics["Test Precision"] = ("#f74f4f", avail_df["test_precision"].tolist())
        if "test_recall"    in avail_df.columns:
            bar_metrics["Test Recall"]    = ("#ffd740", avail_df["test_recall"].tolist())

        if bar_metrics:
            bar_fig = go.Figure()
            for lbl, (color, vals) in bar_metrics.items():
                bar_fig.add_trace(go.Bar(
                    name=lbl, x=x_bar, y=vals,
                    marker_color=color,
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(size=10, color=color),
                    hovertemplate=f"<b>%{{x}}</b><br>{lbl}: %{{y:.4f}}<extra></extra>",
                ))
            bar_layout = dict(PLOTLY_LAYOUT)
            bar_layout.update(
                title="Grouped Metric Comparison",
                barmode="group",
                yaxis=dict(range=[0, 1.1], gridcolor="rgba(255,255,255,0.05)"),
                xaxis_title="Configuration",
                yaxis_title="Score",
                bargap=0.2,
                bargroupgap=0.05,
            )
            bar_fig.update_layout(**bar_layout)
            plotly_card(bar_fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 4: All metrics line chart ────────────────────────────────────
    section_header("Performance Trend — All Metrics", "📈")

    if df is not None:
        avail_df = df[df["train_size"].isin(list(avail.keys()))].sort_values("train_size")
        x_line   = avail_df["train_size"].tolist()

        trend_metrics = {}
        if "best_val_dice" in avail_df.columns:
            trend_metrics["Val Dice"] = ("#00d4aa", avail_df["best_val_dice"].tolist(), "solid")
        if "best_val_iou"  in avail_df.columns:
            trend_metrics["Val IoU"]  = ("#4f8ef7", avail_df["best_val_iou"].tolist(),  "solid")
        if "test_dice"      in avail_df.columns:
            trend_metrics["Test Dice"]      = ("#f7a24f", avail_df["test_dice"].tolist(),      "dot")
        if "test_iou"       in avail_df.columns:
            trend_metrics["Test IoU"]       = ("#d44fd4", avail_df["test_iou"].tolist(),       "dot")
        if "test_precision" in avail_df.columns:
            trend_metrics["Test Precision"] = ("#f74f4f", avail_df["test_precision"].tolist(),  "dashdot")
        if "test_recall"    in avail_df.columns:
            trend_metrics["Test Recall"]    = ("#ffd740", avail_df["test_recall"].tolist(),     "dashdot")

        if trend_metrics:
            line_fig = go.Figure()
            for lbl, (color, vals, dash) in trend_metrics.items():
                line_fig.add_trace(go.Scatter(
                    x=x_line, y=vals,
                    mode="lines+markers",
                    name=lbl,
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(size=9, color=color, line=dict(width=2, color="#ffffff")),
                    hovertemplate=f"<b>%{{x}} images</b><br>{lbl}: %{{y:.4f}}<extra></extra>",
                ))
            line_layout = dict(PLOTLY_LAYOUT)
            line_layout.update(
                title="Metric Trend as Training Data Increases",
                xaxis_title="Training Images",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
            )
            line_fig.update_layout(**line_layout)
            plotly_card(line_fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 5: Radar chart (only if ≥ 2 val metrics) ─────────────────────
    if df is not None:
        avail_df  = df[df["train_size"].isin(list(avail.keys()))].sort_values("train_size")
        radar_cols = [c for c in ["best_val_dice","best_val_iou","test_dice","test_iou","test_precision","test_recall"] if c in avail_df.columns]
        radar_nice = {
            "best_val_dice":  "Val Dice",
            "best_val_iou":   "Val IoU",
            "test_dice":      "Test Dice",
            "test_iou":       "Test IoU",
            "test_precision": "Precision",
            "test_recall":    "Recall",
        }

        if len(radar_cols) >= 2:
            section_header("Radar Profile — Configuration Fingerprints", "🕸️")
            r_labels  = [radar_nice[c] for c in radar_cols]
            r_closed  = r_labels + [r_labels[0]]
            radar_fig = go.Figure()

            for i, (_, row) in enumerate(avail_df.iterrows()):
                sz_r  = int(row["train_size"])
                vals  = [float(row[c]) for c in radar_cols]
                v_cl  = vals + [vals[0]]
                hex_c = ACCENT_COLORS[i % len(ACCENT_COLORS)]
                r, g, b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
                radar_fig.add_trace(go.Scatterpolar(
                    r=v_cl, theta=r_closed,
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.12)",
                    line=dict(color=hex_c, width=2.5),
                    name=f"{sz_r}-image configuration",
                    hovertemplate="%{theta}: %{r:.4f}<extra></extra>",
                ))

            radar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(22,27,39,1)",
                    radialaxis=dict(visible=True, range=[0, 1], color="#8892a4",
                                   gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=9)),
                    angularaxis=dict(color="#e8eaf0", gridcolor="rgba(255,255,255,0.08)",
                                     tickfont=dict(size=11)),
                ),
                font=dict(color="#8892a4", size=11),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0")),
                margin=dict(l=60, r=60, t=60, b=60),
                title="Configuration Radar Fingerprints",
            )
            plotly_card(radar_fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 6: Model availability grid ───────────────────────────────────
    section_header("Model File Availability", "💾")
    avail_cols = st.columns(len(TRAIN_SIZES))
    for col, sz in zip(avail_cols, TRAIN_SIZES):
        with col:
            exists = model_exists(sz)
            color  = "#00d4aa" if exists else "#2a3040"
            icon   = "✓" if exists else "○"
            label  = "Available" if exists else "Pending"
            dv_str = ""
            if exists and df is not None and sz in df["train_size"].values:
                r = df[df["train_size"] == sz].iloc[0]
                if "best_val_dice" in r:
                    dv_str = f'<div style="font-size:0.7rem;color:{color};margin-top:2px;">Dice {float(r["best_val_dice"]):.3f}</div>'
            st.markdown(
                f'<div style="text-align:center;background:rgba(22,27,39,1);'
                f'border:1px solid {color}55;border-radius:8px;padding:12px 4px;">'
                f'<div style="font-size:1.1rem;font-weight:700;color:{color};">{sz}</div>'
                f'<div style="font-size:0.72rem;color:{color};margin-top:3px;">{icon} {label}</div>'
                f'{dv_str}'
                f'</div>',
                unsafe_allow_html=True,
            )







# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 5 — TEST EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

elif PAGE == "Test Evaluation":

    st.markdown('<h1 style="font-size:2rem;font-weight:800;margin-bottom:4px;">🧪 Test Evaluation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8892a4;margin-bottom:4px;">'
        'Evaluate model predictions against ground-truth masks from <code>sample_data/</code>.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sk-info" style="font-size:0.8rem;">'
        '📌 Dice, IoU, Precision, and Recall are computed only when a ground-truth mask exists. '
        'These metrics are <em>not</em> computed for arbitrary uploaded images.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    pairs = discover_test_pairs()
    avail = cached_discover_models()

    if not pairs:
        st.markdown(
            '<div class="sk-warn">'
            '📁 No matched image-mask pairs found.<br>'
            'Add images to <code>sample_data/images/</code> and masks to <code>sample_data/masks/</code>.<br>'
            'Naming convention: <code>ISIC_0000001.jpg</code> + <code>ISIC_0000001_segmentation.png</code>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif not avail:
        st.markdown(
            '<div class="sk-warn">No trained models available. Place <code>*.pth</code> files in <code>models/</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        ctrl_col, _, res_col = st.columns([1, 0.05, 2])

        with ctrl_col:
            section_header("Select Sample", "🖼️")
            pair_ids = [p["id"] for p in pairs]

            if "eval_pair_id" not in st.session_state or st.session_state.eval_pair_id not in pair_ids:
                st.session_state.eval_pair_id = pair_ids[0]

            sel_id = st.selectbox(
                "Test Image",
                pair_ids,
                index=pair_ids.index(st.session_state.eval_pair_id),
                key="eval_pair_sel",
                label_visibility="collapsed",
            )
            st.session_state.eval_pair_id = sel_id

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Select Model", "🤖")

            if "eval_model_size" not in st.session_state or st.session_state.eval_model_size not in avail:
                st.session_state.eval_model_size = sorted(avail.keys())[-1]

            sel_eval_size = st.radio(
                "Model",
                options=sorted(avail.keys()),
                format_func=lambda x: f"{x}-image configuration",
                index=list(sorted(avail.keys())).index(st.session_state.eval_model_size),
                key="eval_model_radio",
                label_visibility="collapsed",
            )
            st.session_state.eval_model_size = sel_eval_size

        with res_col:
            pair = next(p for p in pairs if p["id"] == sel_id)

            # Load image and mask
            try:
                img  = load_image_rgb(pair["image"])
                gt_mask = load_mask_binary(pair["mask"])
            except Exception as e:
                st.error(f"Failed to load image or mask: {e}")
                st.stop()

            # Load model
            mp = str(model_path(sel_eval_size))
            with st.spinner("Loading model…"):
                try:
                    result, device = cached_load_model(mp)
                    model_obj = result["model"]
                except Exception:
                    show_model_missing(sel_eval_size)
                    st.stop()

            # Inference
            with st.spinner("Running evaluation…"):
                pred = predict_mask(model_obj, img, device)

            binary_mask = pred["binary_mask"]
            metrics     = compute_all_metrics(binary_mask, gt_mask)

            # Visualizations
            section_header(f"Evaluation — {sel_id}", "📊")

            img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            gt_rgb      = binary_mask_to_rgb(gt_mask)
            pred_rgb    = binary_mask_to_rgb(binary_mask)
            overlay_rgb = overlay_mask_on_image(img, binary_mask)

            v1, v2, v3, v4 = st.columns(4)
            with v1:
                st.image(img_resized,  caption="Original",      use_container_width=True)
            with v2:
                st.image(gt_rgb,       caption="Ground Truth",  use_container_width=True)
            with v3:
                st.image(pred_rgb,     caption="Prediction",    use_container_width=True)
            with v4:
                st.image(overlay_rgb,  caption="Overlay",       use_container_width=True)

            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Computed Metrics", "📐")
            m1, m2, m3, m4 = st.columns(4)
            for col, (key, lbl) in zip(
                [m1, m2, m3, m4],
                [("dice", "Dice"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall")],
            ):
                with col:
                    st.markdown(metric_card_html(lbl, fmt3(metrics[key])), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 6 — ABOUT PROJECT
# ═══════════════════════════════════════════════════════════════════════════

elif PAGE == "About Project":

    st.markdown('<h1 style="font-size:2rem;font-weight:800;margin-bottom:4px;">ℹ️ About Project</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892a4;margin-bottom:28px;">Research background, architecture, and design rationale.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Overview", "⚙️ Architecture", "🔬 Why Few-Shot?", "📦 Components"])

    with tab1:
        st.markdown("""
### Problem
Skin lesion segmentation requires identifying the **exact lesion region** in dermoscopic images.
Accurate pixel-level delineation enables downstream analysis, clinical decision support, and
lesion-area tracking over time.

### Challenge
Medical image segmentation datasets require **expensive pixel-level annotations** from expert
clinicians. Gathering thousands of annotated dermoscopic images is time-consuming and costly.

### Proposed Approach
This project investigates how well segmentation can be performed when only a **limited number
of labeled training images** are available — a setting commonly referred to as few-shot or
low-data segmentation.

| Step | Detail |
|------|--------|
| Dataset | ISIC 2018 Task 1 (dermoscopic images + binary masks) |
| Training splits | 50 / 100 / 250 / 500 images |
| Encoder | MobileNetV3-Small (ImageNet pretrained) |
| Decoder | U-Net with skip connections |
| Loss | Dice Loss + BCEWithLogitsLoss |
| Metric | Dice Coefficient |
| Inference | Threshold = 0.5 on sigmoid output |

The full training was performed on Google Colab (NVIDIA T4 GPU).
""")

    with tab2:
        st.markdown("### MobileNetV3-Small + U-Net Architecture")
        st.markdown(
            '<div class="sk-card">'
            '<div class="pipe-step">📥 Input Image — 224 × 224 × 3</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">🔷 MobileNetV3-Small Encoder (ImageNet pretrained)<br>'
            '<span style="font-size:0.75rem;color:#8892a4;">Stage 0 → 2 → 4 → 9 → 12 (bridge)</span></div>'
            '<div class="pipe-arrow">↓ ↘ skip connections</div>'
            '<div class="pipe-step">🔵 Bottleneck (576 → 256 channels)</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">🔶 DecoderBlock 4 — 256 + 48 → 128  (stride-16)</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">🔶 DecoderBlock 3 — 128 + 24 → 64   (stride-8)</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">🔶 DecoderBlock 2 — 64 + 16 → 32    (stride-4)</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">🔶 DecoderBlock 1 — 32 + 16 → 16    (stride-2)</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">2× Upsample → 1×1 Conv Head</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">σ Sigmoid output — 224 × 224 × 1</div>'
            '<div class="pipe-arrow">↓</div>'
            '<div class="pipe-step">📤 Binary Lesion Mask (threshold = 0.5)</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Why MobileNetV3?")
        st.markdown("""
- **Lightweight**: designed for mobile and embedded deployment; very few parameters
- **Efficient feature extraction**: inverted residuals + hard-swish activation
- **Strong ImageNet transfer**: encoder weights provide powerful generic visual features
- **Skip-connection compatible**: intermediate feature maps at multiple spatial scales
""")
        st.markdown("### Why U-Net?")
        st.markdown("""
- **Encoder captures semantic information** at low resolution (what the lesion is)
- **Decoder restores spatial resolution** progressively (where the lesion is)
- **Skip connections preserve fine spatial details** from early encoder stages
- Proven architecture for biomedical image segmentation
""")

    with tab3:
        st.markdown("""
### Why Few-Shot / Low-Data Segmentation?

Medical AI systems are often limited by **data scarcity**, not algorithmic capability.
Annotating dermoscopic images at pixel level requires trained dermatologists and is
expensive at scale.

This project studies the following question:

> *How much labeled data does a MobileNetV3-UNet actually need to produce useful segmentations?*

By training four models — each on a progressively larger labeled subset (50, 100, 250, 500) — we can observe:

- The **minimum viable training set** that produces acceptable segmentation
- How **performance scales** with additional labeled data
- Whether **transfer learning** from ImageNet substantially offsets data scarcity

The insights are directly applicable to real-world medical AI scenarios where
annotation budgets are constrained.

### Dataset — ISIC 2018 Task 1

| Property | Value |
|----------|-------|
| Source | International Skin Imaging Collaboration (ISIC) |
| Task | Lesion boundary segmentation |
| Annotations | Binary PNG masks (lesion = white, background = black) |
| Challenge | Diversity in lesion size, shape, color, and imaging conditions |
""")

    with tab4:
        st.markdown("### File Structure")
        st.code("""
skin_lesion_dashboard/
│
├── app.py                      ← Main Streamlit application
├── requirements.txt
├── README.md
│
├── src/
│   ├── model.py                ← MobileNetV3UNet architecture
│   ├── inference.py            ← Prediction + visualization helpers
│   ├── metrics.py              ← Dice, IoU, Precision, Recall
│   ├── preprocessing.py        ← Exact training transforms
│   └── utils.py                ← Discovery, CSV loading, formatting
│
├── models/
│   ├── mobilenetv3_unet_50_best.pth
│   ├── mobilenetv3_unet_100_best.pth
│   ├── mobilenetv3_unet_250_best.pth
│   └── mobilenetv3_unet_500_best.pth
│
├── results/
│   └── few_shot_experiment_results.csv
│
└── sample_data/
    ├── images/    ← ISIC_XXXXXXX.jpg
    └── masks/     ← ISIC_XXXXXXX_segmentation.png
        """, language="text")

        st.markdown("### Disclaimer")
        st.markdown(
            '<div class="sk-warn">'
            'This is a <strong>research prototype</strong>. Predictions are not validated '
            'for clinical use and should not be interpreted as medical diagnoses.'
            '</div>',
            unsafe_allow_html=True,
        )
