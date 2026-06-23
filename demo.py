#!/usr/bin/env python3
"""
Rat Detection Demo — loads precomputed videos + detection JSONs.
No model weights or GPU required. Run: streamlit run demo.py
"""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "demo_data"

MODEL_LABELS = {
    "yolov8m_baseline": "YOLOv8m — Exp2 (rats only)",
    "yolov8m_human":    "YOLOv8m — Exp4 (+human class)",
    "yolov8m_merged":   "YOLOv8m — Exp6 (+rodent_extra)",
    "rtdetr_l":         "RT-DETR-L",
    "yolov9c":          "YOLOv9-C",
    "yolov10m":         "YOLOv10-M",
}

VIDEO_LABELS = {
    "warehouse_1080p": "Warehouse 1080p",
    "warehouse_852p":  "Warehouse 852p",
    "warehouse_450p":  "Warehouse 450p",
}

ZONE_LABELS = [
    ["Top-Left","Top-Centre","Top-Right"],
    ["Mid-Left","Centre","Mid-Right"],
    ["Bottom-Left","Bottom-Centre","Bottom-Right"],
]

RECOMMENDATIONS = {
    "critical": [
        "**Immediate action required** — high-frequency rat activity detected.",
        "Contact a licensed pest control service within 24 hours.",
        "Restrict access to affected areas until treated.",
        "Secure all food and waste storage immediately.",
        "Deploy snap traps and bait stations in all high-activity zones.",
    ],
    "high": [
        "**Significant activity detected** — prompt action recommended.",
        "Inspect entry points (pipes, gaps in walls/floors) in flagged zones.",
        "Place traps and bait stations in high-activity zones.",
        "Ensure all food is sealed and waste bins are covered.",
        "Schedule a professional inspection within 48 hours.",
    ],
    "medium": [
        "**Moderate activity detected** — monitor closely.",
        "Check for entry points and seal gaps in affected zones.",
        "Set preventive traps in detected zones.",
        "Improve sanitation and remove potential food sources.",
        "Reassess within 1 week.",
    ],
    "low": [
        "**Low activity** — continue routine monitoring.",
        "Periodic inspections recommended.",
        "Maintain current sanitation standards.",
        "Re-run detection in 2 weeks.",
    ],
}

def load_meta(mk, vk):
    p = DATA / mk / vk / "detections.json"
    return json.loads(p.read_text()) if p.exists() else None

def load_video_path(mk, vk):
    p = DATA / mk / vk / "annotated.mp4"
    return p if p.exists() else None

def build_heatmap(dets, W, H, sigma=40):
    heat = np.zeros((H, W), dtype=np.float32)
    for d in dets:
        px = int(np.clip(d["cx"]*W, 0, W-1))
        py = int(np.clip(d["cy"]*H, 0, H-1))
        heat[py, px] += 1.0
    heat = gaussian_filter(heat, sigma=sigma)
    if heat.max() > 0: heat /= heat.max()
    return heat

def overlay_heatmap(frame_bgr, heat):
    heat_u8 = (heat*255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    mask = (heat > 0.05).astype(np.float32)[:,:,None]
    return (frame_rgb*(1-0.55*mask) + colored_rgb*0.55*mask).astype(np.uint8)

def get_mid_frame(vp):
    cap = cv2.VideoCapture(str(vp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//2)
    ret, frame = cap.read(); cap.release()
    return frame if ret else None

def severity(n, nf, af):
    if nf == 0: return "low"
    r = af/nf
    if n>200 or r>0.6: return "critical"
    if n>80  or r>0.3: return "high"
    if n>20  or r>0.1: return "medium"
    return "low"

def zone_analysis(dets, rows=3, cols=3):
    counts = np.zeros((rows, cols), dtype=int)
    for d in dets:
        counts[min(int(d["cy"]*rows),rows-1), min(int(d["cx"]*cols),cols-1)] += 1
    return counts

st.set_page_config(page_title="Rat Detection Demo", layout="wide")
st.markdown("""<style>
  body, [data-testid="stAppViewContainer"] { background:#1f91a8; color:#ffffff; }
  [data-testid="stSidebar"] { background:#ffffff; }
  [data-testid="stSidebar"] * { color:#1f91a8 !important; }
  h1,h2,h3 { color:#ffffff; }
  .stTabs [data-baseweb="tab"] { color:#ffffff; }
  .stTabs [aria-selected="true"] { border-bottom:2px solid #ffffff !important; color:#ffffff !important; }
  .metric-row { display:flex; gap:12px; margin-bottom:16px; }
  .mcard { flex:1; background:#ffffff; border-radius:12px; padding:16px; text-align:center; }
  .mval  { font-size:2rem; font-weight:700; color:#1f91a8; }
  .mlbl  { font-size:0.8rem; color:#1f91a8; opacity:0.85; }
  .rec   { background:rgba(255,255,255,0.15); border-left:4px solid #ffffff; border-radius:8px; padding:12px 16px; margin:6px 0; color:#ffffff; }
  video  { max-height:70vh; width:auto !important; margin:0 auto; display:block; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.divider()
    avail_models = sorted([d.name for d in DATA.iterdir() if d.is_dir()]) if DATA.exists() else []
    avail_videos = sorted(set(v.name for m in DATA.glob("*/") for v in m.iterdir() if v.is_dir())) if DATA.exists() else []
    if not avail_models:
        st.error("No precomputed data found in `demo_data/`."); st.stop()
    model_key = st.selectbox("Model", avail_models, format_func=lambda k: MODEL_LABELS.get(k, k))
    video_key = st.selectbox("Video", avail_videos, format_func=lambda k: VIDEO_LABELS.get(k, k))
    sigma = st.slider("Heatmap smoothing", 10, 100, 40, 5)
    st.divider()
    st.caption("Inference precomputed on A100 GPU. No model weights needed.")

meta = load_meta(model_key, video_key)
vid_path = load_video_path(model_key, video_key)
if meta is None or vid_path is None:
    st.error(f"No data for **{model_key} / {video_key}**."); st.stop()

dets = meta["detections"]
fps, W, H = meta["fps"], meta["width"], meta["height"]
n_frames = meta["total_frames"]
active_frames = len(set(d["frame"] for d in dets))
sev = severity(len(dets), n_frames, active_frames)
sev_label = {"low":"Low","medium":"Moderate","high":"High","critical":"Critical"}[sev]

st.title("Analytics Dashboard")
st.markdown(f"""<div class="metric-row">
  <div class="mcard"><div class="mval">{sev_label}</div><div class="mlbl">Severity</div></div>
</div>""", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Annotated Video", "Heatmap", "Timeline", "Recommendations"])

with tab1:
    # centre portrait videos; cap landscape videos at a sensible height
    is_portrait = H > W
    if is_portrait:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.video(str(vid_path))
    else:
        st.video(str(vid_path))

with tab2:
    if dets:
        heat = build_heatmap(dets, W, H, sigma)
        mid = get_mid_frame(vid_path)
        if mid is not None:
            st.caption("Heatmap overlay")
            st.image(overlay_heatmap(mid, heat), use_container_width=True)
    else:
        st.info("No detections — heatmap is empty.")

with tab3:
    if dets:
        times = [d["frame"]/max(fps,1) for d in dets]
        hist, edges = np.histogram(times, bins=max(int(max(times))+1,1))
        st.subheader("Detections per second")
        st.bar_chart(pd.DataFrame({"Time (s)":edges[:-1].round(1),"Detections":hist}).set_index("Time (s)"))
        st.subheader("Confidence over time")
        st.line_chart(pd.DataFrame({"Frame":[d["frame"] for d in dets],"Confidence":[d["conf"] for d in dets]}).set_index("Frame"))
    else:
        st.info("No detections to chart.")

with tab4:
    st.subheader(f"Recommendations — {sev_label} severity")
    for rec in RECOMMENDATIONS[sev]:
        st.markdown(f'<div class="rec">{rec}</div>', unsafe_allow_html=True)
    if dets:
        pk = int(np.argmax(np.histogram([d["frame"]/fps for d in dets], bins=max(int(n_frames/fps),1))[0]))
        st.markdown(f'<div class="rec"><b>Peak activity</b> at ~{pk}s — review this window carefully.</div>', unsafe_allow_html=True)
