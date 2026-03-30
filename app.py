import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

from src.rule_engine import classify_worker

MODEL_PATH = "models/best.pt"


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def analyze_scene(image_path: str, model: YOLO, conf: float = 0.25):
    results = model.predict(source=image_path, conf=conf, save=False, verbose=False)
    result = results[0]

    person_boxes = []
    helmet_boxes = []
    vest_boxes = []

    for box in result.boxes:
        cls = int(box.cls[0])
        xyxy = [float(x) for x in box.xyxy[0].tolist()]
        label = model.names[cls]

        if label == "person":
            person_boxes.append(xyxy)
        elif label == "helmet":
            helmet_boxes.append(xyxy)
        elif label == "vest":
            vest_boxes.append(xyxy)

    safe_count = 0
    unsafe_count = 0
    worker_reports = []

    for i, person_box in enumerate(person_boxes, start=1):
        status, missing = classify_worker(person_box, helmet_boxes, vest_boxes)

        if status == "safe":
            safe_count += 1
        else:
            unsafe_count += 1

        worker_reports.append(
            {
                "worker_id": i,
                "box": person_box,
                "status": status,
                "missing": missing,
            }
        )

    scene_status = "SAFE" if unsafe_count == 0 else "UNSAFE"

    return {
        "result": result,
        "person_boxes": person_boxes,
        "helmet_boxes": helmet_boxes,
        "vest_boxes": vest_boxes,
        "safe_workers": safe_count,
        "unsafe_workers": unsafe_count,
        "workers_detected": len(person_boxes),
        "scene_status": scene_status,
        "worker_reports": worker_reports,
    }


def create_annotated_image(image_path: str, analysis: dict):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for hb in analysis["helmet_boxes"]:
        draw_box(img, hb, (255, 255, 0), "helmet")

    for vb in analysis["vest_boxes"]:
        draw_box(img, vb, (0, 255, 255), "vest")

    for worker in analysis["worker_reports"]:
        color = (0, 180, 0) if worker["status"] == "safe" else (0, 0, 255)
        label = f"worker {worker['worker_id']}: {worker['status']}"
        if worker["missing"]:
            label += f" ({', '.join(worker['missing'])} missing)"
        draw_box(img, worker["box"], color, label)

    scene_color = (0, 180, 0) if analysis["scene_status"] == "SAFE" else (0, 0, 255)
    cv2.putText(
        img,
        f"Scene: {analysis['scene_status']}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        scene_color,
        3,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    st.set_page_config(page_title="Construction Safety Monitor", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for enhanced styling
    st.markdown("""
        <style>
            /* Main styling */
            .main {
                padding: 0px;
            }
            
            /* Header styling */
            .header-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                color: white;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }
            
            .header-title {
                font-size: 2.5em;
                font-weight: 700;
                margin: 0;
                padding: 0;
            }
            
            .header-subtitle {
                font-size: 1.1em;
                margin-top: 10px;
                opacity: 0.95;
            }
            
            /* Card styling */
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
                margin-bottom: 15px;
            }
            
            .metric-card.safe {
                border-left-color: #2ecc71;
            }
            
            .metric-card.unsafe {
                border-left-color: #e74c3c;
            }
            
            /* Status badge */
            .status-badge {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.95em;
            }
            
            .status-safe {
                background-color: #d4edda;
                color: #155724;
            }
            
            .status-unsafe {
                background-color: #f8d7da;
                color: #721c24;
            }
            
            /* Worker report styling */
            .worker-item {
                background: #f8f9fa;
                padding: 12px 15px;
                border-radius: 6px;
                margin-bottom: 10px;
                border-left: 3px solid #667eea;
            }
            
            .worker-item.safe {
                border-left-color: #2ecc71;
            }
            
            .worker-item.unsafe {
                border-left-color: #e74c3c;
            }
            
            /* Stats container */
            .stats-container {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 25px;
            }
            
            .stat-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            
            .stat-box.safe {
                background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            }
            
            .stat-box.unsafe {
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            }
            
            .stat-value {
                font-size: 2em;
                font-weight: 700;
                margin: 10px 0;
            }
            
            .stat-label {
                font-size: 0.95em;
                opacity: 0.95;
            }
            
            /* Upload area */
            .upload-section {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 8px;
                border: 2px dashed #667eea;
                margin-bottom: 20px;
            }
            
            /* Section headers */
            .section-header {
                font-size: 1.4em;
                font-weight: 600;
                margin-top: 25px;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
                color: #2c3e50;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🏗️ Construction Safety Monitor</h1>
            <p class="header-subtitle">AI-Powered Worker Safety Detection System</p>
        </div>
    """, unsafe_allow_html=True)

    if not Path(MODEL_PATH).exists():
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()

    model = load_model(MODEL_PATH)

    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Configuration")
    conf = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05, help="Adjust detection sensitivity")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Instructions")
    st.sidebar.info("""
    1. Upload a construction site image
    2. System analyzes for workers, helmets, and vests
    3. Get instant safety assessment
    4. Review worker-by-worker reports
    """)

    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📸 Upload a construction site image", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #7f8c8d;">
                <p style="font-size: 1.2em;">📁 Ready to analyze a construction site image?</p>
                <p>Upload an image to get started with safety assessment</p>
            </div>
        """, unsafe_allow_html=True)
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_image_path = tmp.name

    try:
        with st.spinner("🔍 Analyzing image..."):
            analysis = analyze_scene(temp_image_path, model, conf=conf)
            
            # Validation: Check if any workers were detected
            if analysis["workers_detected"] == 0:
                st.error("""
                    ❌ **No Workers Detected**
                    
                    The uploaded image does not appear to be a construction site image with workers.
                    
                    Please ensure:
                    • The image contains people/workers
                    • The image is from a construction or industrial site
                    • The image quality is sufficient for detection
                    
                    Try uploading a different image.
                """)
                return
            
            annotated = create_annotated_image(temp_image_path, analysis)

        # Main content layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<h2 class="section-header">🎯 Detection Results</h2>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)

        with col2:
            # Scene status banner
            scene_status = analysis["scene_status"]
            is_safe = scene_status == "SAFE"
            
            status_color = "#2ecc71" if is_safe else "#e74c3c"
            status_emoji = "✅" if is_safe else "⚠️"
            
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, {status_color} 0%, {status_color}dd 100%); 
                            color: white; padding: 20px; border-radius: 8px; text-align: center; 
                            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15); margin-bottom: 20px;">
                    <p style="font-size: 1.1em; margin: 0; opacity: 0.9;">Scene Status</p>
                    <p style="font-size: 2.5em; font-weight: 700; margin: 10px 0 0 0;">{status_emoji} {scene_status}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Stats
            st.markdown("""
                <div class="stats-container">
            """, unsafe_allow_html=True)
            
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-label">👥 Detected</div>
                        <div class="stat-value">{analysis['workers_detected']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_w2:
                st.markdown(f"""
                    <div class="stat-box safe">
                        <div class="stat-label">✅ Safe</div>
                        <div class="stat-value">{analysis['safe_workers']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_w3:
                st.markdown(f"""
                    <div class="stat-box unsafe">
                        <div class="stat-label">⚠️ Unsafe</div>
                        <div class="stat-value">{analysis['unsafe_workers']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Safety percentage
            if analysis["workers_detected"] > 0:
                safety_pct = int((analysis["safe_workers"] / analysis["workers_detected"]) * 100)
            else:
                safety_pct = 0
            
            st.markdown(f"""
                <div style="margin-top: 15px;">
                    <p style="font-size: 0.9em; color: #7f8c8d; margin-bottom: 8px;">Safety Rate</p>
                    <div style="width: 100%; height: 24px; background: #ecf0f1; border-radius: 12px; overflow: hidden;">
                        <div style="width: {safety_pct}%; height: 100%; background: linear-gradient(90deg, #2ecc71, #27ae60); 
                                    display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">
                            {safety_pct}%
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Worker Reports
            st.markdown('<h3 class="section-header" style="margin-top: 20px;">👷 Worker Reports</h3>', unsafe_allow_html=True)
            
            if not analysis["worker_reports"]:
                st.warning("No workers detected in the image.")
            else:
                for worker in analysis["worker_reports"]:
                    is_worker_safe = worker["status"] == "safe"
                    worker_status = "safe" if is_worker_safe else "unsafe"
                    status_emoji = "✅" if is_worker_safe else "⚠️"
                    
                    if is_worker_safe:
                        report_text = f"<strong>Worker {worker['worker_id']}</strong>: {status_emoji} SAFE"
                    else:
                        missing = ", ".join(worker["missing"])
                        report_text = f"<strong>Worker {worker['worker_id']}</strong>: {status_emoji} UNSAFE<br><small style='color: #e74c3c;'>Missing: {missing}</small>"
                    
                    st.markdown(f"""
                        <div class="worker-item {worker_status}">
                            {report_text}
                        </div>
                    """, unsafe_allow_html=True)

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


if __name__ == "__main__":
    main()