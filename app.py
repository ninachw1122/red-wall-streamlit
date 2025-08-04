from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import os
import shutil
from datetime import datetime
import requests

# === 修正 Ultralytics 設定目錄，避免雲端警告 ===
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# === 初始化語言狀態 ===
if "lang" not in st.session_state:
    st.session_state.lang = "zh"  # 預設中文

# === 語言切換按鈕 ===
if st.button("🌐 切換語言 (Switch Language)"):
    st.session_state.lang = "en" if st.session_state.lang == "zh" else "zh"

# 語言字典
TEXT = {
    "zh": {
        "title": "紅牆熱影像處理系統",
        "reset": "🔄 重整並清空結果資料夾",
        "reset_done": "已清空結果資料夾，請重新上傳圖片進行偵測",
        "upload": "上傳兩張圖片",
        "upload_visible": "上傳可見光圖",
        "upload_thermal": "上傳熱影像圖",
        "wall_and_mask": "紅牆偵測與黑色遮罩",
        "wall_detect": "紅牆偵測結果",
        "mask_detect": "黑色遮罩後影像",
        "leak_detect": "滲水偵測與可見光同步結果",
        "leak_result": "滲水偵測結果",
        "visible_result": "可見光滲水結果",
        "download": "📥 下載所有結果",
        "download_done": "結果已儲存至：",
        "no_wall": "未偵測到紅牆，請確認模型或輸入影像",
        "no_leak": "未找到滲水偵測結果，請檢查模型或路徑設定",
        "upload_warn": "請上傳兩張圖片",
        "no_txt": "找不到滲水偵測標註 .txt 檔案"
    },
    "en": {
        "title": "Red Brick Wall Thermal Image Processing System",
        "reset": "🔄 Reset and Clear Results Folder",
        "reset_done": "Results folder cleared. Please re-upload images for detection.",
        "upload": "Upload Two Images",
        "upload_visible": "Upload Visible Image",
        "upload_thermal": "Upload Thermal Image",
        "wall_and_mask": "Wall Detection and Black Mask",
        "wall_detect": "Wall Detection Result",
        "mask_detect": "Black Masked Image",
        "leak_detect": "Leak Detection and Visible Image Result",
        "leak_result": "Leak Detection Result",
        "visible_result": "Visible Leak Result",
        "download": "📥 Download All Results",
        "download_done": "Results saved to: ",
        "no_wall": "No wall detected. Please check the model or input image.",
        "no_leak": "No leak detection result found. Please check the model or path settings.",
        "upload_warn": "Please upload two images",
        "no_txt": "Leak detection annotation .txt file not found"
    }
}

# === 標題 ===
st.title(TEXT[st.session_state.lang]["title"])

# === 清空資料夾 ===
def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# === 重整按鈕 ===
if st.button(TEXT[st.session_state.lang]["reset"]):
    clear_dir(os.path.join("runs", "predict_filtered"))
    clear_dir(os.path.join("runs", "predict_masked"))
    clear_dir(os.path.join("runs", "predict_leakage"))
    st.success(TEXT[st.session_state.lang]["reset_done"])
    st.rerun()

# === 自動下載模型 ===
def load_model(model_path, url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        st.write(f"Downloading model: {model_path} ...")
        with open(model_path, "wb") as f:
            f.write(requests.get(url).content)
        st.success(f"Model downloaded: {model_path}")
    return YOLO(model_path)

# 🚀 載入模型 (改成你的模型下載連結)
wall_model = load_model("models/red_wall/best.pt", "https://你的模型網址/red_wall_best.pt")
leak_model = load_model("models/leak/best.pt", "https://你的模型網址/leak_best.pt")

# === 上傳圖片 ===
st.markdown(f"### {TEXT[st.session_state.lang]['upload']}")
col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    visible_img = st.file_uploader(TEXT[st.session_state.lang]["upload_visible"], type=["jpg", "png", "jpeg"], key="visible")
with col_upload2:
    thermal_img = st.file_uploader(TEXT[st.session_state.lang]["upload_thermal"], type=["jpg", "png", "jpeg"], key="thermal")

def apply_canny(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

if visible_img is not None and thermal_img is not None:
    col_show1, col_show2 = st.columns(2)
    with col_show1:
        st.image(visible_img, caption=TEXT[st.session_state.lang]["upload_visible"], use_container_width=True)
    with col_show2:
        st.image(thermal_img, caption=TEXT[st.session_state.lang]["upload_thermal"], use_container_width=True)

    thermal_np = np.array(Image.open(thermal_img).convert("RGB"))[..., ::-1]
    edges = apply_canny(thermal_np)

    canny_path = "canny_result.jpg"
    cv2.imwrite(canny_path, edges)

    # === YOLO 紅牆偵測 ===
    wall_results = wall_model.predict(
        canny_path,
        save=True,
        project="runs",
        name="predict_filtered",
        exist_ok=True
    )

    if len(wall_results) > 0 and len(wall_results[0].boxes.xyxy) > 0:
        boxes = wall_results[0].boxes.xyxy.cpu().numpy()
        thermal_original = np.array(Image.open(thermal_img).convert("RGB"))
        mask = np.zeros_like(thermal_original)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = thermal_original[y1:y2, x1:x2]

        masked_dir = os.path.join("runs", "predict_masked")
        os.makedirs(masked_dir, exist_ok=True)
        masked_path = os.path.join(masked_dir, "masked_result.jpg")
        cv2.imwrite(masked_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

        st.markdown(f"### {TEXT[st.session_state.lang]['wall_and_mask']}")
        col_wall1, col_wall2 = st.columns(2)
        with col_wall1:
            wall_img = wall_results[0].plot()
            st.image(wall_img, caption=TEXT[st.session_state.lang]["wall_detect"], use_container_width=True)
        with col_wall2:
            st.image(mask, caption=TEXT[st.session_state.lang]["mask_detect"], use_container_width=True)

        # === YOLO 滲水偵測 ===
        leak_results = leak_model.predict(
            masked_path,
            save=True,
            save_txt=True,
            project="runs",
            name="predict_leakage",
            exist_ok=True
        )

        leak_output_path = os.path.join("runs", "predict_leakage", "masked_result.jpg")
        leak_label_path = os.path.join("runs", "predict_leakage", "labels", "masked_result.txt")

        st.markdown(f"### {TEXT[st.session_state.lang]['leak_detect']}")
        if os.path.exists(leak_output_path):
            visible_np = np.array(Image.open(visible_img).convert("RGB"))
            h, w = visible_np.shape[:2]
            if os.path.exists(leak_label_path):
                with open(leak_label_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 7:
                        coords = list(map(float, parts[1:]))
                        if len(coords) % 2 == 0:
                            points = np.array([[int(coords[i] * w), int(coords[i + 1] * h)]
                                               for i in range(0, len(coords), 2)], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(visible_np, [points], isClosed=True, color=(255, 0, 0), thickness=2)
                    elif len(parts) == 5:
                        cls, cx, cy, bw, bh = map(float, parts)
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        cv2.rectangle(visible_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

                col_leak1, col_leak2 = st.columns(2)
                with col_leak1:
                    st.image(leak_output_path, caption=TEXT[st.session_state.lang]["leak_result"], use_container_width=True)
                with col_leak2:
                    st.image(visible_np, caption=TEXT[st.session_state.lang]["visible_result"], use_container_width=True)

                if st.button(TEXT[st.session_state.lang]["download"]):
                    output_dir = "outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_dir = os.path.join(output_dir, f"result_{timestamp}")
                    os.makedirs(result_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(result_dir, "wall_detect.jpg"), wall_img)
                    cv2.imwrite(os.path.join(result_dir, "mask_result.jpg"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
                    shutil.copy(leak_output_path, os.path.join(result_dir, "leak_detect.jpg"))
                    cv2.imwrite(os.path.join(result_dir, "visible_leak.jpg"), cv2.cvtColor(visible_np, cv2.COLOR_RGB2BGR))
                    st.success(f"{TEXT[st.session_state.lang]['download_done']}{result_dir}")
            else:
                st.warning(TEXT[st.session_state.lang]["no_txt"])
        else:
            st.error(TEXT[st.session_state.lang]["no_leak"])
    else:
        st.warning(TEXT[st.session_state.lang]["no_wall"])
else:
    st.warning(TEXT[st.session_state.lang]["upload_warn"])
