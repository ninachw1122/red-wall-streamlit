import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from datetime import datetime
import shutil
import requests
from ultralytics import YOLO
import onnxruntime as ort

# === 修正 Ultralytics 設定目錄，避免雲端警告 ===
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# === 初始化語言狀態 ===
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# === 語言切換 ===
if st.button("🌐 切換語言 (Switch Language)"):
    st.session_state.lang = "en" if st.session_state.lang == "zh" else "zh"

TEXT = {
    "zh": {
        "title": "紅牆熱影像處理系統",
        "mode": "選擇推論模式",
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
        "mode": "Select inference mode",
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

# === 模式選擇 (YOLO / ONNX) ===
mode = st.radio(TEXT[st.session_state.lang]["mode"], ["YOLO", "ONNX"])

# === 清空資料夾 ===
def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

if st.button(TEXT[st.session_state.lang]["reset"]):
    clear_dir("runs/predict_filtered")
    clear_dir("runs/predict_masked")
    clear_dir("runs/predict_leakage")
    st.success(TEXT[st.session_state.lang]["reset_done"])
    st.rerun()

# === 模型載入 ===
def load_model(model_path, url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        st.write(f"Downloading model: {model_path} ...")
        with open(model_path, "wb") as f:
            f.write(requests.get(url).content)
        st.success(f"Model downloaded: {model_path}")
    return YOLO(model_path)

def load_onnx_model(model_path, url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        st.write(f"Downloading ONNX model: {model_path} ...")
        with open(model_path, "wb") as f:
            f.write(requests.get(url).content)
        st.success(f"ONNX model downloaded: {model_path}")
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# === 根據模式載入模型 ===
if mode == "YOLO":
    wall_model = load_model("models/red_wall/best.pt", "https://你的模型網址/red_wall_best.pt")
    leak_model = load_model("models/leak/best.pt", "https://你的模型網址/leak_best.pt")
else:
    wall_model = load_onnx_model("models/red_wall/best.onnx", "https://你的模型網址/red_wall_best.onnx")
    leak_model = load_onnx_model("models/leak/best.onnx", "https://你的模型網址/leak_best.onnx")

# === 上傳圖片 ===
st.markdown(f"### {TEXT[st.session_state.lang]['upload']}")
col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    visible_img = st.file_uploader(TEXT[st.session_state.lang]["upload_visible"], type=["jpg", "png", "jpeg"])
with col_upload2:
    thermal_img = st.file_uploader(TEXT[st.session_state.lang]["upload_thermal"], type=["jpg", "png", "jpeg"])

# === Canny 邊緣檢測 ===
def apply_canny(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 50, 150)

# === ONNX 推論 ===
def run_onnx_inference(session, img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.transpose(2, 0, 1) / 255.0
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    return outputs

# === 主流程 ===
if visible_img and thermal_img:
    col_show1, col_show2 = st.columns(2)
    with col_show1:
        st.image(visible_img, caption=TEXT[st.session_state.lang]["upload_visible"], use_container_width=True)
    with col_show2:
        st.image(thermal_img, caption=TEXT[st.session_state.lang]["upload_thermal"], use_container_width=True)

    thermal_np = np.array(Image.open(thermal_img).convert("RGB"))[..., ::-1]
    edges = apply_canny(thermal_np)
    canny_path = "canny_result.jpg"
    cv2.imwrite(canny_path, edges)

    if mode == "YOLO":
        wall_results = wall_model.predict(canny_path, save=True, project="runs", name="predict_filtered", exist_ok=True)
    else:
        wall_results = run_onnx_inference(wall_model, canny_path)
        st.write("ONNX 紅牆推論結果:", wall_results)

else:
    st.warning(TEXT[st.session_state.lang]["upload_warn"])
