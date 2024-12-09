from collections import Counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)

    # Thay đổi kích thước ảnh về 640x640
    img_resized = cv2.resize(img, (640, 640))

    # Áp dụng Gaussian Blur để giảm nhiễu
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)

    return img_blur

# Hàm phát hiện và hiển thị kết quả
def detect_and_display(image_file):
    # Đọc ảnh từ Streamlit uploader (PIL)
    image = Image.open(image_file)

    # Chuyển đổi ảnh PIL sang numpy array (OpenCV yêu cầu)
    img_np = np.array(image)

    # Chuyển từ RGB sang BGR (OpenCV yêu cầu BGR)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Lưu ảnh tạm thời
    temp_path = "temp/temp_uploaded_image.jpg"
    cv2.imwrite(temp_path, img_bgr)

    # Tiền xử lý ảnh
    processed_img = preprocess_image(temp_path)

    # Lưu ảnh đã tiền xử lý tạm thời
    temp_path_processed = "temp/temp_processed_image.jpg"
    cv2.imwrite(temp_path_processed, processed_img)

    # Thực hiện phát hiện đối tượng
    results = model(temp_path_processed)

    # Tạo bộ đếm để đếm số lượng từng loại trái cây
    fruit_counter = Counter()

    for result in results:
        # Lấy ảnh gốc từ kết quả phát hiện
        img = result.orig_img

        # Kiểm tra nếu có bounding box
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf[0])  # Độ tin cậy của dự đoán
                if conf < 0.4:
                    continue  # Bỏ qua các dự đoán có độ tin cậy < 0.4

                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ góc trên trái và dưới phải
                cls = int(box.cls[0])  # Lớp dự đoán

                # Tăng bộ đếm loại trái cây
                fruit_name = model.names[cls]
                fruit_counter[fruit_name] += 1

                # Vẽ hình chữ nhật và ghi nhãn
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{fruit_name} {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Chuyển ảnh OpenCV sang PIL để hiển thị trong Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Hiển thị ảnh đầu vào và ảnh kết quả
    st.subheader("Input Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Processed Image with Detection Results")
    st.image(img_pil, caption="Detected Fruits", use_column_width=True)

    # Hiển thị kết quả đếm trái cây
    st.subheader("Detection Results:")
    for fruit, count in fruit_counter.items():
        st.write(f"{fruit}: {count}")

# Tạo giao diện Streamlit
st.title("Fruit Detection with YOLO11")

# Upload ảnh từ người dùng
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Kiểm tra nếu người dùng đã tải lên ảnh
if image_file is not None:
    # Khởi tạo mô hình YOLO
    model_path = 'yolo11x.pt'  # Đảm bảo đường dẫn chính xác
    model = YOLO(model_path)

    # Phát hiện và hiển thị kết quả
    detect_and_display(image_file)
