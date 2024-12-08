import tkinter as tk
from collections import Counter
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk
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
def detect_and_display():
    # Chọn tệp ảnh
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return  # Thoát nếu không chọn tệp

    # Tiền xử lý ảnh
    processed_img = preprocess_image(file_path)

    # Lưu ảnh đã tiền xử lý tạm thời
    temp_path = "temp_processed_image.jpg"
    cv2.imwrite(temp_path, processed_img)

    # Thực hiện phát hiện đối tượng
    results = model(temp_path)

    # Tạo bộ đếm để đếm số lượng từng loại trái cây
    fruit_counter = Counter()

    for result in results:
        # Lấy ảnh gốc từ kết quả phát hiện
        img = result.orig_img

        # Kiểm tra nếu có bounding box
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ góc trên trái và dưới phải
                conf = float(box.conf[0])  # Độ tin cậy của dự đoán
                cls = int(box.cls[0])  # Lớp dự đoán

                # Tăng bộ đếm loại trái cây
                fruit_name = model.names[cls]
                fruit_counter[fruit_name] += 1

                # Vẽ hình chữ nhật và ghi nhãn
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{fruit_name} {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị ảnh đã xử lý trên giao diện
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # Lưu tham chiếu để tránh bị xóa

    # Cập nhật thông tin kết quả lên giao diện
    for widget in result_frame.winfo_children():
        widget.destroy()  # Xóa các widget cũ trong frame kết quả

    tk.Label(result_frame, text="Detection Results:", font=("Arial", 14, "bold")).pack(anchor="w")
    for fruit, count in fruit_counter.items():
        tk.Label(result_frame, text=f"{fruit}: {count}", font=("Arial", 12)).pack(anchor="w")

# Tạo giao diện người dùng
root = tk.Tk()
root.title("Fruit Detection")

# Khung chứa ảnh
canvas = tk.Canvas(root, width=640, height=640)
canvas.pack(pady=10)

# Nút chọn ảnh
btn_select_image = tk.Button(root, text="Select Image", command=detect_and_display, font=("Arial", 14))
btn_select_image.pack(pady=10)

# Frame hiển thị kết quả
result_frame = tk.Frame(root)
result_frame.pack(pady=10, padx=10)

# Khởi tạo mô hình YOLO
model_path = './runs/best.pt'
model = YOLO(model_path)

# Chạy giao diện người dùng
root.mainloop()
