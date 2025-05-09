# 🧠 Ứng dụng nhận dạng chữ số viết tay 

Đây là một ứng dụng python đơn giản dùng để vẽ số bằng chuột và dự đoán số đó bằng mô hình học sâu (model.h5) huấn luyện từ tập MNIST.

## 🧩 Chức năng chính

- Vẽ số từ 0–9 bằng chuột trong vùng vẽ
- Dự đoán số bằng mô hình `model.h5`
- Hiển thị phần trăm dự đoán theo màu xanh đậm nhạt
- Giao diện đơn giản bằng Tkinter

## 📁 Cấu trúc thư mục

```
.
├── main.py              # Giao diện đồ họa
├── model.py             # Mã huấn luyện mô hình (tuỳ chọn)
├── model.h5             # Mô hình đã huấn luyện
├── requirements.txt     # Các thư viện cần thiết
└── README.md            # Giới thiệu dự án
```

## 🚀 Cách sử dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
python main.py
```

## 🛠 Môi trường đề xuất

- Python 3.8 trở lên
- TensorFlow 2.x

## 📌 Ghi chú

- Mô hình `model.h5` đã được huấn luyện từ tập MNIST.
- Bạn có thể chỉnh sửa hoặc huấn luyện lại trong file `model.py`.

---

## 📸 Giao diện minh họa

![App Screenshot](./screenshot.png)

---

## 📚 Tác giả

Dự án học tập Deep Learning cá nhân của Nguyễn Thế Huy  
🌟 Sinh viên Đại học Bách Khoa Đà Nẵng
