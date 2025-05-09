import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.h5")

# Cài đặt cửa sổ
window = tk.Tk()
window.title("Vẽ số và dự đoán")

canvas_width = 200
canvas_height = 200

# Vùng vẽ
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.grid(row=0, column=0, rowspan=10, padx=10, pady=10)

image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

# Các label để hiển thị kết quả
prediction_labels = []
percent_labels = []

# Layout mới: mỗi hàng 3 số, số 0 nằm hàng cuối
positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
for idx, digit in enumerate(positions):
    row = idx // 3
    col = idx % 3 + 1

    # Ô số
    lbl = tk.Label(window, text=str(digit), width=6, height=3, font=('Arial', 16), relief='groove')
    lbl.grid(row=row*2, column=col, padx=5, pady=2)
    prediction_labels.append(lbl)

    # Nhãn phần trăm
    percent_lbl = tk.Label(window, text="", font=('Arial', 10))
    percent_lbl.grid(row=row*2+1, column=col)
    percent_labels.append(percent_lbl)

def predict_digit():
    # Resize và xử lý ảnh
    img = image.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    predictions = model.predict(img)[0]

    for idx, digit in enumerate(positions):
        p = predictions[digit]
        intensity = int(p * 255)
        green_hex = f'#{0:02x}{intensity:02x}{0:02x}'
        prediction_labels[idx].config(bg=green_hex, fg="white" if p > 0.5 else "black")
        percent_labels[idx].config(text=f"{p*100:.1f}%")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    for lbl in prediction_labels:
        lbl.config(bg="SystemButtonFace", text=lbl.cget("text"), fg="black")
    for plbl in percent_labels:
        plbl.config(text="")

btn_predict = tk.Button(window, text="Dự đoán", command=predict_digit)
btn_predict.grid(row=8, column=0, pady=10)

btn_clear = tk.Button(window, text="Xóa", command=clear_canvas)
btn_clear.grid(row=9, column=0, pady=5)

window.mainloop()
