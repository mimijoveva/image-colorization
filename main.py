import cv2
import numpy as np
import os
import time

"""
   Vasilka Jugova 231084 i Mimi Joveva 233078

1. Postavete sliki vo GrayScaleImages (formati: .jpg, .png ili .jpeg),
   dokolku papkata e prazna ke se pojavi prozorce za izbiranje slika od vasiot ured.
   (kako primer imame postaveno 15sliki koi se obraboteni so dvata algoritmi)
2. Osigurajte se deka gi imate .prototxt, .caffemodel i .npy fajlovite vo Models.
3. Vo PyCharm (ili terminal) startuvaj go ovoj fajl: "Boenje na crno-beli sliki - Image Colorization.py"
"""

input_folder = "./GrayScaleImages"
output_folder_lab = "./ColorizedImages"
output_folder_dl = "./ColorizedImagesDl"
before_after_folder = "./beforeAndAfter"
logo_path = "assets/logo.png"

prototxt_path = "Models/colorization_deploy_v2.prototxt"
model_path = "Models/colorization_release_v2.caffemodel"
pts_npy_path = "Models/pts_in_hull.npy"

if not os.path.exists(output_folder_lab):
    os.makedirs(output_folder_lab)
if not os.path.exists(output_folder_dl):
    os.makedirs(output_folder_dl)
if not os.path.exists(before_after_folder):
    os.makedirs(before_after_folder)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if len(image_files) == 0:
    print(" Папката GrayScaleImages е празна. Ќе се отвори прозорец за избор на слика...")

    import tkinter as tk
    from tkinter import filedialog

    def choose_image():
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Избери grayscale слика",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        return file_path

    selected_image = choose_image()

    if not selected_image:
        print("Не избравте слика. Програмата се прекинува.")
        exit()

    image_files = [os.path.basename(selected_image)]
    input_folder = os.path.dirname(selected_image)

print("ИЗБЕРИ МЕТОД ЗА БОЕЊЕ ЦРНО-БЕЛИ СЛИКИ - IMAGE COLORIZATION")
print("1 - Традиционален метод (LAB + CLAHE + кластерни бои) - уметничко боење")
print("2 - Advanced метод со Deep Learning (.caffemodel) - реалистично боење")

metod = input("Внесете 1 или 2: ")

def save_before_after(gray_img, color_img, file_name, method_name):
    fixed_height = 400
    aspect_ratio_gray = gray_img.shape[1] / gray_img.shape[0]
    aspect_ratio_color = color_img.shape[1] / color_img.shape[0]

    resized_gray = cv2.resize(gray_img, (int(fixed_height * aspect_ratio_gray), fixed_height))
    resized_color = cv2.resize(color_img, (int(fixed_height * aspect_ratio_color), fixed_height))

    if len(resized_gray.shape) == 2:
        gray_bgr = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)
    else:
        gray_bgr = resized_gray

    combined = np.hstack((gray_bgr, resized_color))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Before", (30, 40), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "After", (gray_bgr.shape[1] + 30, 40), font, 1, (255, 255, 255), 2)

    if os.path.exists(logo_path):
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        scale_percent = 8
        logo_w = int(combined.shape[1] * scale_percent / 100)
        logo_h = int(logo.shape[0] * (logo_w / logo.shape[1]))
        logo = cv2.resize(logo, (logo_w, logo_h))

        y_offset = combined.shape[0] - logo_h - 10
        x_offset = combined.shape[1] - logo_w - 10

        if logo.shape[2] == 4:
            alpha_s = logo[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                combined[y_offset:y_offset+logo_h, x_offset:x_offset+logo_w, c] = (
                    alpha_s * logo[:, :, c] +
                    alpha_l * combined[y_offset:y_offset+logo_h, x_offset:x_offset+logo_w, c]
                )
        else:
            combined[y_offset:y_offset+logo_h, x_offset:x_offset+logo_w] = logo

    output_name = f"before_after_{method_name}_{file_name}"
    output_path = os.path.join(before_after_folder, output_name)
    cv2.imwrite(output_path, combined)

def lab_clahe_colorization():
    print(" Се стартува уметничко боење со LAB + CLAHE + кластерни бои...")
    start_time = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    color_clusters = [(140, 145), (150, 160), (120, 140)]

    for file_name in image_files:
        print(f"🎨 Боење на: {file_name}")
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder_lab, f"colorized_lab_{file_name}")

        gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            print(f" Сликата {file_name} не може да се вчита.")
            continue

        enhanced = clahe.apply(gray_img)
        bgr_base = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        lab_img = cv2.cvtColor(bgr_base, cv2.COLOR_BGR2LAB)
        mask = gray_img > 30
        L = lab_img[:, :, 0] / 255.0
        A = np.zeros_like(L)
        B = np.zeros_like(L)

        for i, (a_val, b_val) in enumerate(color_clusters):
            weight = np.clip(1.0 - abs(L - i / 2.0), 0, 1)
            A += weight * a_val
            B += weight * b_val

        lab_img[:, :, 1] = np.where(mask, A, 128)
        lab_img[:, :, 2] = np.where(mask, B, 128)

        colorized_img = cv2.cvtColor(lab_img.astype("uint8"), cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path, colorized_img)
        save_before_after(gray_img, colorized_img, file_name, "lab")

    print(" LAB + CLAHE боeњето успешно заврши.")
    print(f" Вкупно време: {round(time.time() - start_time, 2)}s")

def deep_learning_colorization():
    print("▶️ Се стартува боење со Deep Learning методот...")
    start_time = time.time()

    if not os.path.exists(prototxt_path) or not os.path.exists(model_path) or not os.path.exists(pts_npy_path):
        print(" Недостасуваат модел фајлови во 'Models' фолдерот.")
        return

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts_in_hull = np.load(pts_npy_path)
    pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    for file_name in image_files:
        print(f"🎨 Боење на: {file_name}")
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder_dl, f"colorized_dl_{file_name}")

        bgr = cv2.imread(input_path)
        if bgr is None:
            print(f" Сликата {file_name} не може да се вчита.")
            continue

        h, w = bgr.shape[:2]
        img_rgb = bgr.astype("float32") / 255.
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2Lab)
        l_channel = img_lab[:, :, 0]
        img_rs = cv2.resize(l_channel, (224, 224)) - 50

        net.setInput(cv2.dnn.blobFromImage(img_rs))
        ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_dec_us = cv2.resize(ab_dec, (w, h))

        lab_out = np.concatenate((l_channel[:, :, np.newaxis], ab_dec_us), axis=2)
        bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
        bgr_out = np.clip(bgr_out, 0, 1)
        bgr_out = (255 * bgr_out).astype("uint8")

        cv2.imwrite(output_path, bgr_out)
        gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        save_before_after(gray_img, bgr_out, file_name, "dl")

    print("Deep Learning боeњето успешно заврши.")
    print(f"⏱ Вкупно време: {round(time.time() - start_time, 2)}s")

while True:
    if metod == "1":
        lab_clahe_colorization()
        break
    elif metod == "2":
        deep_learning_colorization()
        break
    else:
        print(" Погрешен влез, ве молиме внесете 1 или 2.")
        metod = input("Внесете 1 или 2: ")
