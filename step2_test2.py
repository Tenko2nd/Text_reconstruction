import cv2
import numpy as np
import os
from glob import glob

# === Paramètres ===
input_folder = "cropped_images"
output_folder = "lettres_sep_mix_2"
resize_width = 600
morph_open_kernel = (3, 2) # 3,3
morph_close_kernel = (1, 1)
min_contour_area = 200
min_letter_size = (5, 10)
letter_margin = 3
output_size = 28
border_check_thickness = 5
border_white_ratio_thresh = 0.7

os.makedirs(output_folder, exist_ok=True)


def is_border_mostly_white(thresh, thickness=5, ratio_thresh=0.7):
    h, w = thresh.shape
    borders = np.concatenate([
        thresh[:thickness, :].flatten(),
        thresh[-thickness:, :].flatten(),
        thresh[:, :thickness].flatten(),
        thresh[:, -thickness:].flatten()
    ])
    white_ratio = np.mean(borders == 255)
    return white_ratio > ratio_thresh


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarisation adaptative
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, bin_reg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bin_img = bin_inv if np.mean(gray) > 180 else bin_reg

    # Inversion si bord blanc
    if is_border_mostly_white(bin_img, thickness=border_check_thickness, ratio_thresh=border_white_ratio_thresh):
        bin_img = cv2.bitwise_not(bin_img)

    # Morpho
    if morph_close_kernel:
        kernel = np.ones(morph_close_kernel, np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    if morph_open_kernel:
        kernel = np.ones(morph_open_kernel, np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    return bin_img


def get_letter_boxes(binary_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    boxes = [
        tuple(stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1])
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_contour_area
    ]
    return boxes


def group_by_lines(boxes, max_line_gap_ratio=0.7):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[1])
    median_height = np.median([b[3] for b in boxes])
    max_line_gap = max(10, int(median_height * max_line_gap_ratio))
    lines, current_line = [], [boxes[0]]

    for box in boxes[1:]:
        center_y = box[1] + box[3] / 2
        ref_center = np.mean([b[1] + b[3] / 2 for b in current_line])
        if abs(center_y - ref_center) < max_line_gap:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]

    if current_line:
        lines.append(sorted(current_line, key=lambda b: b[0]))
    return lines


def extract_and_save_letters(binary_img, lines, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    index = 0

    for line in lines:
        for (x, y, w, h) in line:
            if w < min_letter_size[0] or h < min_letter_size[1]:
                continue
            x1, y1 = max(0, x - letter_margin), max(0, y - letter_margin)
            x2, y2 = min(binary_img.shape[1], x + w + letter_margin), min(binary_img.shape[0], y + h + letter_margin)
            crop = binary_img[y1:y2, x1:x2]

            if cv2.countNonZero(crop) == 0:
                continue

            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                mask = np.zeros_like(crop)
                cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
                final = cv2.bitwise_and(crop, mask)
            else:
                final = crop

            if cv2.countNonZero(final) > 0:
                # Resize + padding
                h_crop, w_crop = final.shape
                scale = output_size / max(w_crop, h_crop)
                resized = cv2.resize(final, (int(w_crop * scale), int(h_crop * scale)), interpolation=cv2.INTER_NEAREST)
                pad_y = output_size - resized.shape[0]
                pad_x = output_size - resized.shape[1]
                padded = cv2.copyMakeBorder(resized, pad_y//2, pad_y - pad_y//2, pad_x//2, pad_x - pad_x//2,
                                            cv2.BORDER_CONSTANT, value=0)

                cv2.imwrite(os.path.join(output_dir, f"lettre_{index:03}.png"), padded)
                index += 1
    return index


# === Traitement principal ===
image_paths = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in glob(os.path.join(input_folder, ext))])

for img_path in image_paths:
    print(f"Traitement de : {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("  Erreur de lecture.")
        continue

    if resize_width > 0 and img.shape[1] > 0:
        scale = resize_width / img.shape[1]
        img = cv2.resize(img, (resize_width, int(img.shape[0] * scale)), interpolation=cv2.INTER_LANCZOS4)

    binary = preprocess_image(img)
    boxes = get_letter_boxes(binary)
    lines = group_by_lines(boxes)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(output_folder, img_name)

    count = extract_and_save_letters(binary, lines, save_dir)
    print(f"  {count} lettres extraites dans {save_dir}")