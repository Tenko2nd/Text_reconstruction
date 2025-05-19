import cv2
import numpy as np
import os
from glob import glob

def group_by_lines(boxes, max_line_gap=20):
    lines = []
    boxes = sorted(boxes, key=lambda b: b[1])  # Trier verticalement

    for box in boxes:
        x, y, w, h = box
        inserted = False
        for line in lines:
            if abs(line[0][1] - y) < max_line_gap:
                line.append(box)
                inserted = True
                break
        if not inserted:
            lines.append([box])

    for line in lines:
        line.sort(key=lambda b: b[0])  # Trier horizontalement

    return lines

# === Préparation des dossiers ===
input_folder = "cropped_images"
output_folder = "lettres_sep"
os.makedirs(output_folder, exist_ok=True)

# === Récupération de toutes les images ===
image_paths = sorted(glob(os.path.join(input_folder, "*.*")))  # *.png, *.jpg, etc.

for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Impossible de lire {image_path}")
        continue

    # Redimensionnement proportionnel
    original_height, original_width = img.shape[:2]
    new_width = 500
    new_height = int((new_width / original_width) * original_height)
    img = cv2.resize(img, (new_width, new_height))

    # Prétraitement
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # Suppression des petits objets
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = 50
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    closed_copy = cleaned.copy()
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    lines = group_by_lines(bounding_boxes)

    # Création d'un sous-dossier spécifique
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_folder, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Extraction lettres
    margin = 3
    index = 0
    for line in lines:
        for (x, y, w, h) in line:
            if w > 5 and h > 10:
                x1 = max(x - margin, 0)
                y1 = max(y - margin, 0)
                x2 = min(x + w + margin, closed_copy.shape[1])
                y2 = min(y + h + margin, closed_copy.shape[0])
                img_crop = closed_copy[y1:y2, x1:x2].copy()

                # Garder uniquement le plus grand composant
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_crop, connectivity=8)
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    cleaned_crop = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                else:
                    cleaned_crop = img_crop

                cv2.imwrite(os.path.join(image_output_dir, f"lettre_{index:03}.png"), cleaned_crop)
                index += 1

    print(f"Lettres extraites pour : {image_path}")

# Optionnel : ne pas afficher pour chaque image (commenté)
# cv2.imshow("Dernier nettoyage", cleaned)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
