import cv2
import numpy as np
import os
from glob import glob


def group_by_lines(boxes, max_line_gap_ratio=0.7):  # Ratio de la hauteur médiane des boîtes
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])  # Trier verticalement par y1

    # Calculer la hauteur médiane pour un max_line_gap plus adaptatif
    if len(boxes) > 0:
        median_height = np.median([b[3] for b in boxes])
        max_line_gap = int(median_height * max_line_gap_ratio)
        if max_line_gap < 10:  # un minimum absolu
            max_line_gap = 10
    else:
        max_line_gap = 20  # fallback

    lines = []
    if not boxes:  # Vérification supplémentaire après le tri
        return lines

    current_line = [boxes[0]]
    for i in range(1, len(boxes)):
        prev_box_y, prev_box_h = current_line[-1][1], current_line[-1][3]
        current_box_y = boxes[i][1]

        ref_y_line = np.mean([b[1] + b[3] / 2 for b in current_line])
        box_center_y = boxes[i][1] + boxes[i][3] / 2

        if abs(box_center_y - ref_y_line) < max_line_gap or \
                (boxes[i][1] < current_line[0][1] + current_line[0][3] and \
                 abs(boxes[i][1] - current_line[0][1]) < max_line_gap):
            current_line.append(boxes[i])
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))  # Trier la ligne par x
            current_line = [boxes[i]]

    if current_line:  # Ajouter la dernière ligne
        lines.append(sorted(current_line, key=lambda b: b[0]))

    return lines


# === Paramètres Configurables ===
input_folder = "cropped_images"  # Dossier contenant les images originales
output_folder = "lettres_sep_mix"
os.makedirs(output_folder, exist_ok=True)

# Paramètres de prétraitement
resize_width = 600  # Nouvelle largeur cible pour le redimensionnement, 0 pour ne pas redimensionner
morph_open_kernel_size = (3, 3)
morph_close_kernel_size = (3, 3)
min_contour_area = 30
letter_margin = 3
min_letter_width = 5
min_letter_height = 10

# === Récupération de toutes les images ===
image_paths = []
for ext in ("*.png", "*.jpg", "*.jpeg"):
    image_paths.extend(glob(os.path.join(input_folder, ext)))
image_paths = sorted(image_paths)

for image_path in image_paths:
    print(f"Traitement de : {image_path}")
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"  Impossible de lire {image_path}")
        continue

    img = img_original.copy()

    # Redimensionnement proportionnel (si resize_width > 0)
    if resize_width > 0:
        original_height, original_width = img.shape[:2]
        if original_width > 0:  # Eviter division par zero
            new_height = int((resize_width / original_width) * original_height)
            img = cv2.resize(img, (resize_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            print(f"  Largeur originale de l'image {image_path} est 0. Redimensionnement sauté.")
            # Vous pourriez décider d'ignorer cette image ou de la traiter telle quelle

    # Prétraitement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        thresh_val_inv, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh_val_reg, thresh_reg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        thresh = thresh_inv  # Par défaut pour correspondre au succès de script2

        # Testons avec une condition sur la luminosité moyenne pour choisir:
        if np.mean(gray) > 170:  # Image globalement claire -> fond clair, texte sombre probable (de base : 128) - 150 : marche bien
            print(f"  Image globalement claire (moyenne: {np.mean(gray):.2f}), utilisant THRESH_BINARY_INV.")
            thresh = thresh_inv
        else:  # Image globalement sombre -> fond sombre, texte clair probable
            print(f"  Image globalement sombre (moyenne: {np.mean(gray):.2f}), utilisant THRESH_BINARY.")
            thresh = thresh_reg

    except cv2.error as e:
        print(f"  Erreur Otsu sur {image_path}: {e}. Utilisation d'un seuil fixe.")
        # Fallback si Otsu échoue (par exemple, image unicolore)
        if np.mean(gray) > 128:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


    processed_binary = thresh
    if morph_close_kernel_size is not None and morph_close_kernel_size[0] > 0 and morph_close_kernel_size[1] > 0:
        kernel_close = np.ones(morph_close_kernel_size, np.uint8)
        processed_binary = cv2.morphologyEx(processed_binary, cv2.MORPH_CLOSE, kernel_close)

    if morph_open_kernel_size is not None and morph_open_kernel_size[0] > 0 and morph_open_kernel_size[1] > 0:
        kernel_open = np.ones(morph_open_kernel_size, np.uint8)
        processed_binary = cv2.morphologyEx(processed_binary, cv2.MORPH_OPEN, kernel_open)

    # Suppression des petits objets (comme dans script1, mais sur `processed_binary`)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_binary, connectivity=8)

    cleaned_img = np.zeros_like(processed_binary)
    potential_boxes = []
    for i in range(1, num_labels):  # Ignorer le label 0 (fond)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_contour_area:
            cleaned_img[labels == i] = 255
            # Récupérer les bounding boxes directement de stats
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # Filtrage supplémentaire optionnel ici (ex: aspect ratio)
            if w > 0 and h > 0:  # S'assurer que ce ne sont pas des lignes vides
                potential_boxes.append((x, y, w, h))


    # Utiliser les boîtes issues de connectedComponents
    bounding_boxes = potential_boxes


    lines = group_by_lines(bounding_boxes)

    # Création d'un sous-dossier spécifique pour cette image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_folder, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Extraction et sauvegarde des lettres
    index = 0
    for line_idx, line_boxes in enumerate(lines):
        for box_idx, (x, y, w, h) in enumerate(line_boxes):
            # Filtre de taille de base
            if w >= min_letter_width and h >= min_letter_height:
                # Coordonnées avec marge, en s'assurant de ne pas dépasser les bords de `cleaned_img`
                y1_margin = max(0, y - letter_margin)
                y2_margin = min(cleaned_img.shape[0], y + h + letter_margin)
                x1_margin = max(0, x - letter_margin)
                x2_margin = min(cleaned_img.shape[1], x + w + letter_margin)

                img_crop_binary = cleaned_img[y1_margin:y2_margin, x1_margin:x2_margin].copy()


                if img_crop_binary.size == 0 or cv2.countNonZero(img_crop_binary) == 0:
                    continue


                num_labels_crop, labels_crop, stats_crop, _ = cv2.connectedComponentsWithStats(img_crop_binary,
                                                                                               connectivity=8)

                final_letter_segment = np.zeros_like(img_crop_binary)

                if num_labels_crop > 1:  # S'il y a au moins un objet (plus le fond)

                    largest_label_idx = -1
                    max_area = -1

                    contours_crop, _ = cv2.findContours(img_crop_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_crop:
                        largest_contour = max(contours_crop, key=cv2.contourArea)
                        # Créer un masque à partir du plus grand contour
                        mask = np.zeros_like(img_crop_binary)
                        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                        final_letter_segment = cv2.bitwise_and(img_crop_binary, mask)
                    else:  # Pas de contours trouvés, mais le crop n'était pas vide. Peut-être déjà propre.
                        final_letter_segment = img_crop_binary

                else:  # Un seul "objet" (ou vide, déjà géré)
                    final_letter_segment = img_crop_binary

                if cv2.countNonZero(final_letter_segment) > 0:
                    # Redimensionnement avec conservation du ratio
                    target_size = 28
                    h_crop, w_crop = final_letter_segment.shape

                    # Calcul de l'échelle pour que le plus grand côté corresponde à target_size
                    scale = target_size / max(w_crop, h_crop)
                    new_w, new_h = int(w_crop * scale), int(h_crop * scale)
                    resized_letter = cv2.resize(final_letter_segment, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                    # Padding pour obtenir une image 120x120
                    pad_top = (target_size - new_h) // 2
                    pad_bottom = target_size - new_h - pad_top
                    pad_left = (target_size - new_w) // 2
                    pad_right = target_size - new_w - pad_left

                    padded_letter = cv2.copyMakeBorder(resized_letter, pad_top, pad_bottom, pad_left, pad_right,borderType=cv2.BORDER_CONSTANT, value=0)

                    # Sauvegarde de l'image finale 120x120
                    cv2.imwrite(os.path.join(image_output_dir, f"lettre_{index:03}.png"), padded_letter)
                    index += 1
                # else:
                # print(f"  Segment final vide pour box ({x},{y},{w},{h}) dans {image_path}")

    print(f"  {index} lettres extraites pour : {os.path.basename(image_path)} et sauvegardées dans {image_output_dir}")

