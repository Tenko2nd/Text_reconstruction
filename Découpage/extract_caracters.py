import cv2
import numpy as np
import os
import operator

# --- Paramètres ---
image_path = '../example_images/fith_am.jpeg' # Adaptez ce chemin
# --- MISE A JOUR NOM DOSSIER ---
output_base_dir = 'output_characters_largest_closed'
padding = 5
min_output_width = 25  # Réduit pour l'exemple
min_output_height = 25 # Réduit pour l'exemple
large_area_factor = 2.0
max_recursion_depth = 5
morph_kernel_size = 2 # Utilisé pour l'ouverture ET la fermeture
overlap_threshold_ioa = 0.80
# -------------------------


# --- Créer les dossiers de sortie ---
output_dir_original = os.path.join(output_base_dir, 'original')
# --- MISE A JOUR NOM DOSSIER BINAIRE ---
output_dir_binary = os.path.join(output_base_dir, 'binary_largest_closed')
if not os.path.exists(output_dir_original):
    os.makedirs(output_dir_original)
if not os.path.exists(output_dir_binary):
    os.makedirs(output_dir_binary)

# Liste globale
final_bounding_boxes = []

# --- Fonction de Segmentation (inchangée) ---
def segment_image_recursive(img_segment, offset_x=0, offset_y=0, current_depth=0):
    # (Code de la fonction segment_image_recursive reste identique à la version précédente)
    # ... (coller ici le code de la fonction de la réponse précédente) ...
    global final_bounding_boxes
    h_seg, w_seg = img_segment.shape[:2]

    if current_depth > max_recursion_depth:
        if w_seg >= min_output_width and h_seg >= min_output_height:
             gray_seg = cv2.cvtColor(img_segment, cv2.COLOR_BGR2GRAY) if len(img_segment.shape) == 3 else img_segment
             try: # Ajouter try/except pour Otsu ici aussi
                 _, binary_thresh_seg = cv2.threshold(gray_seg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
             except cv2.error:
                 _, binary_thresh_seg = cv2.threshold(gray_seg, 127, 255, cv2.THRESH_BINARY_INV)

             if morph_kernel_size > 0:
                 kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                 binary_final_seg = cv2.morphologyEx(binary_thresh_seg, cv2.MORPH_OPEN, kernel)
             else:
                 binary_final_seg = binary_thresh_seg
             if cv2.countNonZero(binary_final_seg) > 0:
                 # Calculer l'aire pour la cohérence, même si pas utilisé pour prune ici
                 final_area = w_seg * h_seg
                 final_bounding_boxes.append({'x': offset_x, 'y': offset_y, 'w': w_seg, 'h': h_seg, 'area': final_area, 'binary_crop': binary_final_seg})
        return

    if len(img_segment.shape) == 3: gray_segment = cv2.cvtColor(img_segment, cv2.COLOR_BGR2GRAY)
    else: gray_segment = img_segment

    try: thresh_val, binary_segment_raw = cv2.threshold(gray_segment, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except cv2.error as e: thresh_val, binary_segment_raw = cv2.threshold(gray_segment, 127, 255, cv2.THRESH_BINARY_INV)

    if morph_kernel_size > 0:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        binary_segment_cleaned = cv2.morphologyEx(binary_segment_raw, cv2.MORPH_OPEN, kernel)
    else: binary_segment_cleaned = binary_segment_raw

    contours, _ = cv2.findContours(binary_segment_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if cv2.countNonZero(binary_segment_cleaned) > 0 and w_seg >= min_output_width and h_seg >= min_output_height and current_depth > 0:
             final_area = w_seg * h_seg
             final_bounding_boxes.append({'x': offset_x, 'y': offset_y, 'w': w_seg, 'h': h_seg, 'area': final_area, 'binary_crop': binary_segment_cleaned})
        return

    segment_boxes = []
    total_area = 0
    valid_box_count = 0
    min_contour_area_recursive = 5

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area >= min_contour_area_recursive and w > 1 and h > 1:
            segment_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})
            total_area += area
            valid_box_count += 1

    if valid_box_count == 0:
        if cv2.countNonZero(binary_segment_cleaned) > 0 and w_seg >= min_output_width and h_seg >= min_output_height and current_depth > 0:
             final_area = w_seg * h_seg
             final_bounding_boxes.append({'x': offset_x, 'y': offset_y, 'w': w_seg, 'h': h_seg, 'area': final_area, 'binary_crop': binary_segment_cleaned})
        return

    average_area = total_area / valid_box_count if valid_box_count > 0 else 0

    for box in segment_boxes:
        x, y, w, h, area = box['x'], box['y'], box['w'], box['h'], box['area']
        x_abs = x + offset_x
        y_abs = y + offset_y

        if area > large_area_factor * average_area and current_depth < max_recursion_depth and valid_box_count > 1 and average_area > 0:
            sub_segment = img_segment[y:y+h, x:x+w]
            if sub_segment.size > 0: segment_image_recursive(sub_segment, x_abs, y_abs, current_depth + 1)
        else:
            local_binary_crop_cleaned = binary_segment_cleaned[y:y+h, x:x+w]
            if local_binary_crop_cleaned.size > 0:
                 if cv2.countNonZero(local_binary_crop_cleaned) > 0:
                      final_bounding_boxes.append({'x': x_abs, 'y': y_abs, 'w': w, 'h': h, 'area': area, # Garder l'aire
                                                 'binary_crop': local_binary_crop_cleaned})

# --- Fonction prune_overlapping_boxes (inchangée) ---
def prune_overlapping_boxes(boxes, threshold_ioa):
    # (Code de la fonction prune_overlapping_boxes reste identique)
    # ... (coller ici le code de la fonction de la réponse précédente) ...
    if not boxes: return []
    for i in range(len(boxes)):
        if 'area' not in boxes[i] or boxes[i]['area'] is None or boxes[i]['area'] <= 0: # Ajout vérif area > 0
             boxes[i]['area'] = float(boxes[i]['w'] * boxes[i]['h']) # Assurer float
             if boxes[i]['area'] == 0: boxes[i]['area'] = 1.0 # Eviter division par zero stricte

    num_boxes = len(boxes)
    discarded = [False] * num_boxes
    for i in range(num_boxes):
        if discarded[i]: continue
        for j in range(i + 1, num_boxes):
            if discarded[j]: continue
            box_i = boxes[i]; box_j = boxes[j]
            x_left = max(box_i['x'], box_j['x']); y_top = max(box_i['y'], box_j['y'])
            x_right = min(box_i['x'] + box_i['w'], box_j['x'] + box_j['w'])
            y_bottom = min(box_i['y'] + box_i['h'], box_j['y'] + box_j['h'])
            intersection_w = x_right - x_left; intersection_h = y_bottom - y_top

            if intersection_w > 0 and intersection_h > 0:
                intersection_area = float(intersection_w * intersection_h) # Assurer float
                if box_i['area'] >= box_j['area']: larger_box_idx, smaller_box_idx, smaller_box_area = i, j, box_j['area']
                else: larger_box_idx, smaller_box_idx, smaller_box_area = j, i, box_i['area']

                if smaller_box_area > 0: # Vérification déjà présente mais on garde
                    ioa = intersection_area / smaller_box_area
                    if ioa > threshold_ioa:
                        discarded[smaller_box_idx] = True
                        # print(f"  Pruning: Box {smaller_box_idx} overlaps significantly (IoA={ioa:.2f}) with Box {larger_box_idx}. Discarding smaller.") # Moins verbeux
    pruned_boxes = [boxes[i] for i in range(num_boxes) if not discarded[i]]
    print(f"Pruning complete. Kept {len(pruned_boxes)} out of {num_boxes} boxes.")
    return pruned_boxes

# --- Code Principal ---

# 1. Charger image
original_image = cv2.imread(image_path)
if original_image is None: exit(f"Erreur: Impossible de charger {image_path}")
img_height, img_width = original_image.shape[:2]

# 2. Segmentation récursive
print("--- Starting Segmentation ---")
final_bounding_boxes = []
segment_image_recursive(original_image, offset_x=0, offset_y=0, current_depth=0)
print(f"--- Segmentation complete. Found {len(final_bounding_boxes)} raw boxes ---")

# 3. Élagage des superpositions
final_bounding_boxes.sort(key=lambda b: (b['x'], b['y']))
print(f"\n--- Pruning overlapping boxes (IoA Threshold: {overlap_threshold_ioa}) ---")
pruned_final_boxes = prune_overlapping_boxes(final_bounding_boxes, overlap_threshold_ioa)


# 4. Extraire, Isoler+Lisser Binaire, Padder, Filtrer et Sauvegarder
char_count = 0
skipped_count = 0
print(f"\nApplying final binary processing, padding, size filter, and saving {len(pruned_final_boxes)} boxes to '{output_base_dir}'...")

image_with_final_boxes = original_image.copy()

# Créer le noyau morphologique une seule fois (si utilisé)
if morph_kernel_size > 0:
    morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
else:
    morph_kernel = None

for i, box_data in enumerate(pruned_final_boxes):
    x, y, w, h = box_data['x'], box_data['y'], box_data['w'], box_data['h']
    binary_crop_input = box_data['binary_crop'] # Le crop binaire nettoyé (peut avoir plusieurs composants)

    # --- <<< NOUVELLE ÉTAPE : Isoler plus grand contour et appliquer Fermeture >>> ---
    final_binary_crop = None # Initialiser
    if binary_crop_input.size > 0 and cv2.countNonZero(binary_crop_input) > 0:
        # 1. Trouver les contours DANS le crop binaire
        contours_in_crop, _ = cv2.findContours(binary_crop_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_in_crop:
            # 2. Trouver le contour avec la plus grande aire
            max_area = 0
            max_contour = None
            for c in contours_in_crop:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_contour = c

            if max_contour is not None:
                # 3. Créer un masque juste pour le plus grand contour
                mask_largest = np.zeros_like(binary_crop_input)
                cv2.drawContours(mask_largest, [max_contour], -1, (255), thickness=cv2.FILLED)

                # 4. Appliquer la fermeture morphologique sur ce masque
                if morph_kernel is not None:
                    final_binary_crop = cv2.morphologyEx(mask_largest, cv2.MORPH_CLOSE, morph_kernel)
                    # print(f"  Applied Morphological Closing to largest contour of box {i}")
                else:
                    final_binary_crop = mask_largest # Pas de fermeture si kernel_size = 0
            else:
                 # print(f"Warning: No valid max contour found in crop {i}")
                 skipped_count += 1; continue # Passer si aucun contour max trouvé
        else:
            # print(f"Warning: No contours found in binary crop {i} despite non-zero pixels.")
            skipped_count += 1; continue # Passer si pas de contours trouvés
    else:
        # print(f"Skipping box {i}: Input binary crop is empty.")
        skipped_count += 1; continue # Passer si le crop initial est vide

    # Vérifier si final_binary_crop a été créé et n'est pas vide
    if final_binary_crop is None or final_binary_crop.size == 0 or cv2.countNonZero(final_binary_crop) == 0:
        # print(f"Skipping box {i}: Final binary crop is empty after processing.")
        skipped_count += 1; continue
    # --- <<< FIN NOUVELLE ÉTAPE >>> ---

    # Dessiner la boîte sur l'image de démo
    cv2.rectangle(image_with_final_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # Appliquer le padding (comme avant)
    x_pad = max(0, x - padding); y_pad = max(0, y - padding)
    x2_pad = min(img_width, x + w + padding); y2_pad = min(img_height, y + h + padding)

    # Extraire l'original paddé
    char_image_padded_original = original_image[y_pad:y2_pad, x_pad:x2_pad]

    # Créer l'image binaire finale paddée (en utilisant `final_binary_crop`)
    final_h = y2_pad - y_pad; final_w = x2_pad - x_pad
    if final_h > 0 and final_w > 0:
        crop_h, crop_w = final_binary_crop.shape[:2] # Utiliser les dimensions du crop final
        if crop_h > 0 and crop_w > 0:
            char_image_padded_binary = np.zeros((final_h, final_w), dtype=np.uint8)
            paste_x = padding if x > padding else x
            paste_y = padding if y > padding else y
            end_paste_y = paste_y + crop_h
            end_paste_x = paste_x + crop_w
            if end_paste_y <= final_h and end_paste_x <= final_w:
                 char_image_padded_binary[paste_y:end_paste_y, paste_x:end_paste_x] = final_binary_crop
            else:
                 max_h = min(crop_h, final_h - paste_y); max_w = min(crop_w, final_w - paste_x)
                 if max_h > 0 and max_w > 0:
                      char_image_padded_binary[paste_y:paste_y + max_h, paste_x:paste_x + max_w] = final_binary_crop[0:max_h, 0:max_w]
                 else: skipped_count += 1; continue
        else: skipped_count += 1; continue
    else: skipped_count += 1; continue

    # Filtrage par taille finale et sauvegarde
    if char_image_padded_original.size > 0:
        h_final_check, w_final_check = char_image_padded_original.shape[:2]
        if h_final_check >= min_output_height and w_final_check >= min_output_width:
            base_filename = f'char_{i:03d}.png'
            output_path_original = os.path.join(output_dir_original, base_filename)
            output_path_binary = os.path.join(output_dir_binary, base_filename)
            try:
                cv2.imwrite(output_path_original, char_image_padded_original)
                cv2.imwrite(output_path_binary, char_image_padded_binary) # Sauvegarde du binaire final
                char_count += 1
            except Exception as e: print(f"Erreur sauvegarde {base_filename}: {e}"); skipped_count += 1
        else: skipped_count += 1
    else: skipped_count += 1

print(f"\nTerminé.")
print(f"{char_count} paires d'images (original + binaire finalisé) sauvegardées dans '{output_base_dir}'.")
print(f"{skipped_count} images potentielles ignorées.")

# Optionnel: Affichage final
cv2.imshow('Final Processed Boxes', image_with_final_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()