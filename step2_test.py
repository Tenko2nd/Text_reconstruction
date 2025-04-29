import cv2
import numpy as np
import os

# Chargement et redimensionnement
img = cv2.imread("cropped_images/coca.jpg")
original_height, original_width = img.shape[:2]
new_width = 500
new_height = int((new_width / original_width) * original_height)
img = cv2.resize(img, (new_width, new_height))

# Grayscale et binarisation
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Fermeture morphologique
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed", closed)
closed_copy = closed.copy()

# Détection des contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrage et tri des contours (par coordonnée X pour lire de gauche à droite)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # Tri gauche -> droite

# Création du dossier de sortie
os.makedirs("lettres_sep", exist_ok=True)

margin = 3
for idx, (x, y, w, h) in enumerate(bounding_boxes):
    if w > 5 and h > 10:
        # Coordonnées avec marge, sans dépasser l'image
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, closed_copy.shape[1])
        y2 = min(y + h + margin, closed_copy.shape[0])

        img_crop = closed_copy[y1:y2, x1:x2].copy()

        # Sauvegarder l'image contenant uniquement le plus grand contour
        cv2.imwrite(f"lettres_sep/lettre_{idx}.png", img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
