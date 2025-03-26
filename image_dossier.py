import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def csv_to_image(row, image_size=28):
    """
    Transforme une ligne du CSV en une image NumPy.
    
    Args:
        row: Une ligne du DataFrame Pandas contenant les données du pixel.
        image_size: La taille de l'image (par défaut : 28x28).
    
    Returns:
        Une image NumPy de taille (image_size, image_size).
    """
    image = np.array(row[2:], dtype=np.float32).reshape(image_size, image_size)  # Spécifie le type de données
    return image

def save_image(image, label, image_size=28):
    """
    Sauvegarde l'image dans un dossier spécifique au code ASCII du label avec un nom de fichier unique.
    
    Args:
        image: L'image NumPy à sauvegarder.
        label: L'étiquette de l'image, utilisée pour créer un dossier avec son code ASCII.
        image_size: La taille de l'image (par défaut : 28x28).
    """
    # Convertir le label en son code ASCII
    label_ascii = ord(str(label))  # Utiliser ord() pour obtenir le code ASCII du label
    
    # Créer un dossier pour le code ASCII du label si ce n'est pas déjà fait
    label_dir = f'images/{label_ascii}'
    os.makedirs(label_dir, exist_ok=True)
    
    # Trouver le prochain index pour l'image (le nombre d'images déjà présentes dans le dossier)
    existing_images = os.listdir(label_dir)
    next_index = len(existing_images)  # Utilise le nombre de fichiers existants comme prochain index
    
    # Sauvegarder l'image
    filename = f'{label_dir}/{label_ascii}_{next_index}.png'
    plt.imsave(filename, image, cmap='gray')
    print(f"Image sauvegardée sous : {filename}")

# Chargement des données à partir du CSV
try:
    df = pd.read_csv('94_character_TMNIST.csv')  
except FileNotFoundError:
    print("Fichier '94_character_TMNIST.csv' introuvable. Assurez-vous que le fichier se trouve dans le répertoire courant.")
    exit()

# Remplacer les valeurs manquantes par 0
df.fillna(0, inplace=True)

# Parcourir les lignes du DataFrame et sauvegarder les images
for i in range(len(df)):
    row = df.iloc[i]  # Accède à la i-ème ligne du DataFrame
    image = csv_to_image(row)
    label = row['labels']  # Récupère l'étiquette de la colonne 'labels'
    save_image(image, label)  # Sauvegarde l'image avec le code ASCII du label
