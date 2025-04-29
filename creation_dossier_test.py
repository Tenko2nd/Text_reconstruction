import os
import shutil
import random

def create_test_dataset(images_dir, test_dir, test_ratio=0.1):
    """
    Crée un dataset de test en sélectionnant un pourcentage des images de chaque sous-dossier dans 'images_dir' et les déplaçant dans 'test_dir'.
    
    :param images_dir: Répertoire source contenant les sous-dossiers d'images.
    :param test_dir: Répertoire de destination où les images de test seront stockées.
    :param test_ratio: Le pourcentage des images à sélectionner pour le test (par défaut 10%).
    """
    # Créer le répertoire de test s'il n'existe pas
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Parcours des sous-dossiers dans le dossier des images
    for subfolder in os.listdir(images_dir):
        subfolder_path = os.path.join(images_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Créer un sous-dossier pour cette classe dans le répertoire de test
            test_subfolder_path = os.path.join(test_dir, subfolder)
            if not os.path.exists(test_subfolder_path):
                os.makedirs(test_subfolder_path)
            
            # Récupérer toutes les images dans ce sous-dossier
            images = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
            
            # Calculer le nombre d'images à prendre pour le test (10% des images)
            num_test_images = int(len(images) * test_ratio)
            
            # Sélectionner aléatoirement les images pour le test
            test_images = random.sample(images, num_test_images)
            
            # Copier les images sélectionnées dans le sous-dossier de test
            for image in test_images:
                src_image_path = os.path.join(subfolder_path, image)
                dst_image_path = os.path.join(test_subfolder_path, image)
                shutil.copy(src_image_path, dst_image_path)
                os.remove(src_image_path)

if __name__ == "__main__":
    # Chemins des dossiers (remplacer par vos chemins réels)
    images_dir = "images"  # Le dossier contenant les sous-dossiers de caractères
    test_dir = "test"      # Le dossier où les images de test seront stockées
    
    # Créer le dossier de test en sélectionnant 10% des images
    create_test_dataset(images_dir, test_dir, test_ratio=0.1)
