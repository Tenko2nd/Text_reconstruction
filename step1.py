import json
import os
import cv2 as cv
import glob


def extraction_texte():
    # ---- Chargement JSON ------
    with open('TextOCR_0.1_train.json', 'r') as file:
        data = json.load(file)

    # ------- Dictionnaire pour stocker bbox par image_id ----------
    bbox_by_image = {}

    for key, value in data['anns'].items():
        image_id = value['image_id']
        bbox = value['bbox']

        if image_id in bbox_by_image:
            bbox_by_image[image_id].append(bbox)
        else:
            bbox_by_image[image_id] = [bbox]

    # ------- Création des dossiers de sortie ---------
    output_dir = "output_bbox"
    crops_dir = "output_crops"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    for image_id, bboxes in bbox_by_image.items():

        image_path = os.path.join("train_images", image_id + ".jpg")

        if os.path.isfile(image_path):
            print(f"Le fichier {image_path} existe !!")

            image = cv.imread(image_path)

            if image_id in bbox_by_image and bbox_by_image[image_id]:
                for idx, bbox in enumerate(bbox_by_image[image_id]):
                    xmin, ymin, width, height = bbox
                    xmax = xmin + width
                    ymax = ymin + height

                    # --- Dessin du rectangle ---
                    cv.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

                    # --- Crop et sauvegarde ---
                    cropped = image[int(ymin):int(ymax), int(xmin):int(xmax)]
                    crop_filename = os.path.join(crops_dir, f"{image_id}_crop_{idx}.jpg")
                    cv.imwrite(crop_filename, cropped)

                # Sauvegarde de l'image avec bboxes
                output_path = os.path.join(output_dir, f"{image_id}_bbox.jpg")
                cv.imwrite(output_path, image)

                cv.waitKey(0)
                cv.destroyAllWindows()
            else:
                print(f"Aucune bounding box trouvée pour {image_id}")
        else:
            print(f"Le fichier {image_path} n'existe pas.")


    # image_id = "0a4c96f56882bffd"
    #
    # image_path = os.path.join("train_images", image_id + ".jpg")
    #
    #
    # if os.path.isfile(image_path):
    #     print(f"Le fichier {image_path} existe !!")
    #
    #     image = cv.imread(image_path)
    #
    #     if image_id in bbox_by_image and bbox_by_image[image_id]:
    #         for bbox in bbox_by_image[image_id]:  # Boucle sur toutes les bboxes
    #
    #             xmin, ymin, width, height = bbox
    #             xmax = xmin + width
    #             ymax = ymin + height
    #
    #             cv.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
    #
    #         cv.imwrite(image_id + "_bbox.jpg", image)
    #         cv.waitKey(0)
    #         cv.destroyAllWindows()
    #     else:
    #         print(f"Aucune bounding box trouvée pour {image_id}")
    #
    # else:
    #     print(f"Le fichier {image_path} n'existe pas.")

if __name__ == "__main__":
    extraction_texte()