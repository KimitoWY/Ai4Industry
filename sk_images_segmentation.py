#Import sk-image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io, segmentation, feature, future
from skimage.segmentation import slic
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from functools import partial   
    
    
#============================================================================================================
#                                              SK-image
#============================================================================================================

# # Chargez votre image
# full_img = io.imread("./output/frame_1516.jpg")

# # Optionnel : convertissez en niveaux de gris
# #full_img = rgb2gray(full_img)

# # Si nécessaire, découpez l'image
# img = full_img  # Utilisez toute l'image ou découpez si besoin

# # Définissez les étiquettes d'entraînement
# training_labels = np.zeros(img.shape[:2], dtype=np.uint8)
# training_labels[:130] = 1
# training_labels[:170, :400] = 1
# training_labels[600:900, 200:650] = 2
# training_labels[330:430, 210:320] = 3
# training_labels[260:340, 60:170] = 4
# training_labels[150:200, 720:860] = 4
# training_labels[100:200, 100:200] = 5
# training_labels[300:400, 500:600] = 6

# # Paramètres pour les fonctionnalités
# sigma_min = 1
# sigma_max = 16
# features_func = partial(
#     feature.multiscale_basic_features,
#     intensity=True,
#     edges=False,
#     texture=True,
#     sigma_min=sigma_min,
#     sigma_max=sigma_max,
#     channel_axis=-1 if len(img.shape) == 3 else None,
# )

# # Extraction des fonctionnalités
# features = features_func(img)

# # Entraînement du classificateur
# clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
# clf = future.fit_segmenter(training_labels, features, clf)

# # Prédiction
# result = future.predict_segmenter(features, clf)

# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 4))

# # Appliquer les frontières sur l'image segmentée
# segmented_with_boundaries = segmentation.mark_boundaries(result, result, mode='thick')

# # Afficher les frontières sur la segmentation
# ax[0].imshow(segmented_with_boundaries, cmap='viridis')  # Utilisez une colormap pour mieux visualiser
# ax[0].set_title('Segmentation avec frontières')

# # Afficher uniquement la segmentation
# ax[1].imshow(result, cmap='viridis')
# ax[1].set_title('Segmentation seule')

# ax[2].imshow(full_img)

# fig.tight_layout()
# plt.show()

#============================================================================================================
#                                     Segmentation des images
#============================================================================================================

# Dossier contenant les images
input_folder = "./output"  # Remplacez par le chemin de votre dossier
output_folder = "./output_segmented_images"  # Dossier pour les images segmentées
os.makedirs(output_folder, exist_ok=True)  # Créer le dossier si nécessaire

# Paramètres de segmentation
sigma_min = 2
sigma_max = 16
features_func = partial(
    feature.multiscale_basic_features,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    channel_axis=-1,  # Prise en charge des images couleur
)

# Étape 1 : Charger les images et définir les étiquettes sur une image de référence
# Charger une image de référence
reference_image_path = os.path.join(input_folder, os.listdir(input_folder)[0])
reference_image = io.imread(reference_image_path)

# Labels d'entraînement (ajustez selon votre cas)
training_labels = np.zeros(reference_image.shape[:2], dtype=np.uint8)
training_labels[:100, :200] = 1
training_labels[300:400, 500:600] = 2

# Extraire les fonctionnalités de l'image de référence
features = features_func(reference_image)

# Entraîner le classificateur
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
clf = future.fit_segmenter(training_labels, features, clf)

# Étape 2 : Appliquer la segmentation à toutes les images
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        img = io.imread(image_path)

        # Extraire les fonctionnalités
        features = features_func(img)

        # Prédire la segmentation
        result = future.predict_segmenter(features, clf)

        # Sauvegarder l'image segmentée
        output_path = os.path.join(output_folder, filename)
        result_colored = segmentation.mark_boundaries(result, result, mode='thick')
        io.imsave(output_path, (result_colored * 255).astype(np.uint8))

        print(f"Segmented and saved: {output_path}")

#============================================================================================================
# Trouver les classes sur l'images segmenter
#============================================================================================================

# # Charger deux images
# image1 = "./output_segmented_images/frame_0044.jpg"
# image2 = "./output_segmented_images/frame_0045.jpg"

# # Segmentation avec SLIC
# labels1 = slic(image1, n_segments=100, compactness=10)
# labels2 = slic(image2, n_segments=100, compactness=10)

# # Calcul de la similarité des segmentations
# score, precision, recall = adapted_rand_error(labels1, labels2)
# print(f"Score de similarité des segmentations : {score}")

# # Exemple de classification par intensité moyenne
# def classify_region(image, labels):
#     unique_labels = np.unique(labels)
#     classes = {}
#     for label in unique_labels:
#         mask = labels == label
#         mean_intensity = image[mask].mean()
#         classes[label] = 'classe_A' if mean_intensity > 128 else 'classe_B'
#     return classes

# # Classification des régions pour les deux images
# classes1 = classify_region(rgb2gray(image1), labels1)
# classes2 = classify_region(rgb2gray(image2), labels2)

# # Comparaison des classes
# for label in classes1:
#     if label in classes2:
#         print(f"Région {label} : classe1 = {classes1[label]}, classe2 = {classes2[label]}")

#============================================================================================================
# Map
#============================================================================================================

# # Dossier contenant les images segmentées
# segmented_folder = "./output_segmented_images"
# output_image_path = "./kart_path.jpg"

# # Classe correspondant au kart dans la segmentation (par exemple, 3)
# kart_class = 3

# # Liste pour stocker les positions du kart
# kart_positions = []

# # Parcourir les images segmentées
# for filename in sorted(os.listdir(segmented_folder)):
#     if filename.endswith((".png", ".jpg", ".jpeg")):
#         image_path = os.path.join(segmented_folder, filename)
#         segmented_image = io.imread(image_path)

#         # Identifier les pixels correspondant au kart
#         kart_mask = (segmented_image == kart_class)

#         # Calculer le centre de masse de la région du kart
#         if np.any(kart_mask):
#             y, x = center_of_mass(kart_mask)
#             kart_positions.append((int(x), int(y)))

# # Créer une image de fond (par exemple, la première image segmentée)
# background_image = io.imread(os.path.join(segmented_folder, os.listdir(segmented_folder)[0]))
# path_image = np.copy(background_image)

# # Dessiner le chemin parcouru par le kart
# for i in range(1, len(kart_positions)):
#     cv2.line(path_image, kart_positions[i - 1], kart_positions[i], (255, 0, 0), 2)  # Ligne bleue

# # Sauvegarder l'image avec le chemin tracé
# io.imsave(output_image_path, path_image)
# print(f"Kart path saved at: {output_image_path}")