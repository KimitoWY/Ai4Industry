import cv2
import numpy as np
from matplotlib import pyplot as plt

# Charger l'image
image_path = 'test.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Détecter les contours
edges = cv2.Canny(image, 100, 200)

# Afficher l'image d'entrée et de sortie
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Image d\'entrée')
plt.axis('off')

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Contours détectés')
plt.axis('off')

plt.show()


