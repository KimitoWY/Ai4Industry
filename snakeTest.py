import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt

# Charger la vidéo ou activer la webcam (0 pour webcam)
video = cv2.VideoCapture("edges_output.avi")  # Remplacez "0" par "video.mp4" pour une vidéo

# Paramètres de l'algorithme Snake
snake_alpha = 0.01  # Poids de la tension
snake_beta = 10  # Poids de la rigidité
snake_gamma = 0.01  # Poids pour l'énergie externe
snake_iterations = 500  # Nombre d'itérations pour Snake

# Définir une forme initiale de contour (par exemple, un cercle au centre de la frame)
ret, first_frame = video.read()
height, width, _ = first_frame.shape

s = np.linspace(0, 2 * np.pi, 400)
center_x, center_y = width // 2, height // 2  # Centre du cercle
radius = 100  # Rayon initial
x = center_x + radius * np.cos(s)
y = center_y + radius * np.sin(s)
init_snake = np.array([x, y]).T

while True:
    # Lire une frame vidéo
    ret, frame = video.read()
    if not ret:
        break  # Fin de la vidéo ou erreur

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lissage de l'image pour réduire le bruit
    smoothed = gaussian(gray, sigma=2, preserve_range=True)

    # Appliquer l'algorithme Snake
    snake = active_contour(smoothed, init_snake, alpha=snake_alpha, beta=snake_beta, gamma=snake_gamma)

    # Dessiner le contour sur l'image originale
    for point in snake.astype(np.int32):
        cv2.circle(frame, (point[0], point[1]), 1, (0, 255, 0), -1)

    # Afficher la vidéo avec le contour
    cv2.imshow("Contours Actifs (Snakes)", frame)

    # Mettre à jour la position initiale du contour pour la prochaine frame
    init_snake = snake

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video.release()
cv2.destroyAllWindows()
