import cv2
import numpy as np

# Dimensions de la carte globale
MAP_WIDTH = 2000
MAP_HEIGHT = 2000

# Création d'une carte globale vierge
global_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)

# Facteur d'échelle pour ajuster les contours au sein de la carte globale
scale_factor = 5  # Ajustez selon les dimensions réelles du circuit

# Coordonnées initiales de l'origine sur la carte (centre de la carte)
map_origin = (MAP_WIDTH // 2, MAP_HEIGHT // 2)

# Fonction pour ajouter les contours d'une frame à la carte globale
def add_frame_to_map(frame, global_map, map_origin, scale_factor):
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Appliquer l'algorithme Canny pour extraire les contours
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Redimensionner les contours selon l'échelle
    edges_resized = cv2.resize(edges, (edges.shape[1] * scale_factor, edges.shape[0] * scale_factor))

    # Obtenir les coordonnées où des contours sont détectés
    y_indices, x_indices = np.where(edges_resized > 0)

    # Mettre à jour la carte globale (en déplaçant selon map_origin)
    for x, y in zip(x_indices, y_indices):
        map_x = map_origin[0] + x
        map_y = map_origin[1] - y  # Inverser l'axe Y si nécessaire
        if 0 <= map_x < global_map.shape[1] and 0 <= map_y < global_map.shape[0]:
            global_map[map_y, map_x] = 255  # Marquer les contours

# Charger une vidéo ou des images successives
video_path = './curves_output.mp4'  # Remplacez par le chemin de votre vidéo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo.")
    exit()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:

        break  # Fin de la vidéo

    # Ajouter la frame au schéma global
    add_frame_to_map(frame, global_map, map_origin, scale_factor)

    # Afficher la progression
    cv2.imshow("Carte globale en construction", global_map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"Frame {frame_count} traitée.")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Sauvegarder la carte finale
cv2.imwrite("carte_globale.png", global_map)

print(f"Carte globale générée avec {frame_count} frames.")
