import numpy as np
import cv2
from collections import deque
from scipy.spatial.transform import Rotation

class TrackReconstructor:
    def __init__(self, max_history=1000):
        # Stockage de l'historique des positions et orientations
        self.positions = deque(maxlen=max_history)
        self.orientations = deque(maxlen=max_history)
        # Paramètres de la caméra (à ajuster selon votre setup)
        self.camera_height = 1.0  # hauteur estimée de la caméra
        self.fov = 90  # champ de vision en degrés
        # Carte vue du dessus
        self.top_view_map = None
        self.map_resolution = 0.1  # mètres par pixel
        self.map_size = (1000, 1000)  # taille en pixels
        
    def process_frame(self, canny_frame):
        """Traite une frame avec les contours Canny pour extraire la position relative"""
        height, width = canny_frame.shape
        
        # 1. Séparer les lignes gauche et droite de la piste
        left_region = canny_frame[:, :width//2]
        right_region = canny_frame[:, width//2:]
        
        # 2. Détecter les lignes principales avec Hough
        left_lines = cv2.HoughLinesP(left_region, 1, np.pi/180, 50, 
                                   minLineLength=100, maxLineGap=50)
        right_lines = cv2.HoughLinesP(right_region, 1, np.pi/180, 50, 
                                    minLineLength=100, maxLineGap=50)
        
        if left_lines is None or right_lines is None:
            return None
        
        # 3. Estimer la direction moyenne de la piste
        track_direction = self._estimate_track_direction(left_lines, right_lines)
        
        # 4. Estimer la largeur de la piste
        track_width = self._estimate_track_width(left_lines, right_lines)
        
        # 5. Estimer le déplacement relatif
        relative_position = self._estimate_relative_position(track_direction, track_width)
        
        return relative_position
    
    def _estimate_track_direction(self, left_lines, right_lines):
        """Estime la direction de la piste à partir des lignes détectées"""
        all_angles = []
        
        for lines in [left_lines, right_lines]:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                all_angles.append(angle)
        
        # Utiliser la moyenne des angles comme direction
        mean_angle = np.mean(all_angles)
        return mean_angle
    
    def _estimate_track_width(self, left_lines, right_lines):
        """Estime la largeur de la piste en pixels"""
        # Prendre les points moyens des lignes gauche et droite
        left_points = np.mean([line[0] for line in left_lines], axis=0)
        right_points = np.mean([line[0] for line in right_lines], axis=0)
        
        # Calculer la distance moyenne
        width = np.linalg.norm(right_points - left_points)
        return width
    
    def _estimate_relative_position(self, track_direction, track_width):
        """Estime la position relative basée sur la direction et la largeur"""
        # Convertir la largeur en pixels en distance métrique approximative
        estimated_distance = (self.camera_height * track_width) / (2 * np.tan(np.radians(self.fov/2)))
        
        # Créer le vecteur de déplacement
        displacement = np.array([
            estimated_distance * np.cos(track_direction),
            estimated_distance * np.sin(track_direction)
        ])
        
        return displacement
    
    def update_map(self, relative_position):
        """Met à jour la carte vue du dessus avec la nouvelle position"""
        if self.top_view_map is None:
            self.top_view_map = np.zeros(self.map_size, dtype=np.uint8)
        
        # Convertir la position relative en coordonnées de la carte
        map_position = (
            int(relative_position[0] / self.map_resolution + self.map_size[0]//2),
            int(relative_position[1] / self.map_resolution + self.map_size[1]//2)
        )
        
        # Dessiner la nouvelle position sur la carte
        cv2.circle(self.top_view_map, map_position, 2, 255, -1)
        
        # Si on a suffisamment de points, dessiner les connexions
        if len(self.positions) > 1:
            prev_position = self.positions[-1]
            prev_map_position = (
                int(prev_position[0] / self.map_resolution + self.map_size[0]//2),
                int(prev_position[1] / self.map_resolution + self.map_size[1]//2)
            )
            cv2.line(self.top_view_map, prev_map_position, map_position, 255, 1)
        
        # Sauvegarder la position
        self.positions.append(relative_position)
    
    def process_video_frame(self, frame):
        """Traite une frame de la vidéo"""
        # Appliquer Canny si ce n'est pas déjà fait
        if len(frame.shape) > 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray, 50, 150)
        else:
            canny = frame
        
        # Obtenir la position relative
        relative_position = self.process_frame(canny)
        
        if relative_position is not None:
            # Mettre à jour la carte
            self.update_map(relative_position)
            
        return self.top_view_map
    
# Création du reconstructeur
reconstructor = TrackReconstructor()

# Pour une vidéo
cap = cv2.VideoCapture('curves_output.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Obtenir la carte mise à jour
    top_view = reconstructor.process_video_frame(frame)
    
    # Afficher la carte
    if top_view is not None:
        cv2.imshow('Track Map', top_view)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()