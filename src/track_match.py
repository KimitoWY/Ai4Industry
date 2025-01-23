import numpy as np
import cv2
from sklearn.metrics import pairwise_distances

class TrackMatcher:
    def __init__(self, satellite_image):
        self.satellite_image = satellite_image
        self.track_outline = None
        self.start_line = None
        self.initialize_reference_track()

    def initialize_reference_track(self):
        """Extrait le tracé du circuit depuis l'image satellite"""
        # Convertir en niveaux de gris si nécessaire
        if len(self.satellite_image.shape) == 3:
            gray = cv2.cvtColor(self.satellite_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.satellite_image

        # Appliquer un seuillage adaptatif pour isoler la piste
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 115, 2
        )

        # Détecter les contours de la piste
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Prendre le plus grand contour (supposé être la piste)
        track_contour = max(contours, key=cv2.contourArea)

        # Créer une image du tracé
        self.track_outline = np.zeros_like(gray)
        cv2.drawContours(self.track_outline, [track_contour], -1, 255, 2)

        # Détecter la ligne de départ (supposée être une ligne droite)
        self.detect_start_line()

    def detect_start_line(self):
        """Détecte la ligne de départ dans l'image satellite"""
        # Appliquer la détection de lignes de Hough
        lines = cv2.HoughLinesP(
            self.track_outline, 1, np.pi/180, 50,
            minLineLength=100, maxLineGap=10
        )

        if lines is not None:
            # Prendre la ligne la plus horizontale comme ligne de départ
            start_line = min(
                lines,
                key=lambda l: abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))
            )
            self.start_line = start_line[0]

    def match_frame_position(self, canny_frame):
        """
        Compare la frame actuelle avec le tracé de référence
        pour estimer la position sur le circuit
        """
        # Extraire les points caractéristiques de la frame
        points = np.column_stack(np.where(canny_frame > 0))

        if len(points) == 0:
            return None

        # Si une ligne de départ est détectée dans la frame
        if self.detect_start_line_in_frame(canny_frame):
            # Réinitialiser le matching à la position de départ
            return self.start_line[:2]  # Retourne le point de départ

        # Sinon, chercher la meilleure correspondance avec le tracé
        track_points = np.column_stack(np.where(self.track_outline > 0))

        # Calculer les distances entre les points de la frame et du tracé
        distances = pairwise_distances(points, track_points)

        # Trouver le point du tracé le plus proche
        min_dist_idx = np.argmin(distances.min(axis=0))
        matched_position = track_points[min_dist_idx]

        return matched_position

    def detect_start_line_in_frame(self, canny_frame):
        """Détecte si la ligne de départ est visible dans la frame"""
        lines = cv2.HoughLinesP(
            canny_frame, 1, np.pi/180, 50,
            minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return False

        # Chercher une ligne horizontale similaire à la ligne de départ
        for line in lines:
            angle = np.abs(np.arctan2(
                line[0][3]-line[0][1],
                line[0][2]-line[0][0]
            ))
            if angle < 0.1:  # Seuil pour la horizontalité
                return True

        return False

    def visualize_matching(self, canny_frame, matched_position):
        """Visualise la position matchée sur le tracé de référence"""
        if matched_position is None:
            return self.track_outline.copy()

        visualization = cv2.cvtColor(self.track_outline, cv2.COLOR_GRAY2BGR)
        # Marquer la position matchée
        cv2.circle(
            visualization,
            (int(matched_position[1]), int(matched_position[0])),
            5, (0, 0, 255), -1
        )
        # Dessiner la ligne de départ
        if self.start_line is not None:
            cv2.line(
                visualization,
                (self.start_line[0], self.start_line[1]),
                (self.start_line[2], self.start_line[3]),
                (0, 255, 0), 2
            )

        return visualization

def process_video_with_reference(video_path, satellite_image):
    """Traite la vidéo en utilisant l'image satellite comme référence"""
    matcher = TrackMatcher(satellite_image)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Appliquer Canny sur la frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 150)

        # Trouver la position correspondante sur le tracé
        matched_position = matcher.match_frame_position(canny)

        # Visualiser le résultat
        visualization = matcher.visualize_matching(canny, matched_position)


        cv2.imshow('Track Matching', visualization)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
