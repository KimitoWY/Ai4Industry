import numpy as np
import cv2
from scipy.spatial import cKDTree

class TrackMatcher:
    def __init__(self, satellite_image):
        self.satellite_image = satellite_image
        self.track_outline = None
        self.start_line = None
        self.kdtree = None
        self.initialize_reference_track()    

    def initialize_reference_track(self):
        """Extrait le tracé du circuit depuis l'image satellite"""
        # Convertir en niveaux de gris si l'image est en couleur
        if len(self.satellite_image.shape) == 3:
            gray = cv2.cvtColor(self.satellite_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.satellite_image

        # Réduction de taille pour accélérer le traitement
        scale = 0.5
        gray = cv2.resize(gray, None, fx=scale, fy=scale)

        # Appliquer Canny pour détecter les contours
        edges = self.circuit_cannying(gray)
        cv2.imwrite('satellite_canny.png', edges)

        # Appliquer un seuillage simple pour binariser l'image
        _, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Détecter les contours dans l'image binarisée
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Aucun contour trouvé dans l'image du circuit.")

        # Sélectionner le contour ayant la plus grande surface (le tracé du circuit)
        track_contour = max(contours, key=cv2.contourArea)

        # Simplifier le contour pour réduire le nombre de points
        epsilon = 0.01 * cv2.arcLength(track_contour, True)
        track_contour = cv2.approxPolyDP(track_contour, epsilon, True)

        # Créer une image du tracé avec les contours
        self.track_outline = np.zeros_like(gray)
        cv2.drawContours(self.track_outline, [track_contour], -1, 255, 2)

        # Extraire les points du tracé
        track_points = np.column_stack(np.where(self.track_outline > 0))

        # Sous-échantillonner les points pour alléger le KD-Tree
        track_points = track_points[::10]
        self.kdtree = cKDTree(track_points)

        # Détecter la ligne de départ
        self.detect_start_line()

    def apply_anisotropic_filter(image, iterations=1, kappa=50, gamma=0.1):
        """Applique un filtre anisotrope sur l'image pour lisser les contours"""
        # Convertir l'image en float64 pour les calculs
        image = image.astype(np.float64)

        for _ in range(iterations):
            # Calculer les gradients
            gradient_north = np.roll(image, -1, axis=0) - image
            gradient_south = np.roll(image, 1, axis=0) - image
            gradient_east = np.roll(image, -1, axis=1) - image
            gradient_west = np.roll(image, 1, axis=1) - image

            # Calculer les coefficients de conduction
            c_north = np.exp(-(gradient_north / kappa) ** 2)
            c_south = np.exp(-(gradient_south / kappa) ** 2)
            c_east = np.exp(-(gradient_east / kappa) ** 2)
            c_west = np.exp(-(gradient_west / kappa) ** 2)

            # Mettre à jour l'image
            image += gamma * (
                c_north * gradient_north +
                c_south * gradient_south +
                c_east * gradient_east +
                c_west * gradient_west
            )

        # Reconvertir en uint8 avant de retourner l'image
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def circuit_cannying(image, scale=2, interpolation=cv2.INTER_LINEAR):
        """Applique Canny sur l'image du circuit après réduction"""
        # Appliquer un filtre noir ou blanc
        _, binary = cv2.threshold(image, 105, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(binary, (5, 5), 0)

        anisotropic = TrackMatcher.apply_anisotropic_filter(binary, iterations=2, kappa=50, gamma=0.1)
        cv2.imwrite('satellite_anisotropic.png', anisotropic)        

        # Appliquer un filtre noir ou blanc
        # _, binary = cv2.threshold(image_resized, 105, 255, cv2.THRESH_BINARY)
        # blur = cv2.GaussianBlur(binary, (7, 7), 0)

        # Appliquer Canny pour détecter les contours
        edges = cv2.Canny(blur, 150, 230)
        edges2 = cv2.Canny(anisotropic, 150, 230)

        # Dilater les contours pour combler les discontinuités
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        kernel2 = np.ones((5, 5), np.uint8)
        edges_dilated2 = cv2.dilate(edges2, kernel2, iterations=1)

        # Réduire la taille de l'image pour accélérer le traitement
        width = edges_dilated.shape[1] // scale
        height = edges_dilated.shape[0] // scale
        image_resized = cv2.resize(edges_dilated, (width, height), interpolation=interpolation)

        width2 = edges_dilated2.shape[1] // scale
        height2 = edges_dilated2.shape[0] // scale
        image_resized2 = cv2.resize(edges_dilated2, (width2, height2), interpolation=interpolation)

        cv2.imshow('edges', image_resized)
        cv2.imshow('edges2', image_resized2)
        
        return image_resized , image_resized2

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
        """Version optimisée du matching de position"""
        # Redimensionner la frame pour correspondre à l'échelle du tracé
        canny_frame = cv2.resize(canny_frame, 
                               (self.track_outline.shape[1], 
                                self.track_outline.shape[0]))
        
        # Sous-échantillonner les points de la frame
        points = np.column_stack(np.where(canny_frame > 0))[::20]  # Prendre 1 point sur 20
        
        if len(points) == 0:
            return None
            
        # Si une ligne de départ est détectée
        if self.detect_start_line_in_frame(canny_frame):
            return self.start_line[:2]
            
        # Utiliser KD-Tree pour trouver le point le plus proche
        distance, index = self.kdtree.query(points, k=1)
        
        # Prendre le point avec la distance minimale
        best_match_idx = np.argmin(distance)
        matched_position = points[best_match_idx]
        
        return matched_position

    def detect_start_line_in_frame(self, canny_frame):
        """Version optimisée de la détection de ligne de départ"""
        # Réduire la zone de recherche à la partie supérieure de l'image
        search_region = canny_frame[:canny_frame.shape[0]//3, :]
        
        lines = cv2.HoughLinesP(
            search_region, 1, np.pi/180, 
            threshold=50,
            minLineLength=50,  # Réduit pour plus de robustesse
            maxLineGap=20
        )
        
        if lines is None:
            return False
            
        # Vectoriser le calcul des angles
        dx = lines[:, 0, 2] - lines[:, 0, 0]
        dy = lines[:, 0, 3] - lines[:, 0, 1]
        angles = np.abs(np.arctan2(dy, dx))
        
        # Vérifier si au moins une ligne est proche de l'horizontale
        return np.any(angles < 0.1)
    
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
        
        print("Traitement de la frame...")

        # Appliquer Canny sur la frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 150)
        
        print("Recherche de la position...")
        # Trouver la position correspondante sur le tracé
        matched_position = matcher.match_frame_position(canny)
        
        print("Position trouvée:", matched_position)
        # Visualiser le résultat
        visualization = matcher.visualize_matching(canny, matched_position)

        print("Affichage de la frame...")
        
        cv2.imshow('Track Matching', visualization)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sauvegarder la visualisation
        output_frame = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'test/output_frame_{cap.get(cv2.CAP_PROP_POS_FRAMES):04.0f}.png', output_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    

# Charger l'image satellite
satellite_image = cv2.imread('satellite_crop.png')

# Traiter la vidéo
# process_video_with_reference('curves_output.avi', satellite_image)

matcher = TrackMatcher(satellite_image)
print("its good")