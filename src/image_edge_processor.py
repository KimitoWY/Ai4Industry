import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class ImageEdgeProcessor:

    @staticmethod
    def display_images(original, edges):
        """
        Display the original and edge-detected images side by side.

        Args:
            original (numpy.ndarray): The original grayscale image.
            edges (numpy.ndarray): The edge-detected image.
        """
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.imshow(original, cmap='gray')
        plt.title("Image d'entrée")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(edges, cmap='gray')
        plt.title("Contours détectés")
        plt.axis('off')

        plt.show()

    @staticmethod
    def detect_edges(image_path, scale=2, interpolation=cv2.INTER_AREA):
        """
        Detect edges in an image.

        Args:
            image_path (str): Path to the input image.
            scale (int): Scaling factor to resize the image.
            interpolation: Interpolation method for resizing.

        Returns:
            tuple: The resized image and the edge-detected image.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        width = image.shape[1] // scale
        height = image.shape[0] // scale
        image = cv2.resize(image, (width, height), interpolation=interpolation)

        # print("taille de l'image: ", image.shape, "\n")

        edges = cv2.Canny(image, 150, 250)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return image, edges

    @staticmethod
    def process_images_from_folder(folder_path, output_folder,start_index, end_index, scale=5, interpolation=cv2.INTER_AREA):
        """
        Process and detect edges in a sequence of images from a folder.

        Args:
            folder_path (str): Path to the folder containing images.
            start_index (int): Starting frame index.
            end_index (int): Ending frame index.
            scale (int): Scaling factor to resize images.
            interpolation: Interpolation method for resizing.
        """
        os.makedirs(output_folder, exist_ok=True)
        for i in range(start_index, end_index + 1):
            image_path = os.path.join(folder_path, f'frame_{i:04d}.jpg')
            if os.path.exists(image_path):
                image, edges = ImageEdgeProcessor.detect_edges(image_path, scale, interpolation)
                cv2.imwrite(output_folder + f'frame_{i:04d}.jpg' , edges)
            else:
                # print(f"Image {image_path} not found.")
                pass

    @staticmethod
    def process_video(video_path, scale=2, interpolation=cv2.INTER_AREA):
        """
        Process and detect edges in a video, and save the output as video files.

        Args:
            video_path (str): Path to the input video.
            scale (int): Scaling factor to resize video frames.
            interpolation: Interpolation method for resizing.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = min(25, input_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // scale
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // scale

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_original = cv2.VideoWriter('original_output.avi', fourcc, fps, (width, height), False)
        out_edges = cv2.VideoWriter('edges_output.avi', fourcc, fps, (width, height), False)

        delay = int(1000 / fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (width, height), interpolation=interpolation)

            edges = cv2.Canny(gray_resized, 150, 380)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            out_original.write(gray_resized)
            out_edges.write(edges)

            combined = np.hstack((gray_resized, edges))
            cv2.imshow('Original and Edges', combined)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        out_original.release()
        out_edges.release()
        cv2.destroyAllWindows()

    @staticmethod
    def detect_and_mask_edges(image_path, output_path, roi_corners):
        """
        Applique la détection des contours Canny sur une image et conserve uniquement les contours
        dans une région d'intérêt définie.

        Arguments :
        - image_path : str : Chemin de l'image d'entrée.
        - output_path : str : Chemin pour sauvegarder l'image résultante.
        - roi_corners : list : Liste des points définissant un polygone pour la région d'intérêt (ROI).

        Exemple de roi_corners :
        roi_corners = [[(50, 720), (640, 360), (1230, 720)]]
        """
        # Charger l'image en niveaux de gris
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Impossible de charger l'image à partir du chemin spécifié.")

        # Appliquer la détection des contours (Canny)
        edges = cv2.Canny(image, threshold1=100, threshold2=200)

        # Créer un masque noir de la même taille que l'image
        mask = np.zeros_like(edges)

        # Convertir les coins de la région d'intérêt en un tableau numpy
        roi_corners_np = np.array(roi_corners, dtype=np.int32)

        # Dessiner la région d'intérêt sur le masque
        cv2.fillPoly(mask, roi_corners_np, 255)

        # Appliquer le masque sur l'image des contours
        masked_edges = cv2.bitwise_and(edges, mask)

        # Sauvegarder le résultat final
        cv2.imwrite(output_path, masked_edges)

        return masked_edges

    @staticmethod
    def extract_large_curves(edges, output_path = None):
        """
        Extrait et redessine uniquement les grandes courbes à partir d'une image d'entrée.

        Arguments :
        - image_path : str : Chemin de l'image d'entrée.
        - output_path : str : Chemin pour sauvegarder l'image résultante.

        Retourne :
        - L'image avec les grandes courbes dessinées.
        """
        # Charger l'image en niveaux de gris
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # if image is None:
        #     raise ValueError("Impossible de charger l'image à partir du chemin spécifié.")

        # Appliquer la détection des contours avec Canny
        # edges = cv2.Canny(image, threshold1=100, threshold2=200)

        # Appliquer la détection de lignes/courbes avec la Transformée de Hough
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=30)

        # Créer une image vide pour dessiner les grandes courbes
        curves_image = np.zeros_like(edges)

        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(curves_image, (x1, y1), (x2, y2), 255, thickness=2)

        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]

        #         # Calculer les différences en x et y
        #         delta_x = abs(x2 - x1)
        #         delta_y = abs(y2 - y1)

        #         # Conserver uniquement les lignes où delta_y > delta_x (plus verticales que horizontales)
        #         if delta_y > delta_x:
        #             cv2.line(curves_image, (x1, y1), (x2, y2), 255, thickness=2)

        if lines is not None:
            # Stocker les lignes avec leurs longueurs
            line_lengths = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # # Calculer les différences en x et y
                # delta_x = abs(x2 - x1)
                # delta_y = abs(y2 - y1)

                # # Conserver uniquement les lignes où delta_y > delta_x (plus verticales que horizontales)
                # if delta_y > delta_x:
                #     length = np.sqrt(delta_x**2 + delta_y**2)  # Calculer la longueur de la ligne
                #     line_lengths.append(((x1, y1, x2, y2), length))

            # Trier les lignes par longueur décroissante
            line_lengths = sorted(line_lengths, key=lambda x: x[1], reverse=True)

            # Conserver les `max_lines` lignes les plus longues
            top_lines = line_lengths[:100]

            # Dessiner ces lignes sur l'image
            for (x1, y1, x2, y2), _ in top_lines:
                cv2.line(curves_image, (x1, y1), (x2, y2), 255, thickness=2)

        # Sauvegarder l'image des grandes courbes
        if output_path is not None:
            cv2.imwrite(output_path, curves_image)

        return curves_image

    @staticmethod
    def extract_centered_curves(edges, output_path=None, width_factor=0.3):
        """
        Extrait et redessine les courbes les plus proches du centre horizontal de l'image.

        Arguments :
        - edges : np.ndarray : Image contenant les contours détectés (résultat de Canny).
        - output_path : str : Chemin pour sauvegarder l'image résultante (optionnel).
        - width_factor : float : Facteur de largeur pour la zone centrale (par défaut 0.3, soit 30%).

        Retourne :
        - L'image avec les courbes/lignes assemblées et centrées.
        """
        # Dimensions de l'image
        height, width = edges.shape
        center_x = width // 2

        # Définir la zone centrale
        central_min_x = int(center_x - (width * width_factor / 2))
        central_max_x = int(center_x + (width * width_factor / 2))

        # Trouver les contours (pixels actifs)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Créer une image vide pour les courbes centrales
        centered_image = np.zeros_like(edges)

        # Filtrer les contours proches du centre horizontal
        for contour in contours:
            # Vérifier si au moins un pixel du contour est dans la zone centrale
            for point in contour:
                x, y = point[0]
                if central_min_x <= x <= central_max_x:
                    cv2.drawContours(centered_image, [contour], -1, 255, thickness=2)
                    break

        # Optionnel : relier les fragments de contours dans la région centrale
        # Appliquer une dilatation pour combiner les fragments proches
        kernel = np.ones((5, 5), np.uint8)
        assembled_image = cv2.dilate(centered_image, kernel, iterations=1)

        # Sauvegarder l'image si un chemin est fourni
        if output_path:
            cv2.imwrite(output_path, assembled_image)

        return assembled_image

    @staticmethod
    def new_process_video(video_path, scale=2, interpolation=cv2.INTER_AREA):
        """
        Traite une vidéo, détecte les contours, redessine les grandes courbes, et sauvegarde les sorties.

        Arguments :
        - video_path (str) : Chemin de la vidéo d'entrée.
        - scale (int) : Facteur d'échelle pour redimensionner les frames.
        - interpolation : Méthode d'interpolation pour redimensionner.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur lors de l'ouverture de la vidéo : {video_path}")
            return

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = min(25, input_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // scale
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // scale

        # Calculate the middle third vertically
        third_height = height // 3
        start_y = 3* third_height
        end_y = 7 * third_height
        width = width *2

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_original = cv2.VideoWriter('original_output.avi', fourcc, fps, (width, end_y - start_y), False)
        out_blur = cv2.VideoWriter('blur_output.avi', fourcc, fps, (width, end_y - start_y), False)
        out_edges = cv2.VideoWriter('edges_output.avi', fourcc, fps, (width, end_y - start_y), False)
        out_curves = cv2.VideoWriter('curves_output.avi', fourcc, fps, (width, end_y - start_y), False)
        delay = int(1000 / fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the middle third vertically
            frame_cropped = frame[start_y:end_y, :]

            # Contrast
            contrasted =  cv2.convertScaleAbs(frame_cropped, alpha=0.9, beta=0)

            # Convertir en niveaux de gris et redimensionner
            gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (width,  end_y - start_y), interpolation=interpolation)

            # Appliquer un flou pour réduire le bruit
            blur = cv2.GaussianBlur(gray_resized, (9, 9), 0)


            # Calculer l'histogramme de l'image
            min_val = max(0, np.percentile(blur, 15))  # Abaissez le seuil inférieur
            max_val = min(255, np.percentile(blur, 85))  # Relevez le seuil supérieur

            # Appliquer Canny avec des seuils dynamiques
            edges = cv2.Canny(blur, min_val, max_val, L2gradient=True)

            # Détection par segmentation des couleurs pour le blanc
            hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)  # Convertir en HSV pour mieux isoler les couleurs
            mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))  # Masque pour isoler les blancs

            # Nettoyage du masque pour réduire les petits points lumineux
            kernel = np.ones((5, 5), np.uint8)  # Kernel pour les opérations morphologiques
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)  # Ferme les trous
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, morph_kernel)  # Supprime les petits objets

            # Flou gaussien pour adoucir les bords du masque
            mask_cleaned = cv2.GaussianBlur(mask_cleaned, (5, 5), 0)

            # Redimensionner le masque pour qu'il corresponde à gray_resized
            mask_resized = cv2.resize(mask, (width,  end_y - start_y), interpolation=interpolation)

            # S'assurer que le masque est en type CV_8U
            mask_resized = mask_resized.astype(np.uint8)

            # Appliquer le masque à l'image grise redimensionnée
            masked_image = cv2.bitwise_and(gray_resized, gray_resized, mask=mask_resized)

            # Fusionner les résultats des blancs avec les contours détectés
            edges = cv2.bitwise_or(edges, masked_image)

            # Détection des grandes courbes
            # curves = ImageEdgeProcessor.extract_large_curves(edges)
            curves = ImageEdgeProcessor.extract_centered_curves(edges)


            # Sauvegarder chaque frame dans les vidéos
            out_original.write(gray_resized)
            out_blur.write(blur)
            out_edges.write(edges)
            out_curves.write(curves)


            # Empilez les deux rangées verticalement
            combined = np.hstack((gray_resized, blur, edges))
            cv2.imshow('Original, Edges, and Curves', combined)

            # Arrêter si l'utilisateur appuie sur 'q'
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        # Libérer les ressources
        cap.release()
        out_original.release()
        out_edges.release()
        out_curves.release()
        cv2.destroyAllWindows()
