import cv2
import numpy as np

def process_video(input_video_path, scale=1):
    # Charger la vidéo ou utiliser la webcam
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo : {input_video_path}")
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = min(25, input_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // scale
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // scale

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_find_contours = cv2.VideoWriter('output_video_find_contours.avi', fourcc, fps, (width, height), True)

    while True:
        # Lire une frame
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo ou erreur

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Appliquer un flou pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Détection des bords avec Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Trouver les contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer et classer les contours par longueur décroissante
        sorted_contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:4]  # Garde les 6 plus grandes courbes

        # Dessiner les contours sur l'image originale
        for contour in sorted_contours:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        # Redimensionner la frame
        resized_frame = cv2.resize(frame, (width, height))

        # Sauvegarder chaque frame en PNG
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"{frame_number}")
        myframe = cv2.imwrite(f"./output_find_contours/frame_{frame_number}.png", resized_frame)
        if myframe == None:
            print("Erreur lors de la sauvegarde de la frame")
        


        # Écrire la frame dans la vidéo de sortie
        out_find_contours.write(resized_frame)

        # Afficher la vidéo avec les contours détectés
        cv2.imshow("Contours - Top 6", resized_frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    out_find_contours.release()
    cv2.destroyAllWindows()

def draw_circuit(input_video_path, output_video_path, scale=1):
    def are_parallel(line1, line2, angle_threshold=10):
        def angle(line):
            x1, y1, x2, y2 = line[0]
            return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        angle1 = angle(line1)
        angle2 = angle(line2)
        return abs(angle1 - angle2) < angle_threshold

    # Charger la vidéo ou utiliser la webcam
    video = cv2.VideoCapture(input_video_path)

    input_fps = video.get(cv2.CAP_PROP_FPS)
    fps = min(25, input_fps)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) // scale
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) // scale

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_find_contours = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), False)

    while True:
        # Lire une frame
        ret, frame = video.read()
        if not ret:
            break  # Fin de la vidéo ou erreur

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Appliquer un flou pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Détection des bords avec Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Trouver les lignes avec HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

        if lines is not None:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    if are_parallel(lines[i], lines[j]):
                        cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 2)
                        cv2.line(frame, (lines[j][0][0], lines[j][0][1]), (lines[j][0][2], lines[j][0][3]), (0, 255, 0), 2)

        # Afficher la vidéo avec les contours détectés
        cv2.imshow("Circuit", frame)        

        # Écrire la frame dans la vidéo de sortie
        out_find_contours.write(cv2.resize(frame, (width, height)))

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    video.release()
    out_find_contours.release()
    cv2.destroyAllWindows()

def map_3d_to_2d(satellite_image_path, first_person_video_path, output_image_path):
    # Charger l'image satellite
    satellite_image = cv2.imread(satellite_image_path)
    satellite_gray = cv2.cvtColor(satellite_image, cv2.COLOR_BGR2GRAY)

    # Charger la vidéo de vue à la première personne
    video = cv2.VideoCapture(first_person_video_path)
    input_fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Détection des caractéristiques avec ORB pour l'image satellite
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(satellite_gray, None)

    # Trouver les contours dans l'image satellite
    edges = cv2.Canny(satellite_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l'image satellite
    cv2.drawContours(satellite_image, contours, -1, (0, 255, 0), 2)

    # Créer une image blanche pour dessiner le circuit
    traced_circuit = np.ones((satellite_image.shape[0], satellite_image.shape[1]), dtype=np.uint8) * 255

    while True:
        # Lire une frame de la vidéo
        ret, frame = video.read()
        if not ret:
            break  # Fin de la vidéo ou erreur

        # Convertir en niveaux de gris
        first_person_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection des caractéristiques avec ORB pour la frame de la vidéo
        keypoints2, descriptors2 = orb.detectAndCompute(first_person_gray, None)

        # Correspondance des caractéristiques avec BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Dessiner les correspondances
        result_frame = cv2.drawMatches(satellite_image, keypoints1, frame, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Trouver les lignes avec HoughLinesP
        edges = cv2.Canny(first_person_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(traced_circuit, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Afficher l'image avec les correspondances
        cv2.imshow("Matches", result_frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Sauvegarder l'image tracée
    cv2.imwrite(output_image_path, traced_circuit)

    # Libérer les ressources
    video.release()
    cv2.destroyAllWindows()

# Example usage:
process_video("edges_output.avi")
# draw_circuit("edges_output.avi", "circuit_output.avi")
# map_3d_to_2d("Ancenis.png", "edges_output.avi", "mapped_output.jpg")
