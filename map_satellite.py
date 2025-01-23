import cv2
import os
import numpy as np
import folium

def detect_features(image_path, mask=None):
    # Lire l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Créer un détecteur ORB
    orb = cv2.ORB_create()
    
    # Détecter les points d'intérêt et calculer les descripteurs
    keypoints, descriptors = orb.detectAndCompute(image, mask)
    
    return keypoints, descriptors

def create_mask(image_shape, regions):
    mask = np.ones(image_shape, dtype=np.uint8) * 255  # Masque blanc (255)
    for region in regions:
        cv2.rectangle(mask, region[0], region[1], 0, -1)  # Dessiner un rectangle noir (0) sur chaque région
    return mask

def filter_keypoints(keypoints, min_response=0.01, min_size=5):
    filtered_keypoints = [kp for kp in keypoints if kp.response >= min_response and kp.size >= min_size]
    return filtered_keypoints

def track_features(prev_image_path, next_image_path, prev_keypoints):
    prev_image = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
    next_image = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)

    prev_image = prev_image.copy()  # Assurez-vous que l'image est continue
    next_image = next_image.copy()  # Assurez-vous que l'image est continue

    
    prev_pts = np.array([kp.pt for kp in prev_keypoints], dtype=np.float32).reshape(-1, 1, 2)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_pts, None)
    
    good_prev_pts = prev_pts[status == 1]
    good_next_pts = next_pts[status == 1]
    
    return good_prev_pts, good_next_pts

def estimate_motion(good_prev_pts, good_next_pts):
    E, mask = cv2.findEssentialMat(good_next_pts, good_prev_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, good_next_pts, good_prev_pts)
    return R, t

def rotate_point(x, y, angle):
    radians = np.deg2rad(angle)
    cos = np.cos(radians)
    sin = np.sin(radians)
    nx = cos * x - sin * y
    ny = sin * x + cos * y
    return nx, ny

def plot_route(movements,image_path='data/Ancenis.png'):
    # Charger l'image de fond
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Initialiser les coordonnées
    coords = [[0, 0]]
    current_position = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
    current_rotation = np.eye(3, dtype=np.float32)
    count = 0
    for R, t in movements:
        # Mettre à jour la position actuelle en appliquant la rotation et la translation
        current_position += current_rotation @ t
        current_rotation = R @ current_rotation
        
        # Ajouter les nouvelles coordonnées à la liste
        coords.append([current_position[0, 0], current_position[2, 0]])
        count += 1
        
    position_factor_x = 3000  # Ajuster ce facteur selon les dimensions de l'image
    position_factor_y = 1500  # Ajuster ce facteur selon les dimensions de l'image
    scale_factor = 100  # Adjust this factor as needed to fit the image dimensions
    rotation_factor = 180  # Adjust this factor as needed for rotation in degrees


    # Tracer la route sur l'image
    for i in range(0, count - 2):
        pt1 = rotate_point(coords[i][0], coords[i][1], rotation_factor)
        pt2 = rotate_point(coords[i + 1][0], coords[i + 1][1], rotation_factor)
        
        
        pt1 = pt1[0] * scale_factor + position_factor_x, pt1[1] * scale_factor + position_factor_y
        pt2 = pt2[0] * scale_factor + position_factor_x, pt2[1] * scale_factor + position_factor_y
        
        
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        print(f"Drawing line from {pt1} to {pt2}")  # Debug print
        cv2.line(image, pt1, pt2, (255, 0, 0), 6)
    
    # Sauvegarder l'image avec la route tracée
    cv2.imwrite("route_map.png", image)

def generate_map():
    frame_count = 1574
    movements = []
    if not os.path.exists("result"):
            os.makedirs("result")
    # Boucle sur les images pour suivre le mouvement
    # for i in range(frame_count - 1):
    for i in range(0, frame_count - 1,4):
        image_path = f'output_find_contours/frame_{i+1}.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Image not found: {image_path}")
            continue
        prev_keypoints, _ = detect_features(image_path)
        filtered_keypoints = filter_keypoints(prev_keypoints, min_response=0.006, min_size=5)

        # Dessiner les points d'intérêt sur l'image
        image_with_keypoints = cv2.drawKeypoints(image, filtered_keypoints, None, color=(0, 255, 0))
        cv2.imwrite(f"result/frame_{i:04d}_keypoints.png", image_with_keypoints)

        good_prev_pts, good_next_pts = track_features(image_path, f'output_find_contours/frame_{i+2}.png', filtered_keypoints)
        # print(i)
        R, t = estimate_motion(good_prev_pts, good_next_pts)
        movements.append((R, t))
    # Afficher la carte
    plot_route(movements)

generate_map()
