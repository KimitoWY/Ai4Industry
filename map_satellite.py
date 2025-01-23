import cv2
import os
import numpy as np
import folium

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:04d}.png'), frame)
        frame_count += 1
    
    cap.release()
    return frame_count

#frame_count = extract_frames('./data/20240914_target.mp4', 'output_frames')

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
    
    prev_pts = np.array([kp.pt for kp in prev_keypoints], dtype=np.float32).reshape(-1, 1, 2)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_pts, None)
    
    good_prev_pts = prev_pts[status == 1]
    good_next_pts = next_pts[status == 1]
    
    return good_prev_pts, good_next_pts

def estimate_motion(good_prev_pts, good_next_pts):
    E, mask = cv2.findEssentialMat(good_next_pts, good_prev_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, good_next_pts, good_prev_pts)
    return R, t

def plot_route(movements):
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    coords = [[0, 0]]
    for R, t in movements:
        last_coord = coords[-1]
        new_coord = [last_coord[0] + t[0][0], last_coord[1] + t[2][0]]
        coords.append(new_coord)
    
    folium.PolyLine(coords, color="blue", weight=2.5, opacity=1).add_to(m)
    m.save('route_map.html')

# Définir les région a cacher (x1, y1, x2, y2)
car_region = [((0, 350), (1920, 0)), ((0, 1080), (1920, 650))]

# Créer le masque
image = cv2.imread('output_frames/frame_0000.png', cv2.IMREAD_GRAYSCALE)
mask = create_mask(image.shape, car_region)



# Détecter les caractéristiques avec le masque
prev_keypoints, descriptors = detect_features('output_frames/frame_0000.png')

# Filtrer les keypoints
filtered_keypoints = filter_keypoints(prev_keypoints, min_response=0.001, min_size=2)

# Dessiner les points d'intérêt sur l'image
image_with_keypoints = cv2.drawKeypoints(cv2.imread('output_frames/frame_0000.png'), filtered_keypoints, None, color=(0, 255, 0))
cv2.imwrite("frame_0000_keypoints.png", image_with_keypoints)

'''
# Suivre les caractéristiques
good_prev_pts, good_next_pts = track_features('output_frames/frame_0000.png', 'output_frames/frame_0001.png', prev_keypoints)
R, t = estimate_motion(good_prev_pts, good_next_pts)


frame_count = 1575
movements = []
# Boucle sur les images pour suivre le mouvement
for i in range(frame_count - 1):
    image = cv2.imread(f'output_frames/frame_{i:04d}.png', cv2.IMREAD_GRAYSCALE)
    mask = create_mask(image.shape, car_region)
    prev_keypoints, _ = detect_features(f'output_frames/frame_{i:04d}.png',mask)
    good_prev_pts, good_next_pts = track_features(f'output_frames/frame_{i:04d}.png', f'output_frames/frame_{i+1:04d}.png', prev_keypoints)
    R, t = estimate_motion(good_prev_pts, good_next_pts)
    movements.append((R, t))
# Afficher la carte
plot_route(movements)
'''