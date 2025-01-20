import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def display_images(original, edges):
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.imshow(original, cmap='gray')
        plt.title('Image d\'entrée')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(edges, cmap='gray')
        plt.title('Contours détectés')
        plt.axis('off')

        plt.show()

def detect_edges(image_path, scale=2, interpolation=cv2.INTER_AREA):
    # Charger l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionner l'image avec une méthode d'interpolation pour réduire le bruit
    width = image.shape[1] // scale
    height = image.shape[0] // scale
    image = cv2.resize(image, (width, height), interpolation=interpolation)

    print("taille de l'image: ", image.shape, "\n")
    # Détecter les contours
    # Les valeurs 100 et 200 représentent les seuils inférieur et supérieur pour la détection des contours
    edges = cv2.Canny(image, 150, 250)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return image, edges

def process_images_from_folder(folder_path, start_index, end_index, scale=2, interpolation=cv2.INTER_AREA):
    for i in range(start_index, end_index + 1):
        image_path = os.path.join(folder_path, f'frame_{i}.png')
        if os.path.exists(image_path):
            image, edges = detect_edges(image_path, scale, interpolation)
            display_images(image, edges)
        else:
            print(f"Image {image_path} not found.")


# def process_video(video_path, scale=2, interpolation=cv2.INTER_AREA):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error opening video file {video_path}")
#         return

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // scale
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // scale
#     # fps = cap.get(cv2.CAP_PROP_FPS)
#     fps = 25

#     # Define the codec and create VideoWriter objects
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out_original = cv2.VideoWriter('original_output.avi', fourcc, fps, (width, height), False)
#     out_edges = cv2.VideoWriter('edges_output.avi', fourcc, fps, (width, height), False)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Resize the frame
#         gray_resized = cv2.resize(gray, (width, height), interpolation=interpolation)
#         # Detect edges
#         edges = cv2.Canny(gray_resized, 200, 300)
#         kernel = np.ones((5,5), np.uint8)
#         edges = cv2.dilate(edges, kernel, iterations=1)

#         # Write the frames
#         out_original.write(gray_resized)
#         out_edges.write(edges)

#         # Display the frames side by side
#         combined = np.hstack((gray_resized, edges))
#         cv2.imshow('Original and Edges', combined)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out_original.release()
#     out_edges.release()
#     cv2.destroyAllWindows()

def process_video(video_path, scale=2, interpolation=cv2.INTER_AREA):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)  # Fréquence d'images de la vidéo d'entrée
    fps = min(25, input_fps)  # Limiter à 25 fps maximum
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // scale
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // scale

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_original = cv2.VideoWriter('original_output.avi', fourcc, fps, (width, height), False)
    out_edges = cv2.VideoWriter('edges_output.avi', fourcc, fps, (width, height), False)

    delay = int(1000 / fps)  # Calcul du délai pour affichage des frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the frame
        gray_resized = cv2.resize(gray, (width, height), interpolation=interpolation)
        # Detect edges
        edges = cv2.Canny(gray_resized, 150, 380)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Write the frames
        out_original.write(gray_resized)
        out_edges.write(edges)

        # Display the frames side by side
        combined = np.hstack((gray_resized, edges))
        cv2.imshow('Original and Edges', combined)

        if cv2.waitKey(delay) & 0xFF == ord('q'):  # Temporisation ajustée
            break

    cap.release()
    out_original.release()
    out_edges.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Exemple d'utilisation
    # image, edges = detect_edges('test.jpg')

    # Afficher l'image d'entrée et de sortie
    # display_images(image, edges)

    # Charger les images d'un dossier
    # process_images_from_folder('../imageTest/', 1, 2, 8)

    # Charger une vidéo
    process_video('../videoTest/test.mp4', 4)

