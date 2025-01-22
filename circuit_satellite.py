import cv2
import numpy as np
from shapely.geometry import LineString
import geopandas as gpd

def extract_and_reconstruct(video_path, satellite_image_path, output_image_path, geojson_path):
    """
    Reconstitue un circuit à partir d'une vidéo avec contours et superpose sur une vue satellite.
    Exporte également le circuit en format GeoJSON.

    Args:
        video_path (str): Chemin de la vidéo avec contours.
        satellite_image_path (str): Chemin de l'image satellite.
        output_image_path (str): Chemin pour sauvegarder l'image finale avec superposition.
        geojson_path (str): Chemin pour sauvegarder le fichier GeoJSON.
    """
    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        raise ValueError("Impossible de lire la vidéo.")

    # Créer un canevas noir pour accumuler les contours
    height, width = frame.shape[:2]
    accumulated_contours = np.zeros((height, width), dtype=np.uint8)

    while ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        accumulated_contours = cv2.bitwise_or(accumulated_contours, gray_frame)
        ret, frame = cap.read()

    cap.release()

    # Sauvegarder l'image des contours reconstitués
    cv2.imwrite('circuit_reconstruit.png', accumulated_contours)

    # Points de correspondance manuels (à ajuster)
    # Remplacez-les par les points correspondant à votre projet
    src_points = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32)  # Pixels de l'image contours
    dst_points = np.array([[10, 20], [300, 20], [300, 200], [10, 200]], dtype=np.float32)      # Coordonnées satellite

    # Calculer la matrice d'homographie
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)

    # Charger l'image satellite
    satellite_image = cv2.imread(satellite_image_path)
    if satellite_image is None:
        raise ValueError("Impossible de lire l'image satellite.")

    # Transformer les contours avec l'homographie
    warped_contours = cv2.warpPerspective(accumulated_contours, homography_matrix,
                                          (satellite_image.shape[1], satellite_image.shape[0]))

    # Superposer les contours sur l'image satellite
    overlay = cv2.addWeighted(satellite_image, 0.7, cv2.cvtColor(warped_contours, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # Sauvegarder l'image finale
    cv2.imwrite(output_image_path, overlay)

    # Convertir les contours en format GeoJSON (exemple simplifié)
    # Trouver les contours dans l'image reconstituée
    contours, _ = cv2.findContours(accumulated_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        points = contour.squeeze().tolist()  # Extraire les points du contour
        if len(points) > 1:  # Ignorer les petits artefacts
            line = LineString(points)
            lines.append(line)

    # Créer un GeoDataFrame avec les lignes
    gdf = gpd.GeoDataFrame({'geometry': lines}, crs="EPSG:4326")  # EPSG:4326 = coordonnées géographiques

    # Exporter en GeoJSON
    gdf.to_file(geojson_path, driver='GeoJSON')

# Appel de la fonction principale
extract_and_reconstruct(
    video_path='curves_output.avi',
    satellite_image_path='Ancenis.png',
    output_image_path='circuit_sur_satellite.png',
    geojson_path='circuit.geojson'
)
