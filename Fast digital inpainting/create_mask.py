import cv2
import numpy as np

def create_mask(image_path, output_mask_path):
    """
    Crée un masque manuellement en permettant à l'utilisateur de dessiner sur l'image.

    Parameters:
    - image_path: str, chemin vers l'image d'entrée
    - output_mask_path: str, chemin vers le fichier de sortie du masque

    Returns:
    - mask: np.ndarray, l'image du masque binaire créée
    """
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin.")

    # Créer une image de masque vide (noire)
    mask = np.zeros_like(image, dtype=np.uint8)

    # Fonction de dessin
    def draw_mask(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(mask, (x, y), 5, (255), -1)

    # Créer une fenêtre pour afficher l'image et dessiner le masque
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_mask)

    while True:
        # Afficher l'image avec le masque superposé
        combined_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        cv2.imshow('image', combined_image)
        
        # Quitter la fenêtre avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Sauvegarder le masque
    cv2.imwrite(output_mask_path, mask)
    cv2.destroyAllWindows()
    
    return mask

