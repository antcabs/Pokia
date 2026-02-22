import cv2
import numpy as np
import math
import os

# Variables globales pour le modèle d'apprentissage
ML_DISPONIBLE = False

def scanner_carte_pokemon(chemin_image, afficher_etapes=True, angle_rotation=None):
    # 1. Charger l'image
    image_originale = cv2.imread(chemin_image)
    if image_originale is None:
        print(f"Impossible de charger l'image à partir de {chemin_image}")
        return None
    
    if afficher_etapes:
        cv2.imshow("Image originale", image_originale)
        cv2.waitKey(0)
    
    # Redimensionner l'image si elle est trop grande pour améliorer la performance
    hauteur, largeur = image_originale.shape[:2]
    max_dimension = 1200
    if max(hauteur, largeur) > max_dimension:
        facteur = max_dimension / float(max(hauteur, largeur))
        image = cv2.resize(image_originale, None, fx=facteur, fy=facteur)
        print(f"Image redimensionnée de {largeur}x{hauteur} à {int(largeur*facteur)}x{int(hauteur*facteur)}")
        
        if afficher_etapes:
            cv2.imshow("Image redimensionnée", image)
            cv2.waitKey(0)
    else:
        image = image_originale.copy()
    
    # 2. Prétraitement avancé pour améliorer la détection des bords
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if afficher_etapes:
        cv2.imshow("Image en niveaux de gris", gris)
        cv2.waitKey(0)
        
    flou = cv2.GaussianBlur(gris, (5, 5), 0)
    
    if afficher_etapes:
        cv2.imshow("Image floutée", flou)
        cv2.waitKey(0)
    
    # Utiliser Canny pour une meilleure détection des bords
    canny = cv2.Canny(flou, 20, 90)
    
    if afficher_etapes:
        cv2.imshow("Détection de contours (Canny)", canny)
        cv2.waitKey(0)
    
    # Dilatation pour fermer les contours
    kernel = np.ones((7, 7), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=3)
    
    if afficher_etapes:
        cv2.imshow("Dilatation des contours", dilate)
        cv2.waitKey(0)
    
    # 3. Trouver les contours
    contours, _ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours par aire
    contours_filtres = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
    
    if afficher_etapes and contours_filtres:
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours_filtres, -1, (0, 255, 0), 2)
        cv2.imshow("Contours filtrés", contour_img)
        cv2.waitKey(0)
    
    if not contours_filtres:
        print("Aucun contour significatif trouvé, essai avec méthode alternative...")
        # Méthode alternative: threshold adaptatif
        thresh = cv2.adaptiveThreshold(flou, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        if afficher_etapes:
            cv2.imshow("Seuillage adaptatif", thresh)
            cv2.waitKey(0)
            
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_filtres = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
        
        if afficher_etapes and contours_filtres:
            contour_img = image.copy()
            cv2.drawContours(contour_img, contours_filtres, -1, (0, 255, 0), 2)
            cv2.imshow("Contours alternatifs", contour_img)
            cv2.waitKey(0)
    
    if not contours_filtres:
        print("Aucun contour de carte détecté")
        return None
    
    # Trouver le contour le plus rectangulaire (ratio largeur/hauteur proche de 6.3/8.8)
    meilleur_contour = None
    meilleur_score = float('inf')
    ratio_cible = 2.5 / 3.5  # Ratio largeur/hauteur d'une carte Pokémon
    
    for contour in contours_filtres:
        # Calculer le rectangle englobant
        rect = cv2.minAreaRect(contour)
        largeur, hauteur = rect[1]
        if largeur < hauteur:  # Assurer que largeur est le côté le plus court
            largeur, hauteur = hauteur, largeur
        
        # Calculer le ratio et la différence avec le ratio cible
        if hauteur > 0:  # Éviter division par zéro
            ratio = largeur / hauteur
            difference_ratio = abs(ratio - ratio_cible)
            
            # Calculer le score (différence de ratio et rectangularité)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
            rectangularite = abs(len(approx) - 4)  # 0 si exactement 4 points
            
            score = difference_ratio + rectangularite * 0.5
            
            if score < meilleur_score:
                meilleur_score = score
                meilleur_contour = contour
    
    if meilleur_contour is None:
        print("Aucun contour rectangulaire trouvé")
        return None
    
    if afficher_etapes:
        meilleur_contour_img = image.copy()
        cv2.drawContours(meilleur_contour_img, [meilleur_contour], -1, (0, 0, 255), 3)
        cv2.imshow("Meilleur contour", meilleur_contour_img)
        cv2.waitKey(0)
    
    # 4. Approximer le contour en rectangle
    peri = cv2.arcLength(meilleur_contour, True)
    approx = cv2.approxPolyDP(meilleur_contour, 0.02 * peri, True)
    
    # Si nous avons trop de points ou pas assez, forcer un rectangle
    if len(approx) != 4:
        print(f"Approximation a retourné {len(approx)} points, forçage du rectangle...")
        rect = cv2.minAreaRect(meilleur_contour)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)
    
    if afficher_etapes:
        approx_img = image.copy()
        cv2.drawContours(approx_img, [approx], -1, (255, 0, 0), 3)
        cv2.imshow("Approximation rectangulaire", approx_img)
        cv2.waitKey(0)
    ## 5. Redresser l'image
    points = np.array([pt[0] for pt in approx] if len(approx.shape) > 2 else approx)
    points = ordre_points(points)

    # 1. Appliquer une rotation préliminaire pour redresser la carte
    image_redresse, points = redresser_rotation(image, points, angle_fixe=angle_rotation)

    if afficher_etapes:
        pre_redresse_img = image_redresse.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(pre_redresse_img, (int(x), int(y)), 10, (0, 255, 255), -1)
            cv2.putText(pre_redresse_img, str(i), (int(x) - 20, int(y) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Image pré-redressée", pre_redresse_img)
        cv2.waitKey(0)

    # 2. Forcer un rectangle parfaitement orthogonal
    points = forcer_rectangle_parfait(points)

    # Appliquer les autres améliorations que vous avez déjà
    points = normaliser_rotation(points)
    points = ajouter_marge_securite(points, marge_pourcentage=0.02)
    points = affiner_points_contour(gris, points, taille_fenetre=15)

    # Dimensions cibles en pixels (conversion de cm à pixels avec 300 DPI)
    # Carte Pokémon standard: 6.3cm x 8.8cm (orientation verticale)
    largeur_cible = int(6.3 * 118.11)  # 300 DPI ≈ 118.11 pixels/cm
    hauteur_cible = int(8.8 * 118.11)
    
    # Vérifier l'orientation et corriger si nécessaire
    # Si la largeur détectée est plus grande que la hauteur, la carte est horizontale
    rect_width = np.linalg.norm(points[1] - points[0])
    rect_height = np.linalg.norm(points[3] - points[0])
    
    if rect_width > rect_height:
        # La carte est horizontale, il faut la faire pivoter de 90 degrés
        # Inverser les dimensions cibles
        largeur_cible, hauteur_cible = hauteur_cible, largeur_cible
        # Réorganiser les points pour une rotation de 90 degrés dans le sens horaire
        # Nouveau ordre: point[1] devient le coin supérieur gauche
        points = np.array([points[1], points[2], points[3], points[0]])
    
    if afficher_etapes:
        # Dessiner les points ordonnés
        points_img = image_redresse.copy()  # Utiliser l'image redressée comme base
        for i, (x, y) in enumerate(points):
            cv2.circle(points_img, (int(x), int(y)), 10, (0, 255, 255), -1)
            cv2.putText(points_img, str(i), (int(x) - 20, int(y) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Points ordonnés", points_img)
        cv2.waitKey(0)
    
    # Définir les points cibles
    dst = np.array([
        [0, 0],
        [largeur_cible - 1, 0],
        [largeur_cible - 1, hauteur_cible - 1],
        [0, hauteur_cible - 1]
    ], dtype="float32")
        
    # Calculer la matrice de transformation
    M, _ = cv2.findHomography(points.astype(np.float32), dst, cv2.RANSAC, 5.0)
    
    # Appliquer la transformation
    carte_redressee = cv2.warpPerspective(image_redresse, M, (largeur_cible, hauteur_cible), 
                                     flags=cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)

    if afficher_etapes:
        cv2.imshow("Carte redressée", carte_redressee)
        cv2.waitKey(0)
    
    # 6. Ajouter les bords arrondis et la bordure grise
    # 3. Correction des couleurs pour les cartes jaunes/dorées
    # 3. Correction des couleurs pour les cartes jaunes/vertes
    carte_balancee = balance_des_blancs(carte_redressee)
    carte_corrigee = corriger_couleurs(carte_balancee)

    if afficher_etapes:
        cv2.imshow("Balance des blancs", carte_balancee)
        cv2.waitKey(0)
        cv2.imshow("Couleurs corrigées", carte_corrigee)
        cv2.waitKey(0)

    carte_corrigee_rgb = cv2.cvtColor(carte_corrigee, cv2.COLOR_BGR2RGB)

    # 4. Ajouter les bords arrondis
    resultat = ajouter_bordure_arrondie(carte_corrigee_rgb, rayon_coin=15, epaisseur_bordure=8)

    if afficher_etapes:
        cv2.imshow("Résultat final", cv2.cvtColor(resultat, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return resultat
    
    # Après avoir obtenu le rectangle approximatif, ajoutez une légère marge
def ajouter_marge_securite(points, marge_pourcentage=0.02):
    """
    Ajoute une légère marge autour des points du rectangle pour éviter de couper les bords
    """
    # Calculer le centre du rectangle
    centre_x = np.mean(points[:, 0])
    centre_y = np.mean(points[:, 1])
    
    # Ajuster chaque point en l'éloignant du centre
    for i in range(len(points)):
        # Calculer le vecteur du centre au point
        vect_x = points[i, 0] - centre_x
        vect_y = points[i, 1] - centre_y
        
        # Étendre légèrement le point
        points[i, 0] = centre_x + vect_x * (1 + marge_pourcentage)
        points[i, 1] = centre_y + vect_y * (1 + marge_pourcentage)
    
    return points

def affiner_points_contour(image_gris, points, taille_fenetre=15):
    """
    Affine la position des points de contour en trouvant les coins précis
    dans une petite fenêtre autour de chaque point détecté.
    """
    h, w = image_gris.shape
    points_affines = np.copy(points).astype(float)
    
    for i, (x, y) in enumerate(points):
        # S'assurer que la fenêtre ne dépasse pas les limites de l'image
        x1 = max(0, int(x - taille_fenetre))
        y1 = max(0, int(y - taille_fenetre))
        x2 = min(w - 1, int(x + taille_fenetre))
        y2 = min(h - 1, int(y + taille_fenetre))
        
        # Extraire la région d'intérêt
        roi = image_gris[y1:y2, x1:x2]
        
        if roi.size > 0:
            # Détecter les coins dans la ROI avec Harris
            roi_coins = cv2.cornerHarris(roi, 2, 3, 0.04)
            roi_coins = cv2.dilate(roi_coins, None)
            
            # Trouver le coin le plus fort dans la ROI
            if np.max(roi_coins) > 0.01:
                y_loc, x_loc = np.unravel_index(np.argmax(roi_coins), roi_coins.shape)
                # Ajuster les coordonnées au cadre global
                points_affines[i] = [x1 + x_loc, y1 + y_loc]
    
    return points_affines

def normaliser_rotation(points):
    """
    Normalise la rotation des points pour assurer que la carte est parfaitement droite
    """
    # Calculer l'angle moyen des lignes horizontales et verticales
    angle_haut = np.arctan2(points[1, 1] - points[0, 1], points[1, 0] - points[0, 0])
    angle_bas = np.arctan2(points[2, 1] - points[3, 1], points[2, 0] - points[3, 0])
    angle_gauche = np.arctan2(points[3, 1] - points[0, 1], points[3, 0] - points[0, 0])
    angle_droit = np.arctan2(points[2, 1] - points[1, 1], points[2, 0] - points[1, 0])
    
    # Forcer une géométrie rectangulaire parfaite
    # Calculer les dimensions moyennes
    largeur_haut = np.sqrt((points[1, 0] - points[0, 0])**2 + (points[1, 1] - points[0, 1])**2)
    largeur_bas = np.sqrt((points[2, 0] - points[3, 0])**2 + (points[2, 1] - points[3, 1])**2)
    hauteur_gauche = np.sqrt((points[3, 0] - points[0, 0])**2 + (points[3, 1] - points[0, 1])**2)
    hauteur_droite = np.sqrt((points[2, 0] - points[1, 0])**2 + (points[2, 1] - points[1, 1])**2)
    
    largeur_moy = (largeur_haut + largeur_bas) / 2
    hauteur_moy = (hauteur_gauche + hauteur_droite) / 2
    
    # Créer un rectangle parfait en utilisant le coin supérieur gauche comme référence
    rect_parfait = np.array([
        points[0].copy(),  # Haut gauche (inchangé)
        [points[0, 0] + largeur_moy, points[0, 1]],  # Haut droit
        [points[0, 0] + largeur_moy, points[0, 1] + hauteur_moy],  # Bas droit
        [points[0, 0], points[0, 1] + hauteur_moy]  # Bas gauche
    ])
    
    return rect_parfait
    
    # Après avoir trouvé les points et avant de les utiliser pour la transformation
    # Utiliser une technique plus robuste pour redresser
def redresser_rotation(image, points, angle_fixe=None):
    """
    Redresse l'image en fonction de l'angle détecté ou d'un angle fixe spécifié
    """
    # Si un angle fixe est fourni, l'utiliser directement
    if angle_fixe is not None:
        angle_moyen = angle_fixe
    else:
        # Calculer plus précisément l'angle
        dx1 = points[1, 0] - points[0, 0]  # Côté supérieur
        dy1 = points[1, 1] - points[0, 1]
        angle1 = np.arctan2(dy1, dx1) * 180 / np.pi
        
        dx2 = points[2, 0] - points[3, 0]  # Côté inférieur
        dy2 = points[2, 1] - points[3, 1]
        angle2 = np.arctan2(dy2, dx2) * 180 / np.pi
        
        # Calculer l'angle moyen
        angle_moyen = (angle1 + angle2) / 2
    
    # Obtenir les dimensions de l'image
    h, w = image.shape[:2]
    centre = (w // 2, h // 2)
    
    # Calculer la matrice de rotation
    matrice_rotation = cv2.getRotationMatrix2D(centre, angle_moyen, 1.0)
    
    # Calculer les nouvelles dimensions après rotation pour éviter la troncature
    cos = np.abs(matrice_rotation[0, 0])
    sin = np.abs(matrice_rotation[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Ajuster la matrice de translation pour centrer l'image
    matrice_rotation[0, 2] += (new_w / 2) - centre[0]
    matrice_rotation[1, 2] += (new_h / 2) - centre[1]
    
    # Appliquer la rotation avec les nouvelles dimensions
    image_redresse = cv2.warpAffine(image, matrice_rotation, (new_w, new_h), 
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Recalculer les coordonnées des points
    points_redresses = np.zeros_like(points)
    for i in range(len(points)):
        x, y = points[i]
        px = matrice_rotation[0, 0] * x + matrice_rotation[0, 1] * y + matrice_rotation[0, 2]
        py = matrice_rotation[1, 0] * x + matrice_rotation[1, 1] * y + matrice_rotation[1, 2]
        points_redresses[i] = [px, py]
    
    return image_redresse, points_redresses
def forcer_rectangle_parfait(points):
    """
    Force les points à former un rectangle parfaitement orthogonal
    """
    # Calculer les dimensions moyennes
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    
    # Recréer des points parfaitement alignés
    return np.array([
        [x_min, y_min],  # Haut gauche
        [x_max, y_min],  # Haut droit
        [x_max, y_max],  # Bas droit
        [x_min, y_max]   # Bas gauche
    ], dtype=np.float32)

def balance_des_blancs(image):
    """Applique une correction de balance des blancs simple"""
    # Méthode simple de balance des blancs avec égalisation des canaux
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    
    # Ajuster les canaux pour neutraliser les dominantes
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * 0.8)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * 0.8)
    
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def corriger_couleurs(image):
    """Corrige les couleurs de l'image pour mieux correspondre à l'original"""
    # Convertir en HSV pour un meilleur contrôle des teintes
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Décaler la teinte pour réduire le bleu et favoriser le jaune/vert
    # La teinte est codée sur 180 degrés en OpenCV (0-179)
    # Réduire les teintes bleues (90-130) et augmenter les teintes jaunes (20-40)
    delta_h = np.zeros_like(h)
    
    # Zones bleues -> teintes plus jaunes
    blue_mask = (h > 90) & (h < 130)
    delta_h[blue_mask] = -40  # Déplacer les bleus vers le vert/jaune
    
    # Appliquer le décalage de teinte
    h = np.clip(h + delta_h, 0, 179).astype(np.uint8)
    
    # Augmenter la saturation pour les verts et jaunes
    green_yellow_mask = (h > 20) & (h < 70)
    s[green_yellow_mask] = np.clip(s[green_yellow_mask] * 1.3, 0, 255).astype(np.uint8)
    
    # Améliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Reconstituer l'image HSV et convertir en BGR
    hsv_corrected = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

def ordre_points(pts):
    """
    Ordonne les 4 points d'un contour rectangulaire dans l'ordre:
    haut-gauche, haut-droite, bas-droite, bas-gauche
    """
    # Initialiser un tableau de coordonnées
    rect = np.zeros((4, 2), dtype="float32")
    
    # Le point avec la plus petite somme est en haut à gauche
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Le point avec la plus petite différence est en haut à droite
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # SUPPRIMEZ CES DEUX LIGNES QUI CRÉENT UNE RÉCURSION INFINIE
    # points = ordre_points(points)
    # points = ajouter_marge_securite(points, marge_pourcentage=0.03)
    
    return rect

def ajouter_bordure_arrondie(image, rayon_coin=15, epaisseur_bordure=8, couleur_bordure=(128, 128, 128)):
    """
    Ajoute des coins arrondis et une bordure grise à l'image
    """
    h, w = image.shape[:2]
    
    # Créer un masque avec des coins arrondis
    masque = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(masque, (0, 0), (w, h), 255, -1)
    
    # Arrondir les coins
    cv2.rectangle(masque, (0, 0), (rayon_coin * 2, rayon_coin * 2), 0, -1)
    cv2.circle(masque, (rayon_coin, rayon_coin), rayon_coin, 255, -1)
    
    cv2.rectangle(masque, (w - rayon_coin * 2, 0), (w, rayon_coin * 2), 0, -1)
    cv2.circle(masque, (w - rayon_coin, rayon_coin), rayon_coin, 255, -1)
    
    cv2.rectangle(masque, (0, h - rayon_coin * 2), (rayon_coin * 2, h), 0, -1)
    cv2.circle(masque, (rayon_coin, h - rayon_coin), rayon_coin, 255, -1)
    
    cv2.rectangle(masque, (w - rayon_coin * 2, h - rayon_coin * 2), (w, h), 0, -1)
    cv2.circle(masque, (w - rayon_coin, h - rayon_coin), rayon_coin, 255, -1)
    
    # Créer une image pour la bordure
    bordure = np.zeros_like(image)
    bordure[:] = couleur_bordure
    
    # Créer un masque pour la zone intérieure (sans la bordure)
    masque_interieur = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(masque_interieur, (epaisseur_bordure, epaisseur_bordure), 
                 (w - epaisseur_bordure, h - epaisseur_bordure), 255, -1)
    
    # Arrondir les coins intérieurs
    rayon_int = max(0, rayon_coin - epaisseur_bordure)
    offset = epaisseur_bordure
    
    cv2.rectangle(masque_interieur, (offset, offset), 
                 (offset + rayon_int * 2, offset + rayon_int * 2), 0, -1)
    cv2.circle(masque_interieur, (offset + rayon_int, offset + rayon_int), rayon_int, 255, -1)
    
    cv2.rectangle(masque_interieur, (w - offset - rayon_int * 2, offset), 
                 (w - offset, offset + rayon_int * 2), 0, -1)
    cv2.circle(masque_interieur, (w - offset - rayon_int, offset + rayon_int), rayon_int, 255, -1)
    
    cv2.rectangle(masque_interieur, (offset, h - offset - rayon_int * 2), 
                 (offset + rayon_int * 2, h - offset), 0, -1)
    cv2.circle(masque_interieur, (offset + rayon_int, h - offset - rayon_int), rayon_int, 255, -1)
    
    cv2.rectangle(masque_interieur, (w - offset - rayon_int * 2, h - offset - rayon_int * 2), 
                 (w - offset, h - offset), 0, -1)
    cv2.circle(masque_interieur, (w - offset - rayon_int, h - offset - rayon_int), rayon_int, 255, -1)
    
    # Combiner pour obtenir le résultat final
    resultat = bordure.copy()
    
    # Appliquer le masque intérieur pour la carte
    idx = np.where(masque_interieur == 255)
    resultat[idx] = image[idx]
    
    # Appliquer le masque extérieur pour les coins arrondis
    masque_resultat = cv2.merge([masque, masque, masque])
    resultat = cv2.bitwise_and(resultat, masque_resultat)
    
    return resultat

def analyser_qualite_carte(image_carte, debug=False):
    """
    Analyse la qualité d'une carte Pokémon redressée et attribue une note selon les critères PSA.
    
    Paramètres:
    image_carte (ndarray): Image de la carte redressée (au format RGB)
    debug (bool): Si True, affiche les images de débogage
    
    Retour:
    tuple: (note globale sur 100, note PSA, détails des notes par catégorie, image annotée)
    """
    # Créer une copie de l'image pour les annotations
    img_annotee = image_carte.copy()
    h, w = image_carte.shape[:2]
    
    # Convertir en niveaux de gris et en LAB pour différentes analyses
    gris = cv2.cvtColor(image_carte, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image_carte, cv2.COLOR_RGB2LAB)
    
    # Dictionnaire pour stocker les résultats
    resultats = {
        'coins': 0,     # sur 30 points
        'bords': 0,     # sur 30 points
        'centrage': 0,  # sur 20 points
        'defauts': []   # liste des défauts détectés
    }
    
    # 1. Analyse des coins (30 points)
    resultats['coins'] = analyser_coins(image_carte, gris, img_annotee, debug)
    
    # 2. Analyse des bords (30 points)
    resultats['bords'] = analyser_bords(image_carte, gris, img_annotee, debug)
    
    
    # 4. Analyse du centrage (20 points)
    resultats['centrage'] = analyser_centrage(image_carte, gris, img_annotee, debug)
    
    # Calculer la note globale
    note_globale = resultats['coins'] + resultats['bords'] + resultats['centrage']
    
    # Convertir en note PSA (sur 10)
    note_psa = convertir_note_psa(note_globale)
    
    # Ajouter la note globale à l'image
    cv2.putText(img_annotee, f"Note: {note_globale}/100 (PSA {note_psa})", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Ajouter les notes par catégorie
    cv2.putText(img_annotee, f"Coins: {resultats['coins']}/30", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img_annotee, f"Bords: {resultats['bords']}/30", 
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(img_annotee, f"Centrage: {resultats['centrage']}/20", 
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return note_globale, note_psa, resultats, img_annotee

def analyser_coins(image, gris, img_annotee, debug=False):
    """Analyse la qualité des coins de la carte (30 points)"""
    h, w = image.shape[:2]
    score_total = 0
    
    # Coordonnées approximatives des 4 coins
    coins = [
        (int(w*0.02), int(h*0.02)),  # Haut gauche
        (int(w*0.98), int(h*0.02)),  # Haut droit
        (int(w*0.98), int(h*0.98)),  # Bas droit
        (int(w*0.02), int(h*0.98))   # Bas gauche
    ]
    
    # Taille de la région d'analyse pour chaque coin
    taille_roi = int(min(w, h) * 0.04)
    
    scores_coins = []
    
    for i, (cx, cy) in enumerate(coins):
        # Extraire la région d'intérêt (ROI) pour le coin
        x1 = max(0, cx - taille_roi//2)
        y1 = max(0, cy - taille_roi//2)
        x2 = min(w, cx + taille_roi//2)
        y2 = min(h, cy + taille_roi//2)
        
        roi = gris[y1:y2, x1:x2]
        
        # Détecter les bords dans la ROI
        roi_edges = cv2.Canny(roi, 100, 200)
        
        # Vérifier la netteté des bords
        nettete = np.mean(roi_edges) / 255
        
        # Vérifier la présence d'usure (différence de contraste)
        std_dev = np.std(roi)
        
        # Combiner les métriques pour un score sur 7.5 (30/4 coins)
        score_coin = min(7.5, (nettete * 3.5) + (min(1, std_dev/50) * 4))
        scores_coins.append(score_coin)
        
        # Dessiner le rectangle d'analyse
        couleur = (0, 255, 0) if score_coin > 5 else (0, 165, 255) if score_coin > 3 else (0, 0, 255)
        cv2.rectangle(img_annotee, (x1, y1), (x2, y2), couleur, 2)
        cv2.putText(img_annotee, f"{score_coin:.1f}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)
    
    # Score total des coins
    score_total = sum(scores_coins)
    
    return min(30, score_total)  # Maximum 30 points

def analyser_bords(image, gris, img_annotee, debug=False):
    """Analyse la qualité des bords de la carte (30 points)"""
    h, w = image.shape[:2]
    
    # Définir les 4 bords (haut, droite, bas, gauche) - AJUSTÉS POUR ÊTRE PLUS PRÈS DES BORDURES
    bords = [
        (int(w*0.01), int(h*0.006), int(w*0.99), int(h*0.027)),  # Haut - plus près du bord
        (int(w*0.96), int(h*0.01), int(w*0.998), int(h*0.99)),  # Droite - plus près du bord
        (int(w*0.01), int(h*0.965), int(w*0.99), int(h*0.999)),  # Bas - plus près du bord
        (int(w*0.01), int(h*0.01), int(w*0.035), int(h*0.99))   # Gauche - plus près du bord
    ]
    
    scores_bords = []
    
    for i, (x1, y1, x2, y2) in enumerate(bords):
        # Extraire la ROI pour le bord
        roi = gris[y1:y2, x1:x2]
        
        # Détecter les lignes droites avec Hough
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        # Analyser la rectitude des bords
        score_rectitude = 3.75  # Score de base pour la rectitude
        if lines is not None:
            if len(lines) > 10:  # Trop de lignes détectées, bord abîmé
                score_rectitude = max(1, score_rectitude - len(lines) * 0.05)
        
        # Analyser la texture et l'usure
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        diff = cv2.absdiff(roi, blurred)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        usure = np.sum(thresh) / (roi.shape[0] * roi.shape[1] * 255)
        
        score_usure = 3.75 * (1 - min(1, usure * 10))
        
        # Combiner pour un score total sur 7.5 (30/4 bords)
        score_bord = min(7.5, score_rectitude + score_usure)
        scores_bords.append(score_bord)
        
        # Dessiner le rectangle d'analyse
        couleur = (0, 255, 0) if score_bord > 5 else (0, 165, 255) if score_bord > 3 else (0, 0, 255)
        cv2.rectangle(img_annotee, (x1, y1), (x2, y2), couleur, 2)
        cv2.putText(img_annotee, f"{score_bord:.1f}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)
    
    # Score total des bords
    score_total = sum(scores_bords)
    
    return min(30, score_total)  # Maximum 30 points


def analyser_centrage(image, gris, img_annotee, debug=False):
    """Analyse le centrage de l'image sur la carte (20 points)"""
    h, w = image.shape[:2]
    
    # Détecter les bords de l'illustration
    # Utiliser le seuillage adaptatif pour trouver les contours
    thresh = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Appliquer des opérations morphologiques pour renforcer les contours
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Chercher le contour de l'illustration (généralement le plus grand à l'intérieur)
    illustration_contour = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignorer les contours trop petits ou trop grands
        if area > 0.1 * w * h and area < 0.9 * w * h and area > max_area:
            max_area = area
            illustration_contour = cnt
    
    if illustration_contour is None:
        # Si l'illustration ne peut pas être détectée, utiliser une estimation
        print("Impossible de détecter l'illustration, utilisation d'une estimation")
        x, y, wc, hc = int(w*0.065), int(h*0.09), int(w*0.865), int(h*0.375)
        cv2.rectangle(img_annotee, (x, y), (x+wc, y+hc), (0, 0, 255), 2)
        
        # Score moyen par défaut
        return 14
    
    # Obtenir le rectangle englobant
    x, y, wc, hc = cv2.boundingRect(illustration_contour)
    
 # Suite de la fonction analyser_centrage
    # Dessiner le contour détecté
    cv2.rectangle(img_annotee, (x, y), (x+wc, y+hc), (255, 0, 0), 2)
    
    # Calculer les marges de chaque côté
    marge_gauche = x
    marge_droite = w - (x + wc)
    marge_haut = y
    marge_bas = h - (y + hc)
    
    # Calculer les différences de centrage horizontal et vertical
    diff_h = abs(marge_gauche - marge_droite) / w
    diff_v = abs(marge_haut - marge_bas) / h
    
    # Calculer le score basé sur les différences (moins de différence = meilleur score)
    score_h = 10 * (1 - min(1, diff_h * 5))
    score_v = 10 * (1 - min(1, diff_v * 5))
    
    score_total = score_h + score_v
    
    # Ajouter les informations de centrage à l'image
    cv2.putText(img_annotee, f"H: {diff_h*100:.1f}%, V: {diff_v*100:.1f}%", 
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return min(20, score_total)  # Maximum 20 points

def convertir_note_psa(note_sur_100):
    """Convertit une note sur 100 en note PSA sur 10"""
    if note_sur_100 >= 98:
        return 10  # Gem Mint
    elif note_sur_100 >= 93:
        return 9  # Mint
    elif note_sur_100 >= 85:
        return 8  # Near Mint-Mint
    elif note_sur_100 >= 75:
        return 7  # Near Mint
    elif note_sur_100 >= 65:
        return 6  # Excellent-Near Mint
    elif note_sur_100 >= 55:
        return 5  # Excellent
    elif note_sur_100 >= 45:
        return 4  # Very Good-Excellent
    elif note_sur_100 >= 35:
        return 3  # Very Good
    elif note_sur_100 >= 25:
        return 2  # Good
    else:
        return 1  # Poor

def obtenir_description_psa(note_psa):
    """Retourne la description correspondant à une note PSA"""
    descriptions = {
        10: "Gem Mint",
        9: "Mint",
        8: "Near Mint-Mint",
        7: "Near Mint",
        6: "Excellent-Near Mint",
        5: "Excellent",
        4: "Very Good-Excellent",
        3: "Very Good",
        2: "Good",
        1: "Poor"
    }
    return descriptions.get(note_psa, "")

def generer_rapport_qualite(note_globale, note_psa, resultats, image_annotee, nom_fichier="rapport_qualite.jpg"):
    """Génère un rapport détaillé sur la qualité de la carte"""
    h, w = image_annotee.shape[:2]
    
    # Créer une image pour le rapport (image originale + zone de texte)
    rapport = np.ones((h + 250, w, 3), dtype=np.uint8) * 255
    
    # Copier l'image annotée dans le rapport
    rapport[:h, :w] = image_annotee
    
    # Ajouter un titre
    cv2.putText(rapport, f"RAPPORT D'ANALYSE PSA", 
                (20, h + 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
    
    # Ajouter les détails
    cv2.putText(rapport, f"Note globale: {note_globale:.1f}/100", 
                (20, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(rapport, f"Équivalent PSA: {note_psa} ({obtenir_description_psa(note_psa)})", 
                (20, h + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Détails par catégorie
    y_offset = h + 150
    categories = [
        ("Coins", resultats['coins'], 30),
        ("Bords", resultats['bords'], 30),
        ("Centrage", resultats['centrage'], 20)
    ]
    
    for categorie, score, max_score in categories:
        pourcentage = (score / max_score) * 100
        couleur = (0, 0, 255) if pourcentage < 60 else (0, 165, 255) if pourcentage < 80 else (0, 128, 0)
        
        cv2.putText(rapport, f"{categorie}: {score:.1f}/{max_score} ({pourcentage:.1f}%)", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
        y_offset += 30
    
    # Sauvegarder le rapport
    cv2.imwrite(nom_fichier, cv2.cvtColor(rapport, cv2.COLOR_RGB2BGR))
    print(f"Rapport sauvegardé sous: {nom_fichier}")
    
    return rapport

def sauvegarder_image(image, nom_fichier="carte_scannee.jpg"):
    """Sauvegarde l'image au format RGB en BGR pour OpenCV"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convertir RGB en BGR pour la sauvegarde avec OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(nom_fichier, image_bgr)
    else:
        cv2.imwrite(nom_fichier, image)
    print(f"Image sauvegardée sous: {nom_fichier}")