import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import math
import os
from sklearn.linear_model import RANSACRegressor

def scanner_carte_pokemon_ameliore(chemin_image, afficher_etapes=False, sauvegarder_rapport=True):
    """
    Scanne une carte Pokémon en détectant ses contours, analyse sa qualité
    directement à partir de la bordure verte, puis redresse la carte.
    """
    # 1. Charger l'image
    image_originale = cv2.imread(chemin_image)
    if image_originale is None:
        print(f"Impossible de charger l'image à partir de {chemin_image}")
        return None
    
    # Redimensionner l'image si elle est trop grande pour améliorer la performance
    hauteur, largeur = image_originale.shape[:2]
    max_dimension = 1200
    if max(hauteur, largeur) > max_dimension:
        facteur = max_dimension / float(max(hauteur, largeur))
        image = cv2.resize(image_originale, None, fx=facteur, fy=facteur)
        print(f"Image redimensionnée de {largeur}x{hauteur} à {int(largeur*facteur)}x{int(hauteur*facteur)}")
    else:
        image = image_originale.copy()
    
    # Convertir en RGB pour l'affichage
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if afficher_etapes:
        afficher_image(image_rgb, "Image originale")
    
    # 2. Nouvelle méthode: détection par changement de pixel
    # Convertir en niveaux de gris
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou pour réduire le bruit
    flou = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Seuillage pour obtenir une image binaire
    thresh = cv2.adaptiveThreshold(flou, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 5)
    
    if afficher_etapes:
        afficher_image(thresh, "Seuillage adaptatif")
    
    # Appliquer des opérations morphologiques pour nettoyer l'image
    kernel = np.ones((5, 5), np.uint8)
    fermeture = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    ouverture = cv2.morphologyEx(fermeture, cv2.MORPH_OPEN, kernel, iterations=1)
    
    if afficher_etapes:
        afficher_image(ouverture, "Nettoyage morphologique")
    
    # 3. Trouver les contours
    contours, _ = cv2.findContours(ouverture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours par aire et forme
    contours_filtres = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:  # Ignorer les petits contours
            continue
        
        # Vérifier si le contour est rectangulaire
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # Si le contour a approximativement 4 côtés et une aire suffisante, l'ajouter
        if len(approx) >= 4 and len(approx) <= 8:
            contours_filtres.append(cnt)
    
    if not contours_filtres:
        print("Aucun contour rectangulaire trouvé, essai avec méthode alternative...")
        # Méthode alternative: détection des bords avec Canny
        canny = cv2.Canny(flou, 30, 150)
        dilate = cv2.dilate(canny, kernel, iterations=2)
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_filtres = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
    
    if not contours_filtres:
        print("Aucun contour de carte détecté")
        return None
    
    # Trouver le contour le plus rectangulaire (ratio largeur/hauteur proche de 6.3/8.8)
    meilleur_contour = None
    meilleur_score = float('inf')
    ratio_cible = 6.3 / 8.8  # Ratio largeur/hauteur standard d'une carte Pokémon (63mm x 88mm)
    
    for contour in contours_filtres:
        # Calculer le rectangle englobant avec angle minimal
        rect = cv2.minAreaRect(contour)
        largeur_rect, hauteur_rect = rect[1]
        if largeur_rect < hauteur_rect:  # Assurer que largeur est le côté le plus court
            largeur_rect, hauteur_rect = hauteur_rect, largeur_rect
        
        # Calculer le ratio et la différence avec le ratio cible
        if hauteur_rect > 0:  # Éviter division par zéro
            ratio = largeur_rect / hauteur_rect
            difference_ratio = abs(ratio - ratio_cible)
            
            # Calculer le score (différence de ratio et rectangularité)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            rectangularite = abs(len(approx) - 4)  # 0 si exactement 4 points
            
            # Utiliser un score qui favorise encore plus la rectangularité
            score = difference_ratio + rectangularite * 1.0
            
            if score < meilleur_score:
                meilleur_score = score
                meilleur_contour = contour
    
    if meilleur_contour is None:
        print("Aucun contour valide trouvé")
        return None
    
    # Dessiner le contour trouvé pour vérification
    debug_img = image_rgb.copy()
    cv2.drawContours(debug_img, [meilleur_contour], -1, (0, 255, 0), 3)
    
    if afficher_etapes:
        afficher_image(debug_img, "Contour détecté (vert)")
    
    # ======= PARTIE AJOUTÉE: ANALYSE DE LA QUALITÉ DIRECTEMENT SUR LE CONTOUR VERT =======
    # Analyser le centrage, les coins et les bords avant redressement
    resultats_analyse = analyser_qualite_carte(image, meilleur_contour, debug_img.copy(), afficher_etapes)
    
    if sauvegarder_rapport and resultats_analyse:
        nom_base = os.path.splitext(os.path.basename(chemin_image))[0]
        generer_rapport_qualite(resultats_analyse, chemin_image, f"{nom_base}_rapport_qualite.jpg")
        
        if afficher_etapes:
            afficher_image(resultats_analyse['image_annotee'], "Analyse de qualité")
    # ======= FIN PARTIE AJOUTÉE =======
    
    # Continuer avec le code de détection des lignes et points pour le redressement
    epsilon = 0.01 * cv2.arcLength(meilleur_contour, True)
    approx = cv2.approxPolyDP(meilleur_contour, epsilon, True)
    
    # Si nous avons un contour approximatif avec 4 points, utiliser ces points directement
    if len(approx) == 4:
        points = approx.reshape(4, 2)
    else:
        # (Ici le code de détection alternatif avec Canny, HoughLinesP, RANSAC, etc. reste inchangé)
        # ... [code existant pour extraire "points" via HoughLinesP et RANSAC] ...
        # Pour ne pas allonger l'exemple, le code reste identique jusqu'au tracé du rectangle.
        # On suppose que la variable "points" est correctement définie à ce stade.
        # (Code inchangé)
        # -----------------------
        # Bloc alternatif de détection des coins (déjà présent dans ton code)
        x, y, w, h = cv2.boundingRect(meilleur_contour)
        margin = 20  # marge autour du contour
        roi_x = max(0, x - margin)
        roi_y = max(0, y - margin)
        roi_w = min(image.shape[1] - roi_x, w + 2*margin)
        roi_h = min(image.shape[0] - roi_y, h + 2*margin)
        
        roi = gris[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    angle = 90
                else:
                    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if angle < 45:
                    horizontal_lines.append(line[0])
                else:
                    vertical_lines.append(line[0])
        
        def group_lines(lines, threshold_distance=30, threshold_angle=10):
            if not lines:
                return []
            groups = []
            for line in lines:
                x1, y1, x2, y2 = line
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = math.degrees(math.atan2(y2-y1, x2-x1)) % 180
                added = False
                for group in groups:
                    g_x1, g_y1, g_x2, g_y2 = group["line"]
                    g_length = group["length"]
                    g_angle = group["angle"]
                    if abs(angle - g_angle) < threshold_angle:
                        dist = abs((g_y2-g_y1)*x1 - (g_x2-g_x1)*y1 + g_x2*g_y1 - g_y2*g_x1) / g_length
                        if dist < threshold_distance:
                            if length > g_length:
                                group["line"] = line
                                group["length"] = length
                                group["angle"] = angle
                            added = True
                            break
                if not added:
                    groups.append({"line": line, "length": length, "angle": angle})
            return [group["line"] for group in groups]
        
        grouped_h_lines = group_lines(horizontal_lines)
        grouped_v_lines = group_lines(vertical_lines)
        
        corners = []
        for h_line in grouped_h_lines:
            for v_line in grouped_v_lines:
                x1_h, y1_h, x2_h, y2_h = h_line
                x1_v, y1_v, x2_v, y2_v = v_line
                a1 = y2_h - y1_h
                b1 = x1_h - x2_h
                c1 = x2_h*y1_h - x1_h*y2_h
                a2 = y2_v - y1_v
                b2 = x1_v - x2_v
                c2 = x2_v*y1_v - x1_v*y2_v
                det = a1*b2 - a2*b1
                if det != 0:
                    x_inter = (b1*c2 - b2*c1) / det
                    y_inter = (a2*c1 - a1*c2) / det
                    x_inter += roi_x
                    y_inter += roi_y
                    corners.append([int(x_inter), int(y_inter)])
        
        if len(corners) >= 4:
            def group_points(points, threshold=20):
                groups = []
                for point in points:
                    added = False
                    for group in groups:
                        for g_point in group:
                            dist = np.sqrt((point[0]-g_point[0])**2 + (point[1]-g_point[1])**2)
                            if dist < threshold:
                                group.append(point)
                                added = True
                                break
                        if added:
                            break
                    if not added:
                        groups.append([point])
                centers = []
                for group in groups:
                    x_center = sum(p[0] for p in group) / len(group)
                    y_center = sum(p[1] for p in group) / len(group)
                    centers.append([int(x_center), int(y_center)])
                return centers
            grouped_corners = group_points(corners)
            if len(grouped_corners) == 4:
                points = np.array(grouped_corners, dtype=np.int32)
            else:
                rect = cv2.minAreaRect(np.array(corners))
                points = cv2.boxPoints(rect)
                points = np.array(points, dtype=np.int32)
        else:
            contour_points = meilleur_contour.reshape(-1, 2)
            y_sorted = contour_points[contour_points[:, 1].argsort()]
            top_points = y_sorted[:len(y_sorted)//4]
            bottom_points = y_sorted[-len(y_sorted)//4:]
            x_sorted = contour_points[contour_points[:, 0].argsort()]
            left_points = x_sorted[:len(x_sorted)//4]
            right_points = x_sorted[-len(x_sorted)//4:]
            
            def fit_line_ransac(points, weights=None):
                if len(points) < 2:
                    return None, None
                X = points[:, 0].reshape(-1, 1)
                y_vals = points[:, 1]
                ransac = RANSACRegressor(random_state=42, max_trials=1000, residual_threshold=1.5)
                try:
                    if weights is not None:
                        ransac.fit(X, y_vals, sample_weight=weights)
                    else:
                        ransac.fit(X, y_vals)
                    slope = ransac.estimator_.coef_[0]
                    intercept = ransac.estimator_.intercept_
                    return slope, intercept
                except:
                    return None, None
            
            top_slope, top_intercept = fit_line_ransac(top_points)
            bottom_slope, bottom_intercept = fit_line_ransac(bottom_points)
            left_slope, left_intercept = fit_line_ransac(left_points[:, [1, 0]])
            right_slope, right_intercept = fit_line_ransac(right_points[:, [1, 0]])
            
            weights_top = np.ones(len(top_points))
            for i, point in enumerate(top_points):
                dist = cv2.pointPolygonTest(meilleur_contour, (float(point[0]), float(point[1])), True)
                weights_top[i] = 1.0 / (abs(dist) + 1.0)
            top_slope, top_intercept = fit_line_ransac(top_points, weights=weights_top)
            
            def get_points_on_horizontal_line(slope, intercept, x_min, x_max):
                x1 = x_min
                y1 = slope * x_min + intercept
                x2 = x_max
                y2 = slope * x_max + intercept
                return (int(x1), int(y1)), (int(x2), int(y2))
            
            def get_points_on_vertical_line(slope, intercept, y_min, y_max):
                if slope == 0:
                    slope = 0.0001
                y1 = y_min
                x1 = (y_min - intercept) / slope
                y2 = y_max
                x2 = (y_max - intercept) / slope
                return (int(x1), int(y1)), (int(x2), int(y2))
            
            corners = []
            if all(param is not None for param in [top_slope, top_intercept, bottom_slope, bottom_intercept, 
                                                     left_slope, left_intercept, right_slope, right_intercept]):
                x_min, y_min, w, h = cv2.boundingRect(meilleur_contour)
                x_max = x_min + w
                y_max = y_min + h
                top_p1, top_p2 = get_points_on_horizontal_line(top_slope, top_intercept, x_min, x_max)
                bottom_p1, bottom_p2 = get_points_on_horizontal_line(bottom_slope, bottom_intercept, x_min, x_max)
                left_p1, left_p2 = get_points_on_vertical_line(left_slope, left_intercept, y_min, y_max)
                right_p1, right_p2 = get_points_on_vertical_line(right_slope, right_intercept, y_min, y_max)
                
                def line_intersection(line1, line2):
                    x1, y1, x2, y2 = line1
                    x3, y3, x4, y4 = line2
                    a1 = y2 - y1
                    b1 = x1 - x2
                    c1 = x2*y1 - x1*y2
                    a2 = y4 - y3
                    b2 = x3 - x4
                    c2 = x4*y3 - x3*y4
                    det = a1*b2 - a2*b1
                    if det == 0:
                        return None
                    x_int = (b1*c2 - b2*c1) / det
                    y_int = (a2*c1 - a1*c2) / det
                    return (int(x_int), int(y_int))
                
                top_left = line_intersection((top_p1[0], top_p1[1], top_p2[0], top_p2[1]), 
                                             (left_p1[0], left_p1[1], left_p2[0], left_p2[1]))
                top_right = line_intersection((top_p1[0], top_p1[1], top_p2[0], top_p2[1]), 
                                              (right_p1[0], right_p1[1], right_p2[0], right_p2[1]))
                bottom_left = line_intersection((bottom_p1[0], bottom_p1[1], bottom_p2[0], bottom_p2[1]), 
                                                (left_p1[0], left_p1[1], left_p2[0], left_p2[1]))
                bottom_right = line_intersection((bottom_p1[0], bottom_p1[1], bottom_p2[0], bottom_p2[1]), 
                                                 (right_p1[0], right_p1[1], right_p2[0], right_p2[1]))
                
                if all(corner is not None for corner in [top_left, top_right, bottom_left, bottom_right]):
                    corners = [top_left, top_right, bottom_right, bottom_left]
            if corners:
                points = np.array(corners, dtype=np.int32)
            else:
                rect = cv2.minAreaRect(meilleur_contour)
                points = cv2.boxPoints(rect)
                points = np.array(points, dtype=np.int32)
        
        # Dessiner le rectangle ajusté ET les lignes d'analyse suivant la bordure verte
        debug_img_aligned = image_rgb.copy()
        # Affichage des points détectés
        for point in points:
            cv2.circle(debug_img_aligned, (point[0], point[1]), 5, (0, 0, 255), -1)
        
        # --- Nouvelle méthode d'affichage ---
        # On passe en HSV et on isole la couleur verte pour obtenir le contour précis de la bordure
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_green:
            green_contour = max(contours_green, key=cv2.contourArea)
            epsilon_green = 0.005 * cv2.arcLength(green_contour, True)
            green_contour_approx = cv2.approxPolyDP(green_contour, epsilon_green, True)
            cv2.drawContours(debug_img_aligned, [green_contour_approx], -1, (255, 0, 0), 3)
        # --- Fin de la nouvelle méthode ---
        
        if afficher_etapes:
            afficher_image(debug_img_aligned, "Rectangle de recadrage amélioré (lignes d'analyse)")
            
            if 'grouped_h_lines' in locals() and 'grouped_v_lines' in locals():
                lines_img = image_rgb.copy()
                for line in grouped_h_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(lines_img, (x1 + roi_x, y1 + roi_y), (x2 + roi_x, y2 + roi_y), (0, 255, 0), 2)
                for line in grouped_v_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(lines_img, (x1 + roi_x, y1 + roi_y), (x2 + roi_x, y2 + roi_y), (0, 0, 255), 2)
                afficher_image(lines_img, "Lignes détectées (horizontales: vert, verticales: rouge)")
    
    # Ordonner les points (haut-gauche, haut-droite, bas-droite, bas-gauche)
    points_ordonnes = ordonner_points(points)
    
    # Calculer les dimensions du rectangle final
    largeur_max = max(
        np.linalg.norm(points_ordonnes[0] - points_ordonnes[1]),
        np.linalg.norm(points_ordonnes[2] - points_ordonnes[3])
    )
    
    hauteur_max = max(
        np.linalg.norm(points_ordonnes[0] - points_ordonnes[3]),
        np.linalg.norm(points_ordonnes[1] - points_ordonnes[2])
    )
    
    # Calculer la marge (environ 2mm)
    pixels_par_mm = min(largeur_max / 63.0, hauteur_max / 88.0)
    marge_pixels = int(2 * pixels_par_mm)
    
    nouvelle_largeur = int(largeur_max + 2 * marge_pixels)
    nouvelle_hauteur = int(hauteur_max + 2 * marge_pixels)
    
    pts_dst = np.array([
        [marge_pixels, marge_pixels],
        [marge_pixels + largeur_max - 1, marge_pixels],
        [marge_pixels + largeur_max - 1, marge_pixels + hauteur_max - 1],
        [marge_pixels, marge_pixels + hauteur_max - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(points_ordonnes.astype(np.float32), pts_dst)
    carte_redressee = cv2.warpPerspective(image, M, (nouvelle_largeur, nouvelle_hauteur))
    
    carte_redressee_rgb = cv2.cvtColor(carte_redressee, cv2.COLOR_BGR2RGB)
    
    if afficher_etapes:
        afficher_image(carte_redressee_rgb, "Carte redressée")
    
    if 'ajouter_bordure_arrondie' in globals():
        try:
            resultat_final = ajouter_bordure_arrondie(carte_redressee_rgb, rayon_coin=15, epaisseur_bordure=8)
        except:
            resultat_final = carte_redressee_rgb
    else:
        resultat_final = carte_redressee_rgb
    
    return resultat_final, resultats_analyse


# ===== NOUVELLES FONCTIONS D'ANALYSE DE QUALITÉ BASÉES SUR LA BORDURE VERTE =====

def analyser_qualite_carte(image, contour, img_annotee, debug=False):
    """
    Analyse la qualité de la carte en utilisant directement le contour vert détecté.
    Applique d'abord une rotation pour redresser la carte avant l'analyse.
    """
    # Améliorer la détection du contour vert avec un traitement supplémentaire
    # 1. Créer un masque HSV plus précis pour le contour vert
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Opérations morphologiques pour nettoyer le masque
    kernel = np.ones((3, 3), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Trouver les contours dans le masque
    contours_verts, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si on a trouvé des contours verts, prendre le plus grand
    if contours_verts and len(contours_verts) > 0:
        contour_vert = max(contours_verts, key=cv2.contourArea)
        # Vérifier si le nouveau contour est suffisamment grand
        if cv2.contourArea(contour_vert) > cv2.contourArea(contour) * 0.7:
            contour = contour_vert  # Utiliser le nouveau contour vert si suffisamment grand
    
    # 2. Calculer l'angle d'inclinaison de la carte avec plus de précision
    # Utiliser une meilleure approximation du contour
    epsilon = 0.001 * cv2.arcLength(contour, True)  # Précision accrue
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Si on a un contour approximatif avec 4 points ou plus, utiliser les coins pour calculer l'angle
    if len(approx) >= 4:
        # Calculer le rectangle minimal
        rect = cv2.minAreaRect(approx)
    else:
        rect = cv2.minAreaRect(contour)
    
    angle = rect[2]
    
    # Ajuster l'angle (OpenCV donne parfois l'angle par rapport à la verticale)
    if angle < -45:
        angle = angle + 90
    
    # 3. Créer une matrice de rotation et rotation du contour
    # Centre de rotation (centre du rectangle englobant)
    center = rect[0]
    h, w = image.shape[:2]
    
    # Matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Appliquer la rotation à l'image
    image_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    img_annotee_rotated = cv2.warpAffine(img_annotee, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    # Rotation du contour
    contour_points = contour.reshape(-1, 2)
    ones = np.ones(shape=(len(contour_points), 1))
    points_ones = np.hstack([contour_points, ones])
    
    # Appliquer la rotation aux points du contour
    transformed_points = M.dot(points_ones.T).T
    rotated_contour = transformed_points.reshape(-1, 1, 2).astype(np.int32)
    
    # 4. Détecter un rectangle plus précis à partir du contour pivoté
    # Calculer à nouveau un rectangle minimal plus précis
    rotated_rect = cv2.minAreaRect(rotated_contour)
    rotated_box = cv2.boxPoints(rotated_rect)
    rotated_box = np.array(rotated_box, dtype=np.int32)
    
    # 5. Affiner le rectangle pour qu'il soit parfaitement horizontal/vertical
    x_min, y_min = np.min(rotated_box, axis=0)
    x_max, y_max = np.max(rotated_box, axis=0)
    
    # Créer un rectangle parfaitement aligné avec les axes horizontal et vertical
    aligned_box = np.array([
        [x_min, y_min],  # haut-gauche
        [x_max, y_min],  # haut-droit
        [x_max, y_max],  # bas-droit
        [x_min, y_max]   # bas-gauche
    ], dtype=np.int32)
    
    # Utiliser ce rectangle aligné comme contour final
    aligned_contour = aligned_box.reshape(-1, 1, 2)
    
    # AJOUT: Dessiner directement le contour en rouge pour qu'il suive le bord vert
    cv2.drawContours(img_annotee_rotated, [aligned_contour], -1, (0, 0, 255), 2)
    
    if debug:
        debug_rotated = image_rotated.copy()
        cv2.drawContours(debug_rotated, [aligned_contour], -1, (0, 255, 0), 2)
        afficher_image(debug_rotated, "Contour après rotation et alignement")
    
    # 6. Analyser la carte redressée avec le contour aligné
    # Analyser le centrage
    resultats_centrage = analyser_centrage_direct(image_rotated.shape, aligned_contour, img_annotee_rotated, debug)
    
    # Analyser les coins
    resultats_coins = analyser_coins_direct(image_rotated, aligned_contour, img_annotee_rotated, debug)
    
    # Analyser les bords
    resultats_bords = analyser_bords_direct(image_rotated, aligned_contour, img_annotee_rotated, debug)
    
    # Calculer le score global
    score_total = resultats_centrage["score"] + resultats_coins["score"] + resultats_bords["score"]
    note_psa = convertir_note_psa(score_total)
    
    # Ajouter la note globale à l'image annotée
    cv2.putText(img_annotee_rotated, f"Note: {score_total:.1f}/80 (PSA {note_psa})", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Combiner les résultats
    resultats = {
        "centrage": resultats_centrage,
        "coins": resultats_coins,
        "bords": resultats_bords,
        "score_total": score_total,
        "note_psa": note_psa,
        "image_annotee": img_annotee_rotated
    }
    
    return resultats

def analyser_centrage_direct(image_shape, contour, img_annotee, debug=False):
    """
    Analyse le centrage de la carte par rapport à l'image,
    en utilisant directement le contour vert détecté.
    """
    h_img, w_img = image_shape[:2]
    
    # Calculer le rectangle minimal avec rotation
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)  # Utiliser np.int32 au lieu de np.intp
    
    # Ordonner les points
    points = ordonner_points(box)
    
    # Calculer les coordonnées des bords (assurez-vous qu'ils sont des entiers)
    min_x = int(min(p[0] for p in points))
    max_x = int(max(p[0] for p in points))
    min_y = int(min(p[1] for p in points))
    max_y = int(max(p[1] for p in points))
    
    # Calculer les marges
    marge_gauche = min_x
    marge_droite = w_img - max_x
    marge_haut = min_y
    marge_bas = h_img - max_y
    
    # Calculer les différences de centrage
    diff_h = abs(marge_gauche - marge_droite) / w_img
    diff_v = abs(marge_haut - marge_bas) / h_img
    
    # Calculer le score (moins de différence = meilleur score)
    score_h = 10 * (1 - min(1, diff_h * 5))
    score_v = 10 * (1 - min(1, diff_v * 5))
    score_total = score_h + score_v
    
    # Vérifier que les valeurs sont dans les limites de l'image
    min_y = max(0, min(h_img-1, min_y))
    max_y = max(0, min(h_img-1, max_y))
    min_x = max(0, min(w_img-1, min_x))
    max_x = max(0, min(w_img-1, max_x))
    
    # MODIFICATION: Ne pas dessiner les lignes droites horizontales et verticales
    # Les lignes suivantes sont commentées car nous dessinons maintenant
    # le contour directement dans la fonction analyser_qualite_carte
    
    # Haut
    # cv2.line(img_annotee, (0, min_y), (w_img-1, min_y), (255, 0, 0), 2)
    # Bas
    # cv2.line(img_annotee, (0, max_y), (w_img-1, max_y), (255, 0, 0), 2)
    # Gauche
    # cv2.line(img_annotee, (min_x, 0), (min_x, h_img-1), (255, 0, 0), 2)
    # Droite
    # cv2.line(img_annotee, (max_x, 0), (max_x, h_img-1), (255, 0, 0), 2)
    
    # Ajouter les informations de centrage à l'image
    cv2.putText(img_annotee, f"Centrage H: {diff_h*100:.1f}%, V: {diff_v*100:.1f}%", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_annotee, f"Score centrage: {score_total:.1f}/20", 
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if debug:
        print(f"Différence centrage horizontal: {diff_h*100:.1f}%")
        print(f"Différence centrage vertical: {diff_v*100:.1f}%")
        print(f"Score centrage: {score_total:.1f}/20")
    
    return {
        "score": min(20, score_total),
        "diff_h": diff_h,
        "diff_v": diff_v,
        "marges": {
            "gauche": marge_gauche,
            "droite": marge_droite,
            "haut": marge_haut,
            "bas": marge_bas
        }
    }

def analyser_coins_direct(image, contour, img_annotee, debug=False):
    """
    Analyse la qualité des coins de la carte en utilisant le contour vert.
    """
    # Extraction approximative des 4 coins du contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Calculer le rectangle minimal
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    points = ordonner_points(box)
    points = np.array(points, dtype=np.int32)  # Utiliser np.int32 au lieu de np.intp
    
    h, w = image.shape[:2]
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    scores_coins = []
    taille_roi = int(min(w, h) * 0.05)  # ROI pour les coins
    
    for i, (cx, cy) in enumerate(points):
        # Extraire la région d'intérêt (ROI) pour le coin
        x1 = max(0, cx - taille_roi//2)
        y1 = max(0, cy - taille_roi//2)
        x2 = min(w, cx + taille_roi//2)
        y2 = min(h, cy + taille_roi//2)
        
        # Identifier spécifiquement le coin en haut à gauche
        is_top_left = False
        point_sum = cx + cy
        min_sum = min(p[0] + p[1] for p in points)
        if point_sum <= min_sum + 5:  # Tolérance de 5 pixels
            is_top_left = True
        
        # Ajustement spécifique pour le coin en haut à gauche
        if is_top_left:
            # Valeurs optimisées pour déplacer le carré correctement
            offset_x = 45  # pixels vers la droite
            offset_y = 30  # pixels vers le bas
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
        
        roi = gris[y1:y2, x1:x2]
        
        # Vérifier si ROI est valide
        if roi.size == 0:
            scores_coins.append(0)
            continue
        
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
    
    cv2.putText(img_annotee, f"Score coins: {score_total:.1f}/30", 
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if debug:
        print(f"Scores des coins: {scores_coins}")
        print(f"Score total des coins: {score_total:.1f}/30")
        
    return {
        "score": min(30, score_total),
        "scores_individuels": scores_coins
    }

def analyser_bords_direct(image, contour, img_annotee, debug=False):
    """
    Analyse la qualité des bords de la carte en utilisant le contour vert.
    """
    h, w = image.shape[:2]
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Obtenir les points ordonnés du contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    points = ordonner_points(box)
    points = np.intp(points)
    
    # Définir les 4 bords à partir des points ordonnés
    bords = [
        (points[0], points[1]),  # Haut
        (points[1], points[2]),  # Droite
        (points[2], points[3]),  # Bas
        (points[3], points[0])   # Gauche
    ]
    
    scores_bords = []
    epaisseur_analyse = int(min(w, h) * 0.03)  # Épaisseur de la zone d'analyse
    
    for i, (p1, p2) in enumerate(bords):
        # Calculer la direction et la longueur du bord
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        longueur = np.sqrt(dx**2 + dy**2)
        
        # Calculer la normale au bord (perpendiculaire)
        nx, ny = -dy/longueur, dx/longueur
        
        # Échantillonner des points le long du bord
        nb_points = 20  # Nombre de points à échantillonner
        scores_echantillons = []
        
        for j in range(nb_points):
            # Point sur le bord
            t = j / (nb_points - 1)
            px = int(p1[0] + t * dx)
            py = int(p1[1] + t * dy)
            
            # Points intérieurs et extérieurs perpendiculaires au bord
            px_int = int(px + nx * epaisseur_analyse)
            py_int = int(py + ny * epaisseur_analyse)
            px_ext = int(px - nx * epaisseur_analyse)
            py_ext = int(py - ny * epaisseur_analyse)
            
            # S'assurer que les points sont dans l'image
            px_int = max(0, min(w-1, px_int))
            py_int = max(0, min(h-1, py_int))
            px_ext = max(0, min(w-1, px_ext))
            py_ext = max(0, min(h-1, py_ext))
            
            # Dessiner la ligne d'analyse
            cv2.line(img_annotee, (px_ext, py_ext), (px_int, py_int), (0, 255, 255), 1)
            
            # Extraire une petite région autour du point
            x1 = max(0, px - epaisseur_analyse//2)
            y1 = max(0, py - epaisseur_analyse//2)
            x2 = min(w, px + epaisseur_analyse//2)
            y2 = min(h, py + epaisseur_analyse//2)
            
            if x2 > x1 and y2 > y1:  # Vérifier que la région est valide
                roi = gris[y1:y2, x1:x2]
                
                # Analyser la rectitude et l'usure
                edges = cv2.Canny(roi, 50, 150)
                blurred = cv2.GaussianBlur(roi, (5, 5), 0)
                diff = cv2.absdiff(roi, blurred)
                
                score_echantillon = 1.0 - (np.mean(edges) / 255)  # Moins de bords = meilleur état
                scores_echantillons.append(score_echantillon)
        
        # Score moyen pour ce bord (sur 7.5)
        if scores_echantillons:
            score_bord = min(7.5, np.mean(scores_echantillons) * 10)
        else:
            score_bord = 0
        
        scores_bords.append(score_bord)
        
        # Point milieu du bord pour afficher le score
        mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        couleur = (0, 255, 0) if score_bord > 5 else (0, 165, 255) if score_bord > 3 else (0, 0, 255)
        cv2.putText(img_annotee, f"{score_bord:.1f}", (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)
    
    # Score total des bords
    score_total = sum(scores_bords)
    
    cv2.putText(img_annotee, f"Score bords: {score_total:.1f}/30", 
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    if debug:
        print(f"Scores des bords: {scores_bords}")
        print(f"Score total des bords: {score_total:.1f}/30")
    
    return {
        "score": min(30, score_total),
        "scores_individuels": scores_bords
    }

def generer_rapport_qualite(resultats, chemin_image, nom_fichier="rapport_qualite.jpg"):
    """
    Génère un rapport détaillé sur la qualité de la carte
    """
    # Extraire les scores et l'image annotée
    score_centrage = resultats["centrage"]["score"]
    score_coins = resultats["coins"]["score"]
    score_bords = resultats["bords"]["score"]
    score_global = resultats["score_total"]
    note_psa = resultats["note_psa"]
    img_annotee = resultats["image_annotee"]
    
    # Créer une image pour le rapport (image originale + zone de texte)
    h, w = img_annotee.shape[:2]
    rapport = np.ones((h + 250, w, 3), dtype=np.uint8) * 255
    
    # Copier l'image annotée dans le rapport
    rapport[:h, :w] = img_annotee
    
    # Ajouter un titre
    cv2.putText(rapport, f"RAPPORT D'ANALYSE PSA", 
                (20, h + 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
    
    # Ajouter les détails
    cv2.putText(rapport, f"Note globale: {score_global:.1f}/80", 
                (20, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(rapport, f"Équivalent PSA: {note_psa} ({obtenir_description_psa(note_psa)})", 
                (20, h + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Détails par catégorie
    y_offset = h + 150
    categories = [
        ("Coins", score_coins, 30),
        ("Bords", score_bords, 30),
        ("Centrage", score_centrage, 20)
    ]
    
    for categorie, score, max_score in categories:
        pourcentage = (score / max_score) * 100
        couleur = (0, 0, 255) if pourcentage < 60 else (0, 165, 255) if pourcentage < 80 else (0, 128, 0)
        
        cv2.putText(rapport, f"{categorie}: {score:.1f}/{max_score} ({pourcentage:.1f}%)", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
        y_offset += 30
    
    # Ajouter le nom de l'image
    nom_image = os.path.basename(chemin_image)
    cv2.putText(rapport, f"Image: {nom_image}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Sauvegarder le rapport
    cv2.imwrite(nom_fichier, cv2.cvtColor(rapport, cv2.COLOR_RGB2BGR))
    print(f"Rapport sauvegardé sous: {nom_fichier}")
    
    return rapport

def ordonner_points(pts):
    """
    Ordonne les points d'un rectangle dans l'ordre:
    [haut-gauche, haut-droite, bas-droite, bas-gauche]
    """
    # Convertir en tableau numpy s'il ne l'est pas déjà
    pts = np.array(pts).reshape(4, 2)
    
    # Initialiser le tableau ordonné
    rect = np.zeros((4, 2), dtype="float32")
    
    # Le point avec la plus petite somme est en haut à gauche
    # Le point avec la plus grande somme est en bas à droite
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Calculer la différence entre les points
    # Le point avec la plus petite différence est en haut à droite
    # Le point avec la plus grande différence est en bas à gauche
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def convertir_note_psa(note_sur_80):
    """Convertit une note sur 80 en note PSA sur 10"""
    if note_sur_80 >= 72:
        return 10  # Gem Mint
    elif note_sur_80 >= 64:
        return 9  # Mint
    elif note_sur_80 >= 56:
        return 8  # Near Mint-Mint
    elif note_sur_80 >= 48:
        return 7  # Near Mint
    elif note_sur_80 >= 40:
        return 6  # Excellent-Near Mint
    elif note_sur_80 >= 32:
        return 5  # Excellent
    elif note_sur_80 >= 24:
        return 4  # Very Good-Excellent
    elif note_sur_80 >= 16:
        return 3  # Very Good
    elif note_sur_80 >= 8:
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

def afficher_image(image, titre="Image"):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(titre)
    plt.axis('off')
    plt.show()

# Utilisation de la fonction
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Utiliser le chemin fourni en argument
        chemin_image = sys.argv[1]
    else:
        # Utiliser un chemin par défaut
        chemin_image = "back/IMG_6407.JPG"
        print(f"Aucun chemin d'image spécifié, utilisation de l'image par défaut: {chemin_image}")
    
    # Vérifier si le fichier existe
    if not os.path.exists(chemin_image):
        print(f"ERREUR: Le fichier {chemin_image} n'existe pas.")
        sys.exit(1)
        
    # Scanner la carte avec analyse
    carte_redressee, resultats_analyse = scanner_carte_pokemon_ameliore(chemin_image, afficher_etapes=True)
    
    if carte_redressee is not None:
        # Sauvegarder l'image redressée
        cv2.imwrite("carte_redressee.jpg", cv2.cvtColor(carte_redressee, cv2.COLOR_RGB2BGR))
        print("Image redressée sauvegardée sous: carte_redressee.jpg")
        
        # Afficher un résumé des résultats d'analyse
        if resultats_analyse:
            score_total = resultats_analyse["score_total"]
            note_psa = resultats_analyse["note_psa"]
            
            print(f"\n=== RÉSULTATS DE L'ANALYSE ===")
            print(f"Note globale: {score_total:.1f}/80")
            print(f"Équivalent PSA: {note_psa} ({obtenir_description_psa(note_psa)})")
            print(f"Détails:")
            print(f"- Coins: {resultats_analyse['coins']['score']:.1f}/30")
            print(f"- Bords: {resultats_analyse['bords']['score']:.1f}/30")
            print(f"- Centrage: {resultats_analyse['centrage']['score']:.1f}/20")