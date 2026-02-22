import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import torch

try:
    from model_pokemon import load_model, analyze_card
    ML_DISPONIBLE = True
except ImportError:
    print("Module model_pokemon non trouvé. L'analyse par apprentissage automatique ne sera pas disponible.")
    ML_DISPONIBLE = False

# Variables globales pour le modèle d'apprentissage
MODEL_PATH = 'modele_carte_pokemon.pth'
model = None
device = None

# Fonction pour initialiser le modèle
def init_model():
    global model, device, ML_DISPONIBLE
    
    if not ML_DISPONIBLE:
        return False
        
    try:
        model, device = load_model(MODEL_PATH)
        print("Modèle d'IA chargé avec succès")
        return True
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        ML_DISPONIBLE = False
        return False

# Tenter de charger le modèle au démarrage
try:
    if ML_DISPONIBLE:
        ML_DISPONIBLE = init_model()
except Exception as e:
    print(f"Erreur d'initialisation du modèle: {e}")
    ML_DISPONIBLE = False

def scanner_carte_pokemon(chemin_image, afficher_etapes=False):
    """
    Scanne une carte Pokémon en détectant ses contours par changement de couleur de pixels.
    Particulièrement efficace sur fond blanc.
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
    
    # Seuillage pour obtenir une image binaire (fond blanc = 255, reste = 0)
    # Utiliser un seuillage adaptatif pour s'adapter aux variations d'éclairage
    thresh = cv2.adaptiveThreshold(flou, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 5)
    
    if afficher_etapes:
        afficher_image(thresh, "Seuillage adaptatif")
    
    # Appliquer des opérations morphologiques pour nettoyer l'image
    # Fermeture (dilatation puis érosion) pour fermer les petits trous
    kernel = np.ones((5, 5), np.uint8)
    fermeture = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Ouverture (érosion puis dilatation) pour éliminer les petits bruits
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
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            rectangularite = abs(len(approx) - 4)  # 0 si exactement 4 points
            
            score = difference_ratio + rectangularite * 0.5
            
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
        afficher_image(debug_img, "Contour détecté")
    
    # 4. Extraire la région d'intérêt (ROI) avec une simple délimitation rectangulaire
    x, y, w, h = cv2.boundingRect(meilleur_contour)
    
    # Visualiser le rectangle englobant
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    if afficher_etapes:
        afficher_image(debug_img, "Rectangle englobant")
    
    # Extraire directement la carte sans transformation de perspective
    carte_extraite = image[y:y+h, x:x+w]
    carte_extraite_rgb = cv2.cvtColor(carte_extraite, cv2.COLOR_BGR2RGB)
    
    if afficher_etapes:
        afficher_image(carte_extraite_rgb, "Carte extraite")
    
    # Utiliser l'extraction simple
    resultat = carte_extraite_rgb
    
    # 5. Ajouter les bords arrondis et la bordure grise si la fonction existe
    if 'ajouter_bordure_arrondie' in globals():
        resultat_final = ajouter_bordure_arrondie(resultat, rayon_coin=15, epaisseur_bordure=8)
    else:
        resultat_final = resultat
    
    return resultat_final

def ordonner_points(pts):
    """
    Ordonne les points d'un rectangle dans l'ordre:
    [haut-gauche, haut-droite, bas-droite, bas-gauche]
    """
    # Convertir en tableau numpy s'il ne l'est pas déjà
    pts = pts.reshape(4, 2)
    
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

def afficher_image(image, titre="Image"):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(titre)
    plt.axis('off')
    plt.show()

def sauvegarder_image(image, nom_fichier="carte_scannee.jpg"):
    """Sauvegarde l'image au format RGB en BGR pour OpenCV"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convertir RGB en BGR pour la sauvegarde avec OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(nom_fichier, image_bgr)
    else:
        cv2.imwrite(nom_fichier, image)
    print(f"Image sauvegardée sous: {nom_fichier}")

def traiter_dossier(dossier_entree, dossier_sortie=None, afficher=False):
    """Traite toutes les images d'un dossier"""
    if dossier_sortie is None:
        dossier_sortie = os.path.join(dossier_entree, "resultats")
    
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    nb_succes = 0
    nb_echec = 0
    
    for fichier in os.listdir(dossier_entree):
        if any(fichier.lower().endswith(ext) for ext in extensions):
            chemin_complet = os.path.join(dossier_entree, fichier)
            print(f"Traitement de {fichier}...")
            
            resultat = scanner_carte_pokemon(chemin_complet, afficher_etapes=afficher)
            
            if resultat is not None:
                nom_sortie = os.path.splitext(fichier)[0] + "_scanne.jpg"
                chemin_sortie = os.path.join(dossier_sortie, nom_sortie)
                sauvegarder_image(resultat, chemin_sortie)
                nb_succes += 1
            else:
                print(f"⚠️ Échec pour {fichier}")
                nb_echec += 1
    
    print(f"Traitement terminé: {nb_succes} succès, {nb_echec} échecs")

# --- NOUVELLE PARTIE - ANALYSE ET NOTATION PSA ---

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
    
    # 3. Analyse de la surface (20 points)
    
    # 4. Analyse du centrage (20 points)
    resultats['centrage'] = analyser_centrage(image_carte, gris, img_annotee, debug)
    
    # Calculer la note globale
    note_globale = resultats['coins'] + resultats['bords']  + resultats['centrage']
    
    # Convertir en note PSA (sur 10)
    note_psa = convertir_note_psa(note_globale)
    
    # Ajouter la note globale à l'image
    cv2.putText(img_annotee, f"Note: {note_globale}/80' (PSA {note_psa})", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Ajouter les notes par catégorie
    cv2.putText(img_annotee, f"Coins: {resultats['coins']}/30", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img_annotee, f"Bords: {resultats['bords']}/30", 
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img_annotee, f"Centrage: {resultats['centrage']}/20", 
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if debug:
        afficher_image(img_annotee, "Analyse de qualité")
    
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
    
    if debug:
        print(f"Scores des coins: {scores_coins}")
        print(f"Score total des coins: {score_total}/30")
    
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
    
    if debug:
        print(f"Scores des bords: {scores_bords}")
        print(f"Score total des bords: {score_total}/30")
    
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
    
    if debug:
        print(f"Différence centrage horizontal: {diff_h*100:.1f}%")
        print(f"Différence centrage vertical: {diff_v*100:.1f}%")
        print(f"Score centrage: {score_total:.1f}/20")
    
    return min(20, score_total)  # Maximum 20 points

def convertir_note_psa(note_sur_80):
    """Convertit une note sur 100 en note PSA sur 10"""
    if note_sur_80 >= 78:
        return 10  # Gem Mint
    elif note_sur_80 >= 73:
        return 9  # Mint
    elif note_sur_80 >= 65:
        return 8  # Near Mint-Mint
    elif note_sur_80 >= 55:
        return 7  # Near Mint
    elif note_sur_80 >= 45:
        return 6  # Excellent-Near Mint
    elif note_sur_80 >= 35:
        return 5  # Excellent
    elif note_sur_80 >= 25:
        return 4  # Very Good-Excellent
    elif note_sur_80 >= 15:
        return 3  # Very Good
    elif note_sur_80 >= 5:
        return 2  # Good
    else:
        return 1  # Poor

def generer_rapport_qualite(note_globale, note_psa, resultats, image_annotee, nom_fichier="rapport_qualite.jpg"):
    """Génère un rapport détaillé sur la qualité de la carte avec les lignes de détection"""
    h, w = image_annotee.shape[:2]
    
    # Créer une image pour le rapport (image originale + zone de texte)
    rapport = np.ones((h + 250, w, 3), dtype=np.uint8) * 255
    
    # Copier l'image annotée dans le rapport (elle contient déjà les rectangles de détection)
    rapport[:h, :w] = image_annotee
    
    # Ajouter un titre
    cv2.putText(rapport, f"RAPPORT D'ANALYSE POKIA", 
                (20, h + 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
    
    # Ajouter les détails
    cv2.putText(rapport, f"Note globale: {note_globale:.1f}/85", 
                (20, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(rapport, f"Equivalent POKIA: {note_psa} ({obtenir_description_psa(note_psa)})", 
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
    sauvegarder_image(rapport, nom_fichier)
    print(f"Rapport sauvegardé sous: {nom_fichier}")
    
    return rapport

def obtenir_description_psa(note_psa):
    """Retourne la description correspondant à une note POKIA"""
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

# Fonction principale modifiée pour inclure l'analyse
def scanner_et_analyser_carte(chemin_image, afficher_etapes=False, sauvegarder_rapport=True):
    """Scanne une carte et analyse sa qualité"""
    carte_redressee = scanner_carte_pokemon(chemin_image, afficher_etapes)
    
    if carte_redressee is None:
        print("Impossible d'analyser la carte, numérisation échouée.")
        return None
    
    # Analyse de la qualité
    note_globale, note_psa, resultats, img_annotee = analyser_qualite_carte(carte_redressee, debug=afficher_etapes)
    
    print(f"\n=== RÉSULTATS DE L'ANALYSE ===")
    print(f"Note globale: {note_globale:.1f}/85")
    print(f"Équivalent POKIA: {note_psa} ({obtenir_description_psa(note_psa)})")
    print(f"Détails:")
    print(f"- Coins: {resultats['coins']:.1f}/30")
    print(f"- Bords: {resultats['bords']:.1f}/30")
    print(f"- Centrage: {resultats['centrage']:.1f}/20")
    
    # Sauvegarder le rapport si demandé
    if sauvegarder_rapport:
        nom_base = os.path.splitext(os.path.basename(chemin_image))[0]
        rapport = generer_rapport_qualite(note_globale, note_psa, resultats, img_annotee, 
                                         f"{nom_base}_rapport.jpg")
        
        if afficher_etapes:
            afficher_image(rapport, "Rapport d'analyse POKIA")
    
    return note_globale, note_psa, resultats, img_annotee

def scanner_et_analyser_carte_ml(chemin_image, afficher_etapes=False, sauvegarder_rapport=True):
    """Scanne une carte et analyse sa qualité avec le modèle ML si disponible, sinon utilise l'analyse traditionnelle"""
    global model, device, ML_DISPONIBLE
    
    # Utiliser le code existant pour scanner et redresser la carte
    carte_redressee = scanner_carte_pokemon(chemin_image, afficher_etapes)
    
    if carte_redressee is None:
        print("Impossible d'analyser la carte, numérisation échouée.")
        return None
    
    # Si le modèle ML n'est pas disponible, utiliser l'analyse traditionnelle
    if not ML_DISPONIBLE:
        print("Utilisation de l'analyse traditionnelle (modèle ML non disponible)")
        return scanner_et_analyser_carte(chemin_image, afficher_etapes, sauvegarder_rapport)
    
    # Créer une copie pour les annotations
    img_annotee = carte_redressee.copy()
    
    # Analyser avec le modèle ML
    try:
        resultats_ml = analyze_card(model, device, carte_redressee)
        
        # Extraire les résultats
        note_coins = resultats_ml['coins']
        note_bords = resultats_ml['bords']
        note_centrage = resultats_ml['centrage']
        note_globale = resultats_ml['note_globale']
        note_psa = resultats_ml['grade_psa']
        
        # Stocker les résultats dans le format attendu
        resultats = {
            'coins': note_coins,
            'bords': note_bords,
            'centrage': note_centrage,
            'defauts': []
        }
        
        # Ajouter les annotations à l'image
        cv2.putText(img_annotee, f"Note (ML): {note_globale:.1f}/80 (POKIA {note_psa})", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(img_annotee, f"Coins: {note_coins:.1f}/30", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img_annotee, f"Bords: {note_bords:.1f}/30", 
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img_annotee, f"Centrage: {note_centrage:.1f}/20", 
                    (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Afficher les résultats
        print(f"\n=== RÉSULTATS DE L'ANALYSE (ML) ===")
        print(f"Note globale: {note_globale:.1f}/80")
        print(f"Équivalent PSA: {note_psa} ({obtenir_description_psa(note_psa)})")
        print(f"Détails:")
        print(f"- Coins: {note_coins:.1f}/30")
        print(f"- Bords: {note_bords:.1f}/30")
        print(f"- Centrage: {note_centrage:.1f}/20")
        
        # Sauvegarder le rapport si demandé
        if sauvegarder_rapport:
            nom_base = os.path.splitext(os.path.basename(chemin_image))[0]
            rapport = generer_rapport_qualite(note_globale, note_psa, resultats, img_annotee, 
                                             f"{nom_base}_rapport_ml.jpg")
            
            if afficher_etapes:
                afficher_image(rapport, "Rapport d'analyse POKIA (ML)")
        
        return note_globale, note_psa, resultats, img_annotee
    
    except Exception as e:
        print(f"Erreur avec l'analyse ML: {e}")
        print("Retour à l'analyse traditionnelle...")
        return scanner_et_analyser_carte(chemin_image, afficher_etapes, sauvegarder_rapport)

def traiter_dossier_avec_ml(dossier_entree, dossier_sortie=None, afficher=False):
    """Traite toutes les images d'un dossier avec analyse ML si disponible"""
    if dossier_sortie is None:
        dossier_sortie = os.path.join(dossier_entree, "resultats_ml")
    
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG']
    nb_succes = 0
    nb_echec = 0
    resultats_globaux = []
    
    for fichier in os.listdir(dossier_entree):
        if any(fichier.lower().endswith(ext) for ext in extensions):
            chemin_complet = os.path.join(dossier_entree, fichier)
            nom_base = os.path.splitext(fichier)[0]
            print(f"Traitement de {fichier}...")
            
            try:
                if ML_DISPONIBLE:
                    note_globale, note_psa, details, _ = scanner_et_analyser_carte_ml(
                        chemin_complet, afficher_etapes=afficher, sauvegarder_rapport=True)
                else:
                    note_globale, note_psa, details, _ = scanner_et_analyser_carte(
                        chemin_complet, afficher_etapes=afficher, sauvegarder_rapport=True)
                
                if note_globale is not None:
                    nb_succes += 1
                    resultats_globaux.append({
                        'fichier': fichier,
                        'note_globale': note_globale,
                        'note_Pokia': note_psa
                    })
                else:
                    print(f"⚠️ Échec pour {fichier}")
                    nb_echec += 1
            except Exception as e:
                print(f"⚠️ Erreur lors du traitement de {fichier}: {e}")
                nb_echec += 1
    
    # Générer un rapport CSV des résultats
    if resultats_globaux:
        import csv
        chemin_csv = os.path.join(dossier_sortie, "resultats_analyse.csv")
        with open(chemin_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['fichier', 'note_globale', 'note_psa'])
            writer.writeheader()
            writer.writerows(resultats_globaux)
        print(f"Rapport CSV généré: {chemin_csv}")
    
    print(f"Traitement terminé: {nb_succes} succès, {nb_echec} échecs")

# Exemple d'utilisation modifié
if __name__ == "__main__":
    # Option 1: Traiter une seule image avec analyse traditionnelle et ML
    chemin_image = "static/images/IMG_5817.JPG"
    try:
        # Méthode traditionnelle
        print("\n=== ANALYSE TRADITIONNELLE ===")
        note_globale_trad, note_psa_trad, resultats_trad, image_annotee_trad = scanner_et_analyser_carte(
            chemin_image, afficher_etapes=True)
        
        # Méthode avec apprentissage automatique
        if ML_DISPONIBLE:
            print("\n=== ANALYSE PAR APPRENTISSAGE ===")
            note_globale_ml, note_psa_ml, resultats_ml, image_annotee_ml = scanner_et_analyser_carte_ml(
                chemin_image, afficher_etapes=True)
            
            # Comparer les deux méthodes
            print("\n=== COMPARAISON DES ANALYSES ===")
            print(f"Méthode ML:             Pokia {note_psa_ml} ({note_globale_ml:.1f}/80)")
            
            # Afficher les différences
            diff_coins = abs(resultats_trad['coins'] - resultats_ml['coins'])
            diff_bords = abs(resultats_trad['bords'] - resultats_ml['bords'])
            diff_centrage = abs(resultats_trad['centrage'] - resultats_ml['centrage'])
            
            print(f"Différences: Coins={diff_coins:.1f}, Bords={diff_bords:.1f}, Centrage={diff_centrage:.1f}")
    
    except Exception as e:
        print(f"Erreur: {e}")