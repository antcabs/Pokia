"""
scanner_routes.py - Routes Flask pour le scanner caméra POKIA avec détection ML en temps réel
"""
from flask import Response, render_template, jsonify, request
import cv2
import numpy as np
import time
import threading
import os

# Importer le scanner hardware
from scanner_rpi import get_scanner, CAMERA_DISPONIBLE

# Variables globales pour le streaming
streaming_active = False
last_frame = None
frame_lock = threading.Lock()

# Configuration de la détection
DETECTION_CONFIG = {
    'show_corners': True,      # Afficher les zones de coins
    'show_edges': True,        # Afficher les zones de bords
    'show_centering': True,    # Afficher les lignes de centrage
    'corner_color': (0, 255, 0),      # Vert pour les coins
    'edge_color': (255, 165, 0),      # Orange pour les bords
    'centering_color': (255, 0, 255), # Magenta pour le centrage
    'card_outline_color': (168, 85, 247),  # Violet pour le contour
}


def detect_card_contour(frame):
    """
    Détecte le contour de la carte Pokémon dans la frame.
    Retourne le contour et les 4 coins ordonnés, ou None si non détecté.
    """
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Seuillage adaptatif
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # Opérations morphologiques
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Filtrer les contours par aire et forme
    ratio_cible = 6.3 / 8.8  # Ratio carte Pokémon
    meilleur_contour = None
    meilleur_score = float('inf')
    
    h_frame, w_frame = frame.shape[:2]
    min_area = (w_frame * h_frame) * 0.1  # Au moins 10% de l'image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Calculer le rectangle englobant
        rect = cv2.minAreaRect(contour)
        largeur, hauteur = rect[1]
        
        if largeur < hauteur:
            largeur, hauteur = hauteur, largeur
        
        if hauteur > 0:
            ratio = largeur / hauteur
            difference_ratio = abs(ratio - ratio_cible)
            
            # Vérifier la rectangularité
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            rectangularite = abs(len(approx) - 4)
            
            score = difference_ratio + rectangularite * 0.5
            
            if score < meilleur_score:
                meilleur_score = score
                meilleur_contour = contour
    
    if meilleur_contour is None:
        return None, None
    
    # Obtenir les 4 coins ordonnés
    rect = cv2.minAreaRect(meilleur_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)
    
    # Ordonner les points
    points = order_points(box)
    
    return meilleur_contour, points


def order_points(pts):
    """
    Ordonne les points: [haut-gauche, haut-droite, bas-droite, bas-gauche]
    """
    pts = np.array(pts).reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect.astype(np.int32)


def draw_ml_detection_overlay(frame, contour, points):
    """
    Dessine les zones d'analyse ML sur la frame:
    - Rectangles pour les 4 coins
    - Lignes pour les 4 bords
    - Lignes de centrage
    """
    if points is None:
        return frame
    
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Calculer les dimensions de la carte détectée
    card_width = int(np.linalg.norm(points[1] - points[0]))
    card_height = int(np.linalg.norm(points[3] - points[0]))
    
    # ===== 1. DESSINER LE CONTOUR DE LA CARTE =====
    cv2.drawContours(overlay, [points], -1, DETECTION_CONFIG['card_outline_color'], 2)
    
    # Coins stylisés
    corner_len = 25
    for i, pt in enumerate(points):
        pt = tuple(pt)
        # Déterminer les directions pour les coins
        if i == 0:  # Haut-gauche
            cv2.line(overlay, pt, (pt[0] + corner_len, pt[1]), DETECTION_CONFIG['card_outline_color'], 3)
            cv2.line(overlay, pt, (pt[0], pt[1] + corner_len), DETECTION_CONFIG['card_outline_color'], 3)
        elif i == 1:  # Haut-droite
            cv2.line(overlay, pt, (pt[0] - corner_len, pt[1]), DETECTION_CONFIG['card_outline_color'], 3)
            cv2.line(overlay, pt, (pt[0], pt[1] + corner_len), DETECTION_CONFIG['card_outline_color'], 3)
        elif i == 2:  # Bas-droite
            cv2.line(overlay, pt, (pt[0] - corner_len, pt[1]), DETECTION_CONFIG['card_outline_color'], 3)
            cv2.line(overlay, pt, (pt[0], pt[1] - corner_len), DETECTION_CONFIG['card_outline_color'], 3)
        elif i == 3:  # Bas-gauche
            cv2.line(overlay, pt, (pt[0] + corner_len, pt[1]), DETECTION_CONFIG['card_outline_color'], 3)
            cv2.line(overlay, pt, (pt[0], pt[1] - corner_len), DETECTION_CONFIG['card_outline_color'], 3)
    
    # ===== 2. ZONES D'ANALYSE DES COINS (rectangles verts) =====
    if DETECTION_CONFIG['show_corners']:
        taille_roi = int(min(card_width, card_height) * 0.08)
        
        for i, pt in enumerate(points):
            cx, cy = int(pt[0]), int(pt[1])
            
            # Ajuster la position du rectangle selon le coin
            if i == 0:  # Haut-gauche
                x1, y1 = cx, cy
                x2, y2 = cx + taille_roi, cy + taille_roi
            elif i == 1:  # Haut-droite
                x1, y1 = cx - taille_roi, cy
                x2, y2 = cx, cy + taille_roi
            elif i == 2:  # Bas-droite
                x1, y1 = cx - taille_roi, cy - taille_roi
                x2, y2 = cx, cy
            elif i == 3:  # Bas-gauche
                x1, y1 = cx, cy - taille_roi
                x2, y2 = cx + taille_roi, cy
            
            # S'assurer que les coordonnées sont dans l'image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Dessiner le rectangle de coin
            cv2.rectangle(overlay, (x1, y1), (x2, y2), DETECTION_CONFIG['corner_color'], 2)
            
            # Label
            cv2.putText(overlay, f"C{i+1}", (x1 + 2, y1 + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, DETECTION_CONFIG['corner_color'], 1)
    
    # ===== 3. ZONES D'ANALYSE DES BORDS (rectangles orange) =====
    if DETECTION_CONFIG['show_edges']:
        epaisseur = int(min(card_width, card_height) * 0.03)
        
        # Définir les 4 bords
        edges = [
            (points[0], points[1], "Haut"),    # Haut
            (points[1], points[2], "Droite"),  # Droite
            (points[2], points[3], "Bas"),     # Bas
            (points[3], points[0], "Gauche")   # Gauche
        ]
        
        for (p1, p2, name) in edges:
            # Calculer le milieu du bord
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            
            # Dessiner une ligne le long du bord
            cv2.line(overlay, tuple(p1), tuple(p2), DETECTION_CONFIG['edge_color'], 2)
            
            # Dessiner des lignes perpendiculaires pour montrer la zone d'analyse
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            longueur = np.sqrt(dx**2 + dy**2)
            if longueur > 0:
                nx, ny = -dy/longueur, dx/longueur
                
                # Dessiner quelques lignes perpendiculaires le long du bord
                for t in [0.2, 0.4, 0.6, 0.8]:
                    px = int(p1[0] + t * dx)
                    py = int(p1[1] + t * dy)
                    px_ext1 = int(px + nx * epaisseur)
                    py_ext1 = int(py + ny * epaisseur)
                    px_ext2 = int(px - nx * epaisseur)
                    py_ext2 = int(py - ny * epaisseur)
                    cv2.line(overlay, (px_ext1, py_ext1), (px_ext2, py_ext2), 
                            DETECTION_CONFIG['edge_color'], 1)
    
    # ===== 4. LIGNES DE CENTRAGE (magenta) =====
    if DETECTION_CONFIG['show_centering']:
        # Calculer le centre de la carte
        center_x = int(np.mean([p[0] for p in points]))
        center_y = int(np.mean([p[1] for p in points]))
        
        # Lignes de centrage horizontal et vertical
        min_x = int(min(p[0] for p in points))
        max_x = int(max(p[0] for p in points))
        min_y = int(min(p[1] for p in points))
        max_y = int(max(p[1] for p in points))
        
        # Ligne verticale au centre
        cv2.line(overlay, (center_x, min_y), (center_x, max_y), 
                DETECTION_CONFIG['centering_color'], 1, cv2.LINE_AA)
        
        # Ligne horizontale au centre
        cv2.line(overlay, (min_x, center_y), (max_x, center_y), 
                DETECTION_CONFIG['centering_color'], 1, cv2.LINE_AA)
        
        # Marquer le centre
        cv2.circle(overlay, (center_x, center_y), 5, DETECTION_CONFIG['centering_color'], -1)
    
    # ===== 5. TEXTE D'INFORMATION =====
    cv2.putText(overlay, "Alignez la carte", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, DETECTION_CONFIG['card_outline_color'], 2)
    
    cv2.putText(overlay, "Detection ML active", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return overlay


def draw_guide_rectangle(frame):
    """
    Dessine un rectangle de guidage quand aucune carte n'est détectée.
    """
    h, w = frame.shape[:2]
    
    # Calculer la taille du rectangle guide (proportions carte Pokémon)
    card_h = int(h * 0.85)
    card_w = int(card_h * (6.3 / 8.8))
    
    x1 = (w - card_w) // 2
    y1 = (h - card_h) // 2
    x2 = x1 + card_w
    y2 = y1 + card_h
    
    # Rectangle en pointillés
    color = (168, 85, 247)  # Violet
    
    # Dessiner le rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Coins stylisés
    corner_len = 25
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + corner_len * dx, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + corner_len * dy), color, 3)
    
    # Texte
    cv2.putText(frame, "Alignez la carte", (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "En attente de detection...", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return frame


def generate_frames_with_detection():
    """
    Générateur de frames avec détection ML en temps réel.
    """
    global streaming_active, last_frame
    
    scanner = get_scanner()
    streaming_active = True
    
    while streaming_active:
        if scanner.camera:
            try:
                # Capturer la frame
                image = scanner.camera.capture_array()
                
                # Redimensionner pour le preview
                h, w = image.shape[:2]
                scale = 640 / max(h, w)
                preview = cv2.resize(image, (int(w * scale), int(h * scale)))
                preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
                
                # Détecter la carte
                contour, points = detect_card_contour(preview_bgr)
                
                if contour is not None and points is not None:
                    # Carte détectée - dessiner l'overlay ML
                    preview_bgr = draw_ml_detection_overlay(preview_bgr, contour, points)
                else:
                    # Pas de carte - dessiner le guide
                    preview_bgr = draw_guide_rectangle(preview_bgr)
                
                # Encoder en JPEG
                _, buffer = cv2.imencode('.jpg', preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_bytes = buffer.tobytes()
                
                with frame_lock:
                    last_frame = frame_bytes
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                print(f"[POKIA] Erreur streaming: {e}")
                time.sleep(0.1)
        else:
            # Pas de caméra - image placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera non disponible", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
        
        time.sleep(0.03)  # ~30 FPS


def register_scanner_routes(app):
    """
    Enregistre les routes du scanner caméra dans l'application Flask.
    """
    
    @app.route('/scanner')
    def scanner_page():
        """Page principale du scanner avec flux vidéo."""
        return render_template('scanner.html')
    
    @app.route('/video_feed')
    def video_feed():
        """Flux vidéo MJPEG avec détection ML."""
        return Response(generate_frames_with_detection(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/capture', methods=['POST'])
    def capture_image():
        """Capture une image pour l'analyse."""
        scanner = get_scanner()
        
        try:
            # Capturer l'image
            chemin = scanner.capturer_image()
            
            if chemin and os.path.exists(chemin):
                return jsonify({
                    'success': True,
                    'path': chemin,
                    'message': 'Image capturée avec succès'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Échec de la capture'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 500
    
    @app.route('/scanner/status')
    def scanner_status():
        """Retourne le statut du scanner."""
        scanner = get_scanner()
        return jsonify({
            'camera_available': scanner.camera is not None,
            'led_available': scanner.led_rgb is not None,
            'initialized': scanner.is_initialized,
            'capturing': scanner.is_capturing
        })
    
    @app.route('/scanner/config', methods=['POST'])
    def update_scanner_config():
        """Met à jour la configuration de détection."""
        data = request.get_json()
        
        if 'show_corners' in data:
            DETECTION_CONFIG['show_corners'] = data['show_corners']
        if 'show_edges' in data:
            DETECTION_CONFIG['show_edges'] = data['show_edges']
        if 'show_centering' in data:
            DETECTION_CONFIG['show_centering'] = data['show_centering']
        
        return jsonify({'success': True, 'config': DETECTION_CONFIG})
    
    print("[POKIA] Routes scanner enregistrées")
