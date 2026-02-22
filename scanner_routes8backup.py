"""
scanner_routes.py - Routes Flask pour le scanner caméra POKIA
Utilise les MÊMES zones d'analyse que pokia.py (devant) et pokia_back.py (dos)
"""
from flask import Response, render_template, jsonify, request, send_from_directory
import cv2
import numpy as np
import time
import threading
import os

# Importer le scanner hardware
from scanner_rpi import get_scanner, CAMERA_DISPONIBLE

# Variables globales pour le streaming
streaming_active = False
current_analysis_mode = 'front'  # 'front' ou 'back'

# Configuration des couleurs (comme dans pokia.py)
COLORS = {
    'card_outline': (168, 85, 247),   # Violet - contour carte
    'corner_good': (0, 255, 0),        # Vert - bon score
    'corner_medium': (0, 165, 255),    # Orange - score moyen
    'corner_bad': (0, 0, 255),         # Rouge - mauvais score
    'edge': (255, 165, 0),             # Orange - bords
    'centering': (255, 0, 255),        # Magenta - centrage
}


def detect_card_contour_pokia(frame):
    """
    Détection de carte - cherche le PLUS GRAND contour rectangulaire
    qui ressemble à une carte Pokémon (ratio 6.3/8.8)
    """
    h_frame, w_frame = frame.shape[:2]
    frame_area = h_frame * w_frame
    
    # Convertir en niveaux de gris
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou pour réduire le bruit
    flou = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # === MÉTHODE 1: Canny (détection de bords) ===
    canny = cv2.Canny(flou, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    canny_dilate = cv2.dilate(canny, kernel, iterations=2)
    
    # === MÉTHODE 2: Seuillage adaptatif ===
    thresh = cv2.adaptiveThreshold(flou, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # Combiner les deux méthodes
    combined = cv2.bitwise_or(canny_dilate, thresh)
    
    # Opérations morphologiques pour fermer les trous
    kernel_large = np.ones((9, 9), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Ratio carte Pokémon
    ratio_cible = 6.3 / 8.8  # ~0.716
    
    # FILTRE STRICT: aire minimum = 15% de l'image
    # La carte doit occuper une bonne partie de l'écran
    min_area = frame_area * 0.15
    
    meilleur_contour = None
    meilleur_score = float('inf')
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Ignorer les petits contours (pas la carte entière)
        if area < min_area:
            continue
        
        # Calculer le rectangle englobant
        rect = cv2.minAreaRect(contour)
        largeur, hauteur = rect[1]
        
        if largeur == 0 or hauteur == 0:
            continue
        
        # Assurer largeur < hauteur pour le ratio (carte en portrait)
        if largeur > hauteur:
            largeur, hauteur = hauteur, largeur
        
        ratio = largeur / hauteur
        
        # Vérifier que le ratio est proche d'une carte Pokémon
        diff_ratio = abs(ratio - ratio_cible)
        
        # Tolérance de 25% sur le ratio
        if diff_ratio > 0.25:
            continue
        
        # Vérifier la rectangularité (4 côtés)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Doit avoir entre 4 et 8 sommets
        if len(approx) < 4 or len(approx) > 10:
            continue
        
        # Score: favoriser les GRANDS contours avec bon ratio
        # Plus l'aire est grande, plus le score est bas (meilleur)
        taille_score = 1 - (area / frame_area)  # 0 = très grand, 1 = petit
        ratio_score = diff_ratio * 2
        
        score = taille_score + ratio_score
        
        if score < meilleur_score:
            meilleur_score = score
            meilleur_contour = contour
    
    if meilleur_contour is None:
        return None, None
    
    # Obtenir les 4 coins
    rect = cv2.minAreaRect(meilleur_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)
    
    points = order_points(box)
    
    return meilleur_contour, points


def order_points(pts):
    """Ordonne les points: [haut-gauche, haut-droite, bas-droite, bas-gauche]"""
    pts = np.array(pts).reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect.astype(np.int32)


def draw_pokia_analysis_overlay(frame, contour, points):
    """
    Dessine les zones d'analyse EXACTEMENT comme dans pokia.py
    - Coins: rectangles aux 4 coins (comme analyser_coins)
    - Bords: rectangles le long des 4 bords (comme analyser_bords)
    - Centrage: ligne de centrage (comme analyser_centrage)
    """
    if points is None:
        return frame
    
    overlay = frame.copy()
    points = np.array(points, dtype=np.int32)
    
    # Calculer le rectangle englobant de la carte détectée
    x_min = int(min(p[0] for p in points))
    x_max = int(max(p[0] for p in points))
    y_min = int(min(p[1] for p in points))
    y_max = int(max(p[1] for p in points))
    
    # Dimensions de la carte détectée
    w = x_max - x_min
    h = y_max - y_min
    
    if w <= 0 or h <= 0:
        return frame
    
    # ===== 1. DESSINER LE CONTOUR DE LA CARTE =====
    cv2.drawContours(overlay, [points], -1, COLORS['card_outline'], 2)
    
    # Coins stylisés
    corner_len = min(20, w // 10, h // 10)
    for i, pt in enumerate(points):
        pt = tuple(pt)
        pt_next = tuple(points[(i + 1) % 4])
        pt_prev = tuple(points[(i - 1) % 4])
        
        # Direction vers suivant et précédent
        dx1 = pt_next[0] - pt[0]
        dy1 = pt_next[1] - pt[1]
        len1 = np.sqrt(dx1**2 + dy1**2)
        if len1 > 0:
            dx1, dy1 = dx1/len1, dy1/len1
        
        dx2 = pt_prev[0] - pt[0]
        dy2 = pt_prev[1] - pt[1]
        len2 = np.sqrt(dx2**2 + dy2**2)
        if len2 > 0:
            dx2, dy2 = dx2/len2, dy2/len2
        
        end1 = (int(pt[0] + dx1 * corner_len), int(pt[1] + dy1 * corner_len))
        end2 = (int(pt[0] + dx2 * corner_len), int(pt[1] + dy2 * corner_len))
        
        cv2.line(overlay, pt, end1, COLORS['card_outline'], 3)
        cv2.line(overlay, pt, end2, COLORS['card_outline'], 3)
    
    # ===== 2. ZONES DES COINS (EXACTEMENT comme pokia.py ligne 420-458) =====
    # Coordonnées des 4 coins (comme dans analyser_coins)
    coins_coords = [
        (x_min + int(w * 0.02), y_min + int(h * 0.02)),  # Haut gauche
        (x_min + int(w * 0.98), y_min + int(h * 0.02)),  # Haut droit
        (x_min + int(w * 0.98), y_min + int(h * 0.98)),  # Bas droit
        (x_min + int(w * 0.02), y_min + int(h * 0.98))   # Bas gauche
    ]
    
    # Taille ROI (comme pokia.py ligne 428)
    taille_roi = int(min(w, h) * 0.04)
    
    for i, (cx, cy) in enumerate(coins_coords):
        x1 = max(0, cx - taille_roi // 2)
        y1 = max(0, cy - taille_roi // 2)
        x2 = min(frame.shape[1], cx + taille_roi // 2)
        y2 = min(frame.shape[0], cy + taille_roi // 2)
        
        # Dessiner le rectangle de coin (vert comme dans pokia.py)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS['corner_good'], 2)
        cv2.putText(overlay, f"C{i+1}", (x1 + 2, y1 + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['corner_good'], 1)
    
    # ===== 3. ZONES DES BORDS (EXACTEMENT comme pokia.py ligne 474-513) =====
    # Coordonnées des 4 bords (comme dans analyser_bords)
    bords_coords = [
        # Haut - (x1, y1, x2, y2)
        (x_min + int(w * 0.01), y_min + int(h * 0.006), 
         x_min + int(w * 0.99), y_min + int(h * 0.027)),
        # Droite
        (x_min + int(w * 0.96), y_min + int(h * 0.01), 
         x_min + int(w * 0.998), y_min + int(h * 0.99)),
        # Bas
        (x_min + int(w * 0.01), y_min + int(h * 0.965), 
         x_min + int(w * 0.99), y_min + int(h * 0.999)),
        # Gauche
        (x_min + int(w * 0.01), y_min + int(h * 0.01), 
         x_min + int(w * 0.035), y_min + int(h * 0.99))
    ]
    
    bord_names = ["H", "D", "B", "G"]
    
    for i, (bx1, by1, bx2, by2) in enumerate(bords_coords):
        # S'assurer que les coordonnées sont dans l'image
        bx1 = max(0, min(frame.shape[1]-1, bx1))
        by1 = max(0, min(frame.shape[0]-1, by1))
        bx2 = max(0, min(frame.shape[1]-1, bx2))
        by2 = max(0, min(frame.shape[0]-1, by2))
        
        # Dessiner le rectangle de bord (orange)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), COLORS['edge'], 2)
    
    # ===== 4. LIGNES DE CENTRAGE =====
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # Ligne verticale
    cv2.line(overlay, (center_x, y_min), (center_x, y_max), 
             COLORS['centering'], 1, cv2.LINE_AA)
    # Ligne horizontale
    cv2.line(overlay, (x_min, center_y), (x_max, center_y), 
             COLORS['centering'], 1, cv2.LINE_AA)
    # Point central
    cv2.circle(overlay, (center_x, center_y), 4, COLORS['centering'], -1)
    
    # ===== 5. TEXTE D'INFO =====
    # Calculer le pourcentage de l'image occupé par la carte
    card_area = w * h
    frame_area = frame.shape[0] * frame.shape[1]
    pct_coverage = (card_area / frame_area) * 100
    
    cv2.putText(overlay, "Carte detectee - Analyse POKIA", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(overlay, f"Taille: {w}x{h}px ({pct_coverage:.0f}% image)", (10, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(overlay, f"Mode: {'DEVANT' if current_analysis_mode == 'front' else 'DOS'}", 
               (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return overlay


def draw_guide_rectangle(frame):
    """Rectangle de guidage quand aucune carte n'est détectée."""
    h, w = frame.shape[:2]
    
    card_h = int(h * 0.85)
    card_w = int(card_h * (6.3 / 8.8))
    
    x1 = (w - card_w) // 2
    y1 = (h - card_h) // 2
    x2 = x1 + card_w
    y2 = y1 + card_h
    
    color = COLORS['card_outline']
    
    # Rectangle en pointillés (simulé)
    for i in range(0, card_w, 20):
        cv2.line(frame, (x1 + i, y1), (x1 + min(i + 10, card_w), y1), color, 2)
        cv2.line(frame, (x1 + i, y2), (x1 + min(i + 10, card_w), y2), color, 2)
    for i in range(0, card_h, 20):
        cv2.line(frame, (x1, y1 + i), (x1, y1 + min(i + 10, card_h)), color, 2)
        cv2.line(frame, (x2, y1 + i), (x2, y1 + min(i + 10, card_h)), color, 2)
    
    # Coins stylisés
    corner_len = 30
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + corner_len * dx, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + corner_len * dy), color, 3)
    
    # Messages d'aide
    cv2.putText(frame, "Placez la carte dans le cadre", (x1 - 20, y1 - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "En attente de detection...", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, "Astuce: La carte doit remplir le cadre", (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
    
    return frame


def generate_frames_with_detection():
    """Générateur de frames avec détection POKIA en temps réel."""
    global streaming_active
    
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
                
                # Détecter la carte (méthode POKIA)
                contour, points = detect_card_contour_pokia(preview_bgr)
                
                if contour is not None and points is not None:
                    # Carte détectée - dessiner l'overlay POKIA
                    preview_bgr = draw_pokia_analysis_overlay(preview_bgr, contour, points)
                else:
                    # Pas de carte - dessiner le guide
                    preview_bgr = draw_guide_rectangle(preview_bgr)
                
                # Encoder en JPEG
                _, buffer = cv2.imencode('.jpg', preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                print(f"[POKIA] Erreur streaming: {e}")
                time.sleep(0.1)
        else:
            # Pas de caméra
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera non disponible", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
        
        time.sleep(0.03)  # ~30 FPS


def register_scanner_routes(app):
    """Enregistre les routes du scanner caméra dans l'application Flask."""
    
    @app.route('/scanner')
    def scanner_page():
        """Page principale du scanner."""
        return render_template('scanner.html')
    
    @app.route('/video_feed')
    def video_feed():
        """Flux vidéo MJPEG avec détection POKIA."""
        return Response(generate_frames_with_detection(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/capture', methods=['POST'])
    def capture_image():
        """Capture une image pour l'analyse."""
        scanner = get_scanner()
        
        try:
            chemin = scanner.capturer_image()
            
            if chemin and os.path.exists(chemin):
                return jsonify({
                    'success': True,
                    'path': chemin,
                    'filename': os.path.basename(chemin),
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
    
    @app.route('/analyse_capture/<path:filename>')
    def analyse_capture(filename):
        """Page d'analyse d'une capture."""
        chemin = os.path.join('captures', filename)
        if os.path.exists(chemin):
            return render_template('analyse_capture.html', image_path=chemin, filename=filename)
        else:
            return "Image non trouvée", 404
    
    @app.route('/captures/<path:filename>')
    def serve_capture(filename):
        """Sert les fichiers de capture."""
        return send_from_directory('captures', filename)
    
    @app.route('/scanner/status')
    def scanner_status():
        """Retourne le statut du scanner."""
        scanner = get_scanner()
        return jsonify({
            'camera_available': scanner.camera is not None,
            'led_available': scanner.led_rgb is not None,
            'initialized': scanner.is_initialized,
            'capturing': scanner.is_capturing,
            'mode': current_analysis_mode
        })
    
    @app.route('/scanner/mode', methods=['POST'])
    def set_scanner_mode():
        """Change le mode d'analyse (front/back)."""
        global current_analysis_mode
        data = request.get_json()
        
        if 'mode' in data and data['mode'] in ['front', 'back']:
            current_analysis_mode = data['mode']
            return jsonify({'success': True, 'mode': current_analysis_mode})
        
        return jsonify({'success': False, 'message': 'Mode invalide'}), 400
    
    print("[POKIA] Routes scanner enregistrées (mode POKIA)")
