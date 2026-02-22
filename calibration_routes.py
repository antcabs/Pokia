"""
calibration_routes.py - Routes Flask POKIA Calibration v2
Preview caméra avec zoom-out + overlay bandes/coins dynamiques en temps réel
"""
from flask import Response, render_template, jsonify, request
import cv2
import numpy as np
import json
import os
import time

from scanner_rpi import get_scanner

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

DEFAULT_CONFIG = {
    "coin_offset":  2.0,   # % depuis le bord de la carte
    "coin_size":    4.0,   # % de la petite dimension
    "bord_hb":      2.7,   # % épaisseur bandes Haut/Bas
    "bord_gd":      4.0,   # % épaisseur bandes Gauche/Droite
    "bord_margin":  1.0,   # % marge intérieure (évite les coins)
    "canny_low":    40,
    "canny_high":   130,
    "min_area":     6.0,   # % surface minimale
    "ratio_tol":    0.35,
    "zoom_out":     1.48,  # facteur zoom-out logiciel
}

_current_config  = dict(DEFAULT_CONFIG)
_preview_config  = dict(DEFAULT_CONFIG)


# ─────────────────────────────────────────────
# Config I/O
# ─────────────────────────────────────────────

def load_config():
    global _current_config, _preview_config
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(data)
            _current_config = cfg
            _preview_config = dict(cfg)
            print(f"[POKIA] Config chargée : {CONFIG_PATH}")
        except Exception as e:
            print(f"[POKIA] Erreur config : {e}")
    return _current_config


def save_config(cfg):
    global _current_config, _preview_config
    try:
        merged = dict(DEFAULT_CONFIG)
        merged.update(cfg)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(merged, f, indent=2)
        _current_config = merged
        _preview_config = dict(merged)
        print(f"[POKIA] Config sauvegardée")
        return True
    except Exception as e:
        print(f"[POKIA] Erreur sauvegarde : {e}")
        return False


def get_config():
    return _current_config


# ─────────────────────────────────────────────
# Zoom-out logiciel (indépendant de scanner_rpi)
# ─────────────────────────────────────────────

def _zoom_out(img, facteur):
    """Réduit l'image et ajoute du padding noir pour simuler un recul."""
    if facteur <= 1.0:
        return img
    h, w = img.shape[:2]
    nw = max(1, int(w / facteur))
    nh = max(1, int(h / facteur))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    px = (w - nw) // 2
    py = (h - nh) // 2
    out[py:py+nh, px:px+nw] = small
    return out


# ─────────────────────────────────────────────
# Détection de carte
# ─────────────────────────────────────────────

def _detect_card(frame, cfg):
    h_f, w_f = frame.shape[:2]
    frame_area = h_f * w_f
    canny_low  = int(float(cfg.get('canny_low',  40)))
    canny_high = int(float(cfg.get('canny_high', 130)))
    min_area   = float(cfg.get('min_area', 6.0)) / 100.0
    ratio_tol  = float(cfg.get('ratio_tol', 0.35))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    combined = cv2.bitwise_or(edges, thresh)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8), iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8), iterations=1)

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ratio_cible = 6.3 / 8.8
    best, best_score = None, float('inf')

    for c in cnts:
        area = cv2.contourArea(c)
        if area < frame_area * min_area:
            continue
        rect = cv2.minAreaRect(c)
        lw, lh = rect[1]
        if lw == 0 or lh == 0:
            continue
        if lw > lh:
            lw, lh = lh, lw
        ratio = lw / lh
        if abs(ratio - ratio_cible) > ratio_tol:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if not (4 <= len(approx) <= 12):
            continue
        score = (1 - area/frame_area) + abs(ratio - ratio_cible) * 2
        if score < best_score:
            best_score = score
            best = c

    if best is None:
        return None

    rect = cv2.minAreaRect(best)
    box  = cv2.boxPoints(rect)
    return np.array(box, dtype=np.int32)


# ─────────────────────────────────────────────
# Overlay de calibration
# ─────────────────────────────────────────────

def _draw_overlay(frame, box, cfg):
    """Dessine les zones d'analyse avec les paramètres de calibration."""
    h_f, w_f = frame.shape[:2]

    pts = box
    x_min = int(min(p[0] for p in pts))
    x_max = int(max(p[0] for p in pts))
    y_min = int(min(p[1] for p in pts))
    y_max = int(max(p[1] for p in pts))
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return frame

    coin_off = float(cfg.get('coin_offset', 2.0)) / 100.0
    coin_sz  = float(cfg.get('coin_size',   4.0)) / 100.0
    bord_hb  = float(cfg.get('bord_hb',     2.7)) / 100.0
    bord_gd  = float(cfg.get('bord_gd',     4.0)) / 100.0
    bord_mg  = float(cfg.get('bord_margin', 1.0)) / 100.0

    # ── Contour carte ──
    cv2.drawContours(frame, [box], -1, (168, 85, 247), 2)

    # ── Coins C1-C4 ──
    taille = max(4, int(min(w, h) * coin_sz))
    coins = [
        (x_min + int(w * coin_off), y_min + int(h * coin_off)),
        (x_max - int(w * coin_off), y_min + int(h * coin_off)),
        (x_max - int(w * coin_off), y_max - int(h * coin_off)),
        (x_min + int(w * coin_off), y_max - int(h * coin_off)),
    ]
    for i, (cx, cy) in enumerate(coins):
        x1c = max(0, cx - taille // 2)
        y1c = max(0, cy - taille // 2)
        x2c = min(w_f - 1, cx + taille // 2)
        y2c = min(h_f - 1, cy + taille // 2)
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 100), 2)
        cv2.putText(frame, f"C{i+1}", (x1c + 2, y1c + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 100), 1)

    # ── Bandes de bords ──
    # Haut
    cv2.rectangle(frame,
        (max(0, x_min + int(w * bord_mg)),   max(0, y_min)),
        (min(w_f-1, x_max - int(w * bord_mg)), min(h_f-1, y_min + int(h * bord_hb))),
        (255, 165, 0), 2)
    # Bas
    cv2.rectangle(frame,
        (max(0, x_min + int(w * bord_mg)),   max(0, y_max - int(h * bord_hb))),
        (min(w_f-1, x_max - int(w * bord_mg)), min(h_f-1, y_max)),
        (255, 165, 0), 2)
    # Gauche
    cv2.rectangle(frame,
        (max(0, x_min),                        max(0, y_min + int(h * bord_mg))),
        (min(w_f-1, x_min + int(w * bord_gd)), min(h_f-1, y_max - int(h * bord_mg))),
        (255, 165, 0), 2)
    # Droite
    cv2.rectangle(frame,
        (max(0, x_max - int(w * bord_gd)),  max(0, y_min + int(h * bord_mg))),
        (min(w_f-1, x_max),                 min(h_f-1, y_max - int(h * bord_mg))),
        (255, 165, 0), 2)

    # ── Centrage ──
    cx_c = (x_min + x_max) // 2
    cy_c = (y_min + y_max) // 2
    cv2.line(frame, (cx_c, y_min), (cx_c, y_max), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (x_min, cy_c), (x_max, cy_c), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx_c, cy_c), 4, (255, 0, 255), -1)

    # ── Infos ──
    pct = w * h / (h_f * w_f) * 100
    col = (0, 255, 100) if 35 < pct < 85 else (0, 200, 255)
    cv2.putText(frame, f"Carte: {pct:.0f}%  zoom: {cfg.get('zoom_out',1.48):.2f}x",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(frame,
                f"Coins: off={coin_off*100:.1f}% sz={coin_sz*100:.1f}%  "
                f"Bords: H/B={bord_hb*100:.1f}% G/D={bord_gd*100:.1f}%",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
    return frame


def _draw_guide(frame, cfg):
    """Rectangle guide quand aucune carte n'est détectée."""
    h, w = frame.shape[:2]
    card_h = int(h * 0.70)
    card_w = int(card_h * (6.3 / 8.8))
    x1 = (w - card_w) // 2
    y1 = (h - card_h) // 2
    x2, y2 = x1 + card_w, y1 + card_h
    color = (168, 85, 247)
    cl = 24
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+cl*dx,cy), color, 3)
        cv2.line(frame, (cx,cy), (cx,cy+cl*dy), color, 3)
    cv2.putText(frame, "Placez la carte dans le cadre pour calibrer",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    cv2.putText(frame, f"zoom-out: {cfg.get('zoom_out',1.48):.2f}x",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    return frame


# ─────────────────────────────────────────────
# Générateur MJPEG
# ─────────────────────────────────────────────

def _generate_frames():
    scanner = get_scanner()
    while True:
        cfg = _preview_config   # toujours la config la plus récente

        if scanner.camera:
            try:
                raw = scanner.camera.capture_array()
                h, w = raw.shape[:2]
                scale = 640 / max(h, w)
                frame = cv2.resize(raw, (int(w*scale), int(h*scale)))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 1. Zoom-out avec le facteur courant
                zoom = float(cfg.get('zoom_out', 1.48))
                frame = _zoom_out(frame, zoom)

                # 2. Détection + overlay
                box = _detect_card(frame, cfg)
                if box is not None:
                    frame = _draw_overlay(frame, box, cfg)
                else:
                    frame = _draw_guide(frame, cfg)

                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
            except Exception as e:
                print(f"[POKIA] Calib feed: {e}")
                time.sleep(0.1)
        else:
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Camera non disponible", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            _, buf = cv2.imencode('.jpg', ph)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buf.tobytes() + b'\r\n')
            time.sleep(1)
        time.sleep(0.033)


# ─────────────────────────────────────────────
# Enregistrement des routes
# ─────────────────────────────────────────────

def register_calibration_routes(app):
    load_config()

    @app.route('/calibration')
    def calibration_page():
        return render_template('calibration.html')

    @app.route('/calibration/feed')
    def calibration_feed():
        return Response(_generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/calibration/config')
    def get_config_route():
        return jsonify({'success': True, 'config': _current_config})

    @app.route('/calibration/preview', methods=['POST'])
    def preview_config():
        """Met à jour la config de preview en temps réel (sans sauvegarder)."""
        global _preview_config
        data = request.get_json() or {}
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(data)
        _preview_config = cfg
        return jsonify({'success': True})

    @app.route('/calibration/save', methods=['POST'])
    def save_config_route():
        data = request.get_json() or {}
        ok = save_config(data)
        if ok:
            # Mettre à jour ZOOM_OUT_FACTOR dans scanner_rpi
            try:
                import scanner_rpi
                scanner_rpi.ZOOM_OUT_FACTOR = float(data.get('zoom_out', 1.48))
            except Exception:
                pass
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Erreur écriture'}), 500

    @app.route('/calibration/reset', methods=['POST'])
    def reset_config_route():
        ok = save_config(DEFAULT_CONFIG)
        return jsonify({'success': ok, 'config': DEFAULT_CONFIG})

    print("[POKIA] Routes calibration v2 enregistrées")
