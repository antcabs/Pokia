"""
scanner_rpi.py - Module de controle hardware POKIA
Calibré : Arducam UC-350 (IMX219) fixée à 15cm, plateforme fixe.

Problème observé : carte 504x382px dans frame 640x480 = 79% x 80%
                   → déborde horizontalement ET verticalement
Solution : zoom-out logiciel VERTICAL uniquement, facteur x1.48
           → carte occupe ~70% de la hauteur, bien visible avec marges

La capture pour l'analyse reste en pleine résolution (sans zoom-out).
"""
import os
import time
import threading

CAMERA_DISPONIBLE = False
NEOPIXEL_DISPONIBLE = False

try:
    from picamera2 import Picamera2
    CAMERA_DISPONIBLE = True
    print("[POKIA] Camera UC-350 detectee")
except ImportError:
    print("[POKIA] Picamera2 non disponible")

try:
    import board
    import neopixel
    NEOPIXEL_DISPONIBLE = True
except ImportError:
    pass

WS2811_PIN   = 18
WS2811_COUNT = 30

SENSOR_WIDTH  = 3280
SENSOR_HEIGHT = 2464
CAPTURE_WIDTH  = 1640
CAPTURE_HEIGHT = 1232

# -----------------------------------------------------------------------
# ZOOM-OUT calibré UC-350 à 15cm
# Carte observée : 79% x 80% de la frame → cible : 70% vertical
# Facteur = 0.80 / 0.70 * 1.30 (zoom précédent) ≈ 1.48
# Ajustez si besoin : augmenter = plus de recul, diminuer = moins de recul
# -----------------------------------------------------------------------
ZOOM_OUT_FACTOR = 1.48

PREVIEW_W = 640
PREVIEW_H = 480
CAPTURE_FOLDER = "captures"


def appliquer_zoom_out(image_bgr, facteur=None):
    """
    Zoom-out logiciel : réduit l'image puis ajoute du padding noir.
    La carte passe de ~80% → ~70% du cadre vertical.
    facteur : si None, utilise ZOOM_OUT_FACTOR global (modifiable via calibration)
    """
    import cv2
    import numpy as np

    h, w = image_bgr.shape[:2]
    f = facteur if facteur is not None else ZOOM_OUT_FACTOR
    new_w = max(1, int(w / f))
    new_h = max(1, int(h / f))
    reduit = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_x = (w - new_w) // 2
    pad_y = (h - new_h) // 2

    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = reduit
    return result


class PokiaScanner:
    def __init__(self):
        self.camera   = None
        self.led_rgb  = None
        self.is_initialized = False
        self.is_capturing   = False
        self._lock = threading.Lock()
        os.makedirs(CAPTURE_FOLDER, exist_ok=True)

    def initialiser(self):
        print(f"[POKIA] Init scanner UC-350 (zoom-out x{ZOOM_OUT_FACTOR})...")
        if CAMERA_DISPONIBLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_still_configuration(
                    main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"},
                    buffer_count=2
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)
                # Capteur complet = angle de vue maximal (pas de zoom numérique)
                self.camera.set_controls({
                    "ScalerCrop": (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT),
                })
                time.sleep(0.5)
                print(f"[POKIA] Camera OK ({CAPTURE_WIDTH}x{CAPTURE_HEIGHT})")
            except Exception as e:
                print(f"[POKIA] Erreur camera: {e}")
                self.camera = None

        if NEOPIXEL_DISPONIBLE:
            try:
                self.led_rgb = neopixel.NeoPixel(
                    board.D18, WS2811_COUNT,
                    brightness=0.5, auto_write=False,
                    pixel_order=neopixel.GRB
                )
            except Exception:
                self.led_rgb = None

        self.is_initialized = True
        print("[POKIA] Scanner pret !")
        return True

    # --- LEDs ---
    def rgb_couleur(self, r, g, b):
        if self.led_rgb:
            self.led_rgb.fill((r, g, b))
            self.led_rgb.show()

    def rgb_status_pret(self):   self.rgb_couleur(128, 0, 255)
    def rgb_status_scan(self):   self.rgb_couleur(0, 100, 255)
    def rgb_status_ok(self):     self.rgb_couleur(0, 255, 50)
    def rgb_status_erreur(self): self.rgb_couleur(255, 0, 0)
    def rgb_eteindre(self):
        if self.led_rgb:
            self.led_rgb.fill((0, 0, 0))
            self.led_rgb.show()

    def rgb_animation_scan(self):
        if not self.led_rgb:
            return
        def _animer():
            for _ in range(3):
                for i in range(WS2811_COUNT):
                    self.led_rgb.fill((0, 0, 30))
                    self.led_rgb[i] = (100, 0, 255)
                    if i > 0:
                        self.led_rgb[i-1] = (50, 0, 128)
                    self.led_rgb.show()
                    time.sleep(0.03)
            self.rgb_status_scan()
        threading.Thread(target=_animer, daemon=True).start()

    def rgb_animation_resultat(self, note_psa):
        if not self.led_rgb:
            return
        couleur = (0, 255, 50) if note_psa >= 8 else (255, 165, 0) if note_psa >= 6 else (255, 0, 50)
        def _animer():
            for i in range(WS2811_COUNT):
                self.led_rgb[i] = couleur
                self.led_rgb.show()
                time.sleep(0.05)
            time.sleep(0.5)
            for _ in range(3):
                self.led_rgb.fill((0, 0, 0)); self.led_rgb.show(); time.sleep(0.2)
                self.led_rgb.fill(couleur);   self.led_rgb.show(); time.sleep(0.2)
        threading.Thread(target=_animer, daemon=True).start()

    # --- Capture (pleine résolution, SANS zoom-out → pour l'analyse) ---
    def capturer_image(self, nom_fichier=None):
        if nom_fichier is None:
            nom_fichier = f"capture_{int(time.time()*1000)}.jpg"
        chemin = os.path.join(CAPTURE_FOLDER, nom_fichier)
        with self._lock:
            if self.camera:
                try:
                    self.is_capturing = True
                    self.rgb_animation_scan()
                    time.sleep(0.3)
                    image = self.camera.capture_array()
                    import cv2
                    cv2.imwrite(chemin, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    self.rgb_status_ok()
                    print(f"[POKIA] Capture: {chemin}")
                    self.is_capturing = False
                    return chemin
                except Exception as e:
                    self.rgb_status_erreur()
                    print(f"[POKIA] Erreur capture: {e}")
                    self.is_capturing = False
                    return None
            else:
                print(f"[POKIA] Pas de camera")
                return None

    # --- Preview MJPEG (AVEC zoom-out pour le streaming) ---
    def get_preview_frame(self):
        if self.camera:
            try:
                image = self.camera.capture_array()
                import cv2
                h, w = image.shape[:2]
                scale = PREVIEW_W / max(h, w)
                preview = cv2.resize(image, (int(w*scale), int(h*scale)))
                preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
                preview_bgr = appliquer_zoom_out(preview_bgr)
                ph, pw = preview_bgr.shape[:2]

                # Guide : carte portrait à 70% de la hauteur
                card_h = int(ph * 0.70)
                card_w = int(card_h * (6.3 / 8.8))
                x1 = (pw - card_w) // 2
                y1 = (ph - card_h) // 2
                x2, y2 = x1 + card_w, y1 + card_h
                color = (168, 85, 247)
                cl = 22
                for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(preview_bgr,(cx,cy),(cx+cl*dx,cy),color,3)
                    cv2.line(preview_bgr,(cx,cy),(cx,cy+cl*dy),color,3)
                cv2.putText(preview_bgr,"Alignez la carte dans le cadre",
                            (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
                _, buf = cv2.imencode('.jpg', preview_bgr, [cv2.IMWRITE_JPEG_QUALITY,75])
                return buf.tobytes()
            except Exception as e:
                print(f"[POKIA] Erreur preview: {e}")
                return None
        return None

    # --- Scan d'une face ---
    def scan_face(self, face="front"):
        nom = f"scan_{face}_{int(time.time())}.jpg"
        print(f"[POKIA] SCAN {face.upper()}")
        self.rgb_status_scan()
        time.sleep(0.5)
        chemin = self.capturer_image(nom)
        if chemin and os.path.exists(chemin):
            self.rgb_status_ok()
            return chemin, True
        self.rgb_status_erreur()
        return None, False

    def fermer(self):
        self.rgb_eteindre()
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception:
                pass


_scanner_instance = None

def get_scanner():
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = PokiaScanner()
        _scanner_instance.initialiser()
    return _scanner_instance
