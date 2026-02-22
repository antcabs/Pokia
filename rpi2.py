"""
scanner_rpi.py - Module de controle hardware pour le scanner POKIA
Calibré pour : Arducam UC-350 (IMX219, focale ~4.28mm) fixée à 15cm de la carte.

Problème observé : la carte occupe 91% de l'image → elle sort du cadre.
Solution : zoom-out logiciel x1.30 via padding noir autour de l'image.

Calcul :
  - Carte Pokémon : 63 x 88 mm
  - Zone captée à 15cm : ~129 x 97 mm
  - Carte occupe 91% vertical → facteur zoom-out = 0.91 / 0.70 = 1.30x
  - Image réduite à 492x369 + padding 74px H / 55px V → frame finale 640x480
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
    print("[POKIA] NeoPixel detecte")
except ImportError:
    print("[POKIA] NeoPixel non disponible")

WS2811_PIN = 18
WS2811_COUNT = 30

# Résolution capteur IMX219 complète (8 MP)
SENSOR_WIDTH  = 3280
SENSOR_HEIGHT = 2464

# Résolution de capture
CAPTURE_WIDTH  = 1640
CAPTURE_HEIGHT = 1232

# -----------------------------------------------------------------------
# ZOOM-OUT LOGICIEL calibré pour UC-350 à 15cm
# La carte occupe 91% de l'image brute → on réduit + on padde
# Facteur = 0.91 / 0.70 = 1.30  →  carte occupe 70% du cadre final
# -----------------------------------------------------------------------
ZOOM_OUT_FACTOR = 1.30   # Augmenter si la carte est encore trop grande
                          # Diminuer si la carte est trop petite

PREVIEW_W = 640
PREVIEW_H = 480

CAPTURE_FOLDER = "captures"


def appliquer_zoom_out(image_bgr):
    """
    Applique un zoom-out logiciel à une image BGR.
    Réduit l'image d'un facteur ZOOM_OUT_FACTOR puis ajoute du padding
    noir pour revenir à la taille PREVIEW_W x PREVIEW_H.
    Résultat : la carte occupe ~70% du cadre au lieu de 91%.
    """
    import cv2
    import numpy as np

    h, w = image_bgr.shape[:2]

    # Taille réduite
    new_w = int(w / ZOOM_OUT_FACTOR)
    new_h = int(h / ZOOM_OUT_FACTOR)
    reduit = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding pour revenir à la taille cible
    pad_x = (w - new_w) // 2
    pad_y = (h - new_h) // 2

    # Fond noir
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = reduit

    return result


class PokiaScanner:
    def __init__(self):
        self.camera = None
        self.led_rgb = None
        self.is_initialized = False
        self.is_capturing = False
        self._lock = threading.Lock()
        os.makedirs(CAPTURE_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialisation caméra
    # ------------------------------------------------------------------
    def initialiser(self):
        print("[POKIA] Initialisation scanner UC-350 (15cm, zoom-out x1.30)...")
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

                # Utiliser TOUT le capteur (ScalerCrop maximal = pas de zoom numérique)
                # Le zoom-out supplémentaire est fait en post-traitement (appliquer_zoom_out)
                self.camera.set_controls({
                    "ScalerCrop": (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT),
                })
                time.sleep(0.5)
                print(f"[POKIA] Camera OK ({CAPTURE_WIDTH}x{CAPTURE_HEIGHT}, "
                      f"capteur complet {SENSOR_WIDTH}x{SENSOR_HEIGHT}, "
                      f"zoom-out logiciel x{ZOOM_OUT_FACTOR})")
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
                print("[POKIA] WS2811 initialise")
            except Exception as e:
                print(f"[POKIA] Erreur WS2811: {e}")
                self.led_rgb = None

        self.is_initialized = True
        print("[POKIA] Scanner pret !")
        return True

    # ------------------------------------------------------------------
    # LED helpers
    # ------------------------------------------------------------------
    def allumer_eclairage(self, intensite=1.0):
        pass

    def eteindre_eclairage(self):
        pass

    def rgb_couleur(self, r, g, b):
        if self.led_rgb:
            self.led_rgb.fill((r, g, b))
            self.led_rgb.show()

    def rgb_status_pret(self):
        self.rgb_couleur(128, 0, 255)

    def rgb_status_scan(self):
        self.rgb_couleur(0, 100, 255)

    def rgb_status_ok(self):
        self.rgb_couleur(0, 255, 50)

    def rgb_status_erreur(self):
        self.rgb_couleur(255, 0, 0)

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
                        self.led_rgb[i - 1] = (50, 0, 128)
                    self.led_rgb.show()
                    time.sleep(0.03)
            self.rgb_status_scan()
        threading.Thread(target=_animer, daemon=True).start()

    def rgb_animation_resultat(self, note_psa):
        if not self.led_rgb:
            return
        if note_psa >= 8:
            couleur = (0, 255, 50)
        elif note_psa >= 6:
            couleur = (255, 165, 0)
        else:
            couleur = (255, 0, 50)
        def _animer():
            for i in range(WS2811_COUNT):
                self.led_rgb[i] = couleur
                self.led_rgb.show()
                time.sleep(0.05)
            time.sleep(0.5)
            for _ in range(3):
                self.led_rgb.fill((0, 0, 0))
                self.led_rgb.show()
                time.sleep(0.2)
                self.led_rgb.fill(couleur)
                self.led_rgb.show()
                time.sleep(0.2)
        threading.Thread(target=_animer, daemon=True).start()

    # ------------------------------------------------------------------
    # Capture image (SANS zoom-out → image pleine résolution pour l'analyse)
    # ------------------------------------------------------------------
    def capturer_image(self, nom_fichier=None):
        """Capture l'image brute (pleine résolution, sans zoom-out) pour l'analyse."""
        if nom_fichier is None:
            timestamp = int(time.time() * 1000)
            nom_fichier = f"capture_{timestamp}.jpg"
        chemin = os.path.join(CAPTURE_FOLDER, nom_fichier)
        with self._lock:
            if self.camera:
                try:
                    self.is_capturing = True
                    self.rgb_animation_scan()
                    time.sleep(0.3)
                    image = self.camera.capture_array()
                    import cv2
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # Capture SANS zoom-out : image brute pour pokia.py
                    cv2.imwrite(chemin, image_bgr)
                    self.rgb_status_ok()
                    print(f"[POKIA] Image capturee (brute): {chemin}")
                    self.is_capturing = False
                    return chemin
                except Exception as e:
                    self.rgb_status_erreur()
                    print(f"[POKIA] Erreur capture: {e}")
                    self.is_capturing = False
                    return None
            else:
                print(f"[POKIA] Capture simulee (pas de camera): {chemin}")
                return None

    # ------------------------------------------------------------------
    # Preview frame (streaming MJPEG) — AVEC zoom-out logiciel
    # ------------------------------------------------------------------
    def get_preview_frame(self):
        """
        Retourne une frame MJPEG avec zoom-out logiciel x1.30.
        La carte occupe ~70% du cadre (au lieu de 91% en brut).
        """
        if self.camera:
            try:
                image = self.camera.capture_array()
                import cv2
                import numpy as np

                # 1. Réduire à la taille preview
                h, w = image.shape[:2]
                scale = PREVIEW_W / max(h, w)
                preview = cv2.resize(image, (int(w * scale), int(h * scale)))
                preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

                # 2. Appliquer le zoom-out logiciel
                preview_bgr = appliquer_zoom_out(preview_bgr)
                ph, pw = preview_bgr.shape[:2]

                # 3. Guide rectangle à 70% du cadre
                card_h = int(ph * 0.70)
                card_w = int(card_h * (6.3 / 8.8))
                x1 = (pw - card_w) // 2
                y1 = (ph - card_h) // 2
                x2 = x1 + card_w
                y2 = y1 + card_h

                color = (168, 85, 247)
                # Coins du guide
                corner_len = 22
                for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                        (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(preview_bgr, (cx, cy), (cx + corner_len * dx, cy), color, 3)
                    cv2.line(preview_bgr, (cx, cy), (cx, cy + corner_len * dy), color, 3)
                # Tirets sur les bords
                for i in range(0, card_w, 18):
                    cv2.line(preview_bgr, (x1+i, y1), (x1+min(i+9, card_w), y1), color, 1)
                    cv2.line(preview_bgr, (x1+i, y2), (x1+min(i+9, card_w), y2), color, 1)
                for i in range(0, card_h, 18):
                    cv2.line(preview_bgr, (x1, y1+i), (x1, y1+min(i+9, card_h)), color, 1)
                    cv2.line(preview_bgr, (x2, y1+i), (x2, y1+min(i+9, card_h)), color, 1)

                cv2.putText(preview_bgr, "Alignez la carte dans le cadre",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                _, buffer = cv2.imencode('.jpg', preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                return buffer.tobytes()
            except Exception as e:
                print(f"[POKIA] Erreur preview: {e}")
                return None
        return None

    # ------------------------------------------------------------------
    # Scan d'une face (recto ou verso)
    # ------------------------------------------------------------------
    def scan_face(self, face="front"):
        timestamp = int(time.time())
        nom = f"scan_{face}_{timestamp}.jpg"
        print(f"\n[POKIA] === SCAN {face.upper()} ===")
        self.rgb_status_scan()
        time.sleep(0.5)
        chemin = self.capturer_image(nom)
        if chemin and os.path.exists(chemin):
            self.rgb_status_ok()
            time.sleep(0.3)
            print(f"[POKIA] Face {face} capturee: {chemin}")
            return chemin, True
        else:
            self.rgb_status_erreur()
            print(f"[POKIA] Echec capture face {face}")
            return None, False

    # ------------------------------------------------------------------
    # Fermeture
    # ------------------------------------------------------------------
    def fermer(self):
        print("[POKIA] Fermeture du scanner...")
        self.rgb_eteindre()
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception:
                pass
        print("[POKIA] Scanner ferme.")


_scanner_instance = None


def get_scanner():
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = PokiaScanner()
        _scanner_instance.initialiser()
    return _scanner_instance
