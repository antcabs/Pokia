"""
scanner_rpi.py - Module de controle hardware pour le scanner POKIA
"""
import os
import time
import threading

CAMERA_DISPONIBLE = False
NEOPIXEL_DISPONIBLE = False

try:
    from picamera2 import Picamera2
    CAMERA_DISPONIBLE = True
    print("[POKIA] Camera detectee")
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
CAPTURE_WIDTH = 1640
CAPTURE_HEIGHT = 1232
CAPTURE_FOLDER = "captures"

class PokiaScanner:
    def __init__(self):
        self.camera = None
        self.led_rgb = None
        self.is_initialized = False
        self.is_capturing = False
        self._lock = threading.Lock()
        os.makedirs(CAPTURE_FOLDER, exist_ok=True)

    def initialiser(self):
        print("[POKIA] Initialisation du scanner...")
        if CAMERA_DISPONIBLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_still_configuration(
                    main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"},
                    buffer_count=2)
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)
                # Utiliser tout le capteur (pas de crop/zoom)
                self.camera.set_controls({"ScalerCrop": (0, 0, 3280, 2464)})
                time.sleep(0.5)
                print("[POKIA] Camera initialisee")
            except Exception as e:
                print(f"[POKIA] Erreur camera: {e}")
                self.camera = None
        if NEOPIXEL_DISPONIBLE:
            try:
                self.led_rgb = neopixel.NeoPixel(
                    board.D18, WS2811_COUNT,
                    brightness=0.5, auto_write=False,
                    pixel_order=neopixel.GRB)
                print("[POKIA] WS2811 initialise")
            except Exception as e:
                print(f"[POKIA] Erreur WS2811: {e}")
                self.led_rgb = None
        self.is_initialized = True
        print("[POKIA] Scanner pret !")
        return True

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

    def capturer_image(self, nom_fichier=None):
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
                    cv2.imwrite(chemin, image_bgr)
                    self.rgb_status_ok()
                    print(f"[POKIA] Image capturee: {chemin}")
                    self.is_capturing = False
                    return chemin
                except Exception as e:
                    self.rgb_status_erreur()
                    print(f"[POKIA] Erreur capture: {e}")
                    self.is_capturing = False
                    return None
            else:
                print(f"[POKIA] Capture simulee: {chemin}")
                return None

    def get_preview_frame(self):
        if self.camera:
            try:
                image = self.camera.capture_array()
                import cv2
                h, w = image.shape[:2]
                scale = 640 / max(h, w)
                preview = cv2.resize(image, (int(w * scale), int(h * scale)))
                preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
                ph, pw = preview_bgr.shape[:2]
                card_h = int(ph * 0.85)
                card_w = int(card_h * (6.3 / 8.8))
                x1 = (pw - card_w) // 2
                y1 = (ph - card_h) // 2
                x2 = x1 + card_w
                y2 = y1 + card_h
                cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (168, 85, 247), 2)
                corner_len = 20
                color = (168, 85, 247)
                for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                        (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(preview_bgr, (cx, cy), (cx + corner_len * dx, cy), color, 3)
                    cv2.line(preview_bgr, (cx, cy), (cx, cy + corner_len * dy), color, 3)
                cv2.putText(preview_bgr, "Alignez la carte", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (168, 85, 247), 1)
                _, buffer = cv2.imencode('.jpg', preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                return buffer.tobytes()
            except Exception as e:
                print(f"[POKIA] Erreur preview: {e}")
                return None
        return None

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
            print(f"[POKIA] Face {face} capturee")
            return chemin, True
        else:
            self.rgb_status_erreur()
            print(f"[POKIA] Echec capture face {face}")
            return None, False

    def fermer(self):
        print("[POKIA] Fermeture du scanner...")
        self.rgb_eteindre()
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass
        print("[POKIA] Scanner ferme.")

_scanner_instance = None

def get_scanner():
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = PokiaScanner()
        _scanner_instance.initialiser()
    return _scanner_instance
