"""
scanner_routes.py - Routes Flask POKIA Scanner v3
Workflow : Capturer RECTO → OK → Capturer VERSO → OK → Rapport PDF automatique
"""
from flask import Response, render_template, jsonify, request, send_from_directory, send_file
import cv2
import numpy as np
import time
import os
import json

from scanner_rpi import get_scanner, CAMERA_DISPONIBLE, appliquer_zoom_out

# Session recto/verso
scan_session = {
    'front_path': None,
    'back_path':  None,
    'front_done': False,
    'back_done':  False,
    'session_id': None,
    'pdf_path':   None,
}

current_mode = 'front'   # 'front' | 'back'

COLORS = {
    'card':     (168, 85, 247),
    'corner':   (0, 255, 100),
    'edge':     (255, 165, 0),
    'center':   (255, 0, 255),
    'ok':       (0, 255, 100),
    'warn':     (0, 200, 255),
}


# ===========================================================================
# DÉTECTION DE CARTE
# ===========================================================================

def detect_card(frame):
    """Détecte le contour de la carte dans la frame."""
    h_f, w_f = frame.shape[:2]
    frame_area = h_f * w_f
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flou = cv2.GaussianBlur(gris, (5, 5), 0)

    canny = cv2.Canny(flou, 40, 130)
    k = np.ones((5, 5), np.uint8)
    canny = cv2.dilate(canny, k, iterations=2)

    thresh = cv2.adaptiveThreshold(flou, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    combined = cv2.bitwise_or(canny, thresh)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8), iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8), iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    ratio_cible = 6.3 / 8.8
    best, best_score = None, float('inf')

    for c in contours:
        area = cv2.contourArea(c)
        if area < frame_area * 0.06:
            continue
        rect = cv2.minAreaRect(c)
        lw, lh = rect[1]
        if lw == 0 or lh == 0:
            continue
        if lw > lh:
            lw, lh = lh, lw
        ratio = lw / lh
        if abs(ratio - ratio_cible) > 0.35:
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
        return None, None

    rect = cv2.minAreaRect(best)
    box  = cv2.boxPoints(rect)
    box  = np.array(box, dtype=np.int32)
    pts  = order_pts(box)
    return best, pts


def order_pts(pts):
    pts = np.array(pts).reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype(np.int32)


# ===========================================================================
# OVERLAY D'ANALYSE
# ===========================================================================

def draw_overlay(frame, points, mode='front'):
    """Dessine les zones d'analyse pokia.py sur la carte détectée."""
    if points is None:
        return frame

    pts = np.array(points, dtype=np.int32)
    x_min = int(min(p[0] for p in pts))
    x_max = int(max(p[0] for p in pts))
    y_min = int(min(p[1] for p in pts))
    y_max = int(max(p[1] for p in pts))
    w, h = x_max - x_min, y_max - y_min
    if w <= 0 or h <= 0:
        return frame

    # Contour
    cv2.drawContours(frame, [pts], -1, COLORS['card'], 2)

    # Coins C1-C4
    taille = int(min(w, h) * 0.04)
    for i, (cx, cy) in enumerate([
        (x_min + int(w*0.02), y_min + int(h*0.02)),
        (x_max - int(w*0.02), y_min + int(h*0.02)),
        (x_max - int(w*0.02), y_max - int(h*0.02)),
        (x_min + int(w*0.02), y_max - int(h*0.02)),
    ]):
        cv2.rectangle(frame,
                      (max(0,cx-taille//2), max(0,cy-taille//2)),
                      (min(frame.shape[1],cx+taille//2), min(frame.shape[0],cy+taille//2)),
                      COLORS['corner'], 2)
        cv2.putText(frame, f"C{i+1}", (max(0,cx-taille//2)+2, max(0,cy-taille//2)+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS['corner'], 1)

    # Bords
    for bx1,by1,bx2,by2 in [
        (x_min+int(w*0.01), y_min+int(h*0.006), x_max-int(w*0.01), y_min+int(h*0.027)),
        (x_max-int(w*0.04), y_min+int(h*0.01),  x_max-int(w*0.002),y_max-int(h*0.01)),
        (x_min+int(w*0.01), y_max-int(h*0.027), x_max-int(w*0.01), y_max-int(h*0.006)),
        (x_min+int(w*0.002),y_min+int(h*0.01),  x_min+int(w*0.04), y_max-int(h*0.01)),
    ]:
        cv2.rectangle(frame,
                      (max(0,min(frame.shape[1]-1,bx1)), max(0,min(frame.shape[0]-1,by1))),
                      (max(0,min(frame.shape[1]-1,bx2)), max(0,min(frame.shape[0]-1,by2))),
                      COLORS['edge'], 2)

    # Centrage
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cv2.line(frame, (cx, y_min), (cx, y_max), COLORS['center'], 1, cv2.LINE_AA)
    cv2.line(frame, (x_min, cy), (x_max, cy), COLORS['center'], 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, COLORS['center'], -1)

    # Infos
    pct = w * h / (frame.shape[0] * frame.shape[1]) * 100
    txt_col = COLORS['ok'] if 40 < pct < 85 else COLORS['warn']
    cv2.putText(frame, f"Carte detectee - {pct:.0f}% image",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_col, 1)
    cv2.putText(frame, f"Mode: {'RECTO' if mode=='front' else 'VERSO'}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)

    if pct < 35:
        cv2.putText(frame, ">> Rapprochez la carte <<",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['warn'], 1)
    elif pct > 85:
        cv2.putText(frame, ">> Reculez un peu <<",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['warn'], 1)

    return frame


def draw_guide(frame, mode='front'):
    """Rectangle guide quand aucune carte n'est détectée."""
    h, w = frame.shape[:2]
    card_h = int(h * 0.70)
    card_w = int(card_h * (6.3 / 8.8))
    x1 = (w - card_w) // 2
    y1 = (h - card_h) // 2
    x2, y2 = x1 + card_w, y1 + card_h
    color = COLORS['card']
    cl = 28
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(cx,cy),(cx+cl*dx,cy),color,3)
        cv2.line(frame,(cx,cy),(cx,cy+cl*dy),color,3)
    for i in range(0,card_w,16):
        cv2.line(frame,(x1+i,y1),(x1+min(i+8,card_w),y1),color,1)
        cv2.line(frame,(x1+i,y2),(x1+min(i+8,card_w),y2),color,1)
    for i in range(0,card_h,16):
        cv2.line(frame,(x1,y1+i),(x1,y1+min(i+8,card_h)),color,1)
        cv2.line(frame,(x2,y1+i),(x2,y1+min(i+8,card_h)),color,1)
    label = "RECTO" if mode == 'front' else "VERSO"
    cv2.putText(frame, f"Placez le {label} dans le cadre",
                (x1, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, "En attente de detection...",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    return frame


# ===========================================================================
# STREAMING MJPEG
# ===========================================================================

def generate_frames():
    scanner = get_scanner()
    while True:
        if scanner.camera:
            try:
                image = scanner.camera.capture_array()
                h, w = image.shape[:2]
                scale = 640 / max(h, w)
                preview = cv2.resize(image, (int(w*scale), int(h*scale)))
                bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

                # Zoom-out logiciel calibré UC-350 à 15cm
                bgr = appliquer_zoom_out(bgr)

                contour, pts = detect_card(bgr)
                if contour is not None:
                    bgr = draw_overlay(bgr, pts, current_mode)
                else:
                    bgr = draw_guide(bgr, current_mode)

                _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
            except Exception as e:
                print(f"[POKIA] Streaming: {e}")
                time.sleep(0.1)
        else:
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Camera non disponible", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            _, buf = cv2.imencode('.jpg', ph)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buf.tobytes() + b'\r\n')
            time.sleep(1)
        time.sleep(0.033)


# ===========================================================================
# GÉNÉRATION PDF
# ===========================================================================

def generer_pdf(session_id, res_front, res_back):
    """Génère le rapport PDF recto+verso et retourne son chemin."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, Image as RLImage,
                                        HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER

        os.makedirs('results', exist_ok=True)
        pdf_path = os.path.join('results', f'rapport_pokia_{session_id}.pdf')

        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        s_title = ParagraphStyle('T', parent=styles['Title'], fontSize=22,
                                 textColor=colors.HexColor('#7c3aed'), spaceAfter=4)
        s_h2    = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13,
                                 textColor=colors.HexColor('#5b21b6'), spaceBefore=10, spaceAfter=4)
        s_ctr   = ParagraphStyle('C', parent=styles['Normal'], alignment=TA_CENTER)

        VIOLET = colors.HexColor('#7c3aed')
        LIGHT  = colors.HexColor('#f3e8ff')
        BORDER = colors.HexColor('#d8b4fe')

        def tableau_scores(res, couleur_titre):
            data = [
                ["Critère", "Score", "/ Max", "%"],
                ["Coins",    f"{res.get('coins',0):.1f}",    "30", f"{res.get('coins',0)/30*100:.0f}%"],
                ["Bords",    f"{res.get('bords',0):.1f}",    "30", f"{res.get('bords',0)/30*100:.0f}%"],
                ["Centrage", f"{res.get('centrage',0):.1f}", "20", f"{res.get('centrage',0)/20*100:.0f}%"],
                ["TOTAL",    f"{res.get('note_globale',0):.1f}", "80",
                 f"{res.get('note_globale',0)/80*100:.0f}%"],
            ]
            t = Table(data, colWidths=[5*cm, 4*cm, 3*cm, 4*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0),(-1,0), couleur_titre),
                ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
                ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
                ('ALIGN',      (0,0),(-1,-1),'CENTER'),
                ('VALIGN',     (0,0),(-1,-1),'MIDDLE'),
                ('ROWBACKGROUNDS',(0,1),(-1,-2),[LIGHT, colors.white]),
                ('BACKGROUND', (0,-1),(-1,-1), LIGHT),
                ('FONTNAME',   (0,-1),(-1,-1),'Helvetica-Bold'),
                ('GRID',       (0,0),(-1,-1), 0.5, BORDER),
                ('ROWHEIGHT',  (0,0),(-1,-1), 22),
            ]))
            return t

        story = []
        story.append(Paragraph("POKIA — Rapport d'Analyse", s_title))
        story.append(Paragraph(
            f"Session : {session_id} | {time.strftime('%d/%m/%Y %H:%M')}",
            s_ctr))
        story.append(HRFlowable(width="100%", thickness=2, color=VIOLET))
        story.append(Spacer(1, 0.4*cm))

        # Note combinée
        if res_front and res_back:
            note_c = round(res_front['note_globale']*0.6 + res_back['note_globale']*0.4, 1)
            psa_c  = min(res_front['note_psa'], res_back['note_psa'])
            desc_map = {10:"Gem Mint",9:"Mint",8:"Near Mint-Mint",7:"Near Mint",
                        6:"Excellent-Near Mint",5:"Excellent",4:"Very Good-Excellent",
                        3:"Very Good",2:"Good",1:"Poor"}
            story.append(Paragraph("Note Combinée (60% recto + 40% verso)", s_h2))
            tc = Table([["Note","Grade POKIA","Description"],
                        [f"{note_c}/80", f"PSA {psa_c}", desc_map.get(psa_c,"")]],
                       colWidths=[4*cm, 5*cm, 7*cm])
            tc.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),VIOLET),
                ('TEXTCOLOR', (0,0),(-1,0),colors.white),
                ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
                ('ALIGN',     (0,0),(-1,-1),'CENTER'),
                ('VALIGN',    (0,0),(-1,-1),'MIDDLE'),
                ('BACKGROUND',(0,1),(-1,-1),LIGHT),
                ('FONTNAME',  (0,1),(-1,-1),'Helvetica-Bold'),
                ('FONTSIZE',  (0,1),(-1,-1),13),
                ('GRID',      (0,0),(-1,-1),0.5,BORDER),
                ('ROWHEIGHT', (0,1),(-1,-1),32),
            ]))
            story.append(tc)
            story.append(Spacer(1, 0.5*cm))

        # Recto
        if res_front:
            story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
            story.append(Paragraph("Analyse RECTO (Devant)", s_h2))
            story.append(tableau_scores(res_front, VIOLET))
            story.append(Paragraph(
                f"Grade PSA : {res_front['note_psa']} — {res_front.get('description_psa','')}",
                s_ctr))
            if res_front.get('image_path') and os.path.exists(res_front['image_path']):
                try:
                    story.append(Spacer(1,0.3*cm))
                    story.append(RLImage(res_front['image_path'], width=7*cm, height=9.8*cm))
                except Exception:
                    pass
            story.append(Spacer(1, 0.5*cm))

        # Verso
        if res_back:
            story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
            story.append(Paragraph("Analyse VERSO (Dos)", s_h2))
            story.append(tableau_scores(res_back, colors.HexColor('#5b21b6')))
            story.append(Paragraph(
                f"Grade PSA : {res_back['note_psa']} — {res_back.get('description_psa','')}",
                s_ctr))
            if res_back.get('image_path') and os.path.exists(res_back['image_path']):
                try:
                    story.append(Spacer(1,0.3*cm))
                    story.append(RLImage(res_back['image_path'], width=7*cm, height=9.8*cm))
                except Exception:
                    pass
            story.append(Spacer(1, 0.5*cm))

        story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Rapport généré par POKIA Scanner", s_ctr))

        doc.build(story)
        print(f"[POKIA] PDF: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"[POKIA] Erreur PDF: {e}")
        return None


# ===========================================================================
# ANALYSE D'UNE IMAGE
# ===========================================================================

def analyser_image(image_path, face='front'):
    """Lance pokia.py ou pokia_back.py sur l'image et retourne un dict de résultats."""
    os.makedirs('results', exist_ok=True)
    session_id = scan_session['session_id'] or str(int(time.time()))
    result_path = os.path.join('results', f"{face}_{session_id}.jpg")

    try:
        if face == 'front':
            from pokia import (scanner_carte_pokemon, analyser_qualite_carte,
                               obtenir_description_psa, scanner_et_analyser_carte_ml,
                               ML_DISPONIBLE)
            if ML_DISPONIBLE:
                res = scanner_et_analyser_carte_ml(image_path,
                                                   afficher_etapes=False,
                                                   sauvegarder_rapport=False)
                if res:
                    note_g, note_p, rd, img_ann = res
                    cv2.imwrite(result_path, cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR))
                    return {
                        'note_globale': round(note_g, 1),
                        'note_psa': note_p,
                        'description_psa': obtenir_description_psa(note_p),
                        'coins': round(rd.get('coins', 0), 1),
                        'bords': round(rd.get('bords', 0), 1),
                        'centrage': round(rd.get('centrage', 0), 1),
                        'image_path': result_path,
                    }
            else:
                carte = scanner_carte_pokemon(image_path)
                if carte is not None:
                    note_g, note_p, rd, img_ann = analyser_qualite_carte(carte)
                    cv2.imwrite(result_path, cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR))
                    return {
                        'note_globale': round(note_g, 1),
                        'note_psa': note_p,
                        'description_psa': obtenir_description_psa(note_p),
                        'coins': round(rd.get('coins', 0), 1),
                        'bords': round(rd.get('bords', 0), 1),
                        'centrage': round(rd.get('centrage', 0), 1),
                        'image_path': result_path,
                    }
        else:
            from pokia_back import (scanner_carte_pokemon_ameliore,
                                    obtenir_description_psa as desc_dos)
            res = scanner_carte_pokemon_ameliore(image_path,
                                                 afficher_etapes=False,
                                                 sauvegarder_rapport=False)
            if res and len(res) == 2:
                carte, rd = res
                if carte is not None and rd is not None:
                    img_ann = rd.get('image_annotee', carte)
                    cv2.imwrite(result_path, cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR))
                    return {
                        'note_globale': round(rd.get('score_total', 0), 1),
                        'note_psa': rd.get('note_Pokia', rd.get('note_psa', 1)),
                        'description_psa': desc_dos(rd.get('note_Pokia', rd.get('note_psa', 1))),
                        'coins': round(rd.get('coins', {}).get('score', 0), 1),
                        'bords': round(rd.get('bords', {}).get('score', 0), 1),
                        'centrage': round(rd.get('centrage', {}).get('score', 0), 1),
                        'image_path': result_path,
                    }
    except Exception as e:
        print(f"[POKIA] Erreur analyse {face}: {e}")
    return None


# ===========================================================================
# ENREGISTREMENT DES ROUTES
# ===========================================================================

def register_scanner_routes(app):
    global scan_session, current_mode

    @app.route('/scanner')
    def scanner_page():
        return render_template('scanner.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # --- Statut ---
    @app.route('/scanner/status')
    def scanner_status():
        scanner = get_scanner()
        return jsonify({
            'camera_ok':    scanner.camera is not None,
            'front_done':   scan_session['front_done'],
            'back_done':    scan_session['back_done'],
            'session_id':   scan_session['session_id'],
            'pdf_ready':    scan_session['pdf_path'] is not None and
                            os.path.exists(scan_session['pdf_path'] or ''),
            'mode':         current_mode,
        })

    # --- Changer mode overlay ---
    @app.route('/scanner/mode', methods=['POST'])
    def set_mode():
        global current_mode
        data = request.get_json() or {}
        if data.get('mode') in ('front', 'back'):
            current_mode = data['mode']
        return jsonify({'mode': current_mode})

    # --- ÉTAPE 1 : Capturer RECTO ---
    @app.route('/scanner/capture_front', methods=['POST'])
    def capture_front():
        global scan_session, current_mode
        scanner = get_scanner()
        current_mode = 'front'
        chemin, ok = scanner.scan_face('front')
        if ok and chemin:
            sid = str(int(time.time()))
            scan_session.update({
                'front_path': chemin,
                'front_done': True,
                'back_path':  None,
                'back_done':  False,
                'session_id': sid,
                'pdf_path':   None,
            })
            return jsonify({
                'success': True,
                'session_id': sid,
                'message': 'Recto capturé ✓ — Retournez la carte puis capturez le verso.',
            })
        return jsonify({'success': False, 'message': 'Échec capture recto'}), 500

    # --- ÉTAPE 2 : Capturer VERSO ---
    @app.route('/scanner/capture_back', methods=['POST'])
    def capture_back():
        global scan_session, current_mode
        if not scan_session['front_done']:
            return jsonify({'success': False, 'message': 'Capturez d\'abord le recto'}), 400
        scanner = get_scanner()
        current_mode = 'back'
        chemin, ok = scanner.scan_face('back')
        if ok and chemin:
            scan_session['back_path'] = chemin
            scan_session['back_done'] = True
            return jsonify({
                'success': True,
                'message': 'Verso capturé ✓ — Analyse en cours...',
            })
        return jsonify({'success': False, 'message': 'Échec capture verso'}), 500

    # --- ÉTAPE 3 : Analyser + générer PDF ---
    @app.route('/scanner/analyse_and_pdf', methods=['POST'])
    def analyse_and_pdf():
        global scan_session
        if not scan_session['front_done'] or not scan_session['back_done']:
            return jsonify({'success': False, 'message': 'Les deux faces doivent être capturées'}), 400

        sid = scan_session['session_id']

        res_front = analyser_image(scan_session['front_path'], 'front')
        res_back  = analyser_image(scan_session['back_path'],  'back')

        note_c = None
        psa_c  = None
        if res_front and res_back:
            note_c = round(res_front['note_globale']*0.6 + res_back['note_globale']*0.4, 1)
            psa_c  = min(res_front['note_psa'], res_back['note_psa'])

        pdf_path = generer_pdf(sid, res_front, res_back)
        scan_session['pdf_path'] = pdf_path

        # Sauvegarder JSON session
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('results', f'session_{sid}.json'), 'w') as f:
            json.dump({'session_id': sid, 'front': res_front,
                       'back': res_back, 'note_combinee': note_c,
                       'psa_combine': psa_c}, f, indent=2)

        return jsonify({
            'success':      True,
            'session_id':   sid,
            'front':        res_front,
            'back':         res_back,
            'note_combinee': note_c,
            'psa_combine':  psa_c,
            'pdf_url':      f'/scanner/pdf/{sid}' if pdf_path else None,
        })

    # --- Télécharger PDF ---
    @app.route('/scanner/pdf/<session_id>')
    def download_pdf(session_id):
        pdf_path = os.path.join('results', f'rapport_pokia_{session_id}.pdf')
        if os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True,
                             download_name=f'rapport_pokia_{session_id}.pdf',
                             mimetype='application/pdf')
        return jsonify({'error': 'PDF non trouvé'}), 404

    # --- Réinitialiser ---
    @app.route('/scanner/reset', methods=['POST'])
    def reset_scanner():
        global scan_session, current_mode
        scan_session = {
            'front_path': None, 'back_path': None,
            'front_done': False, 'back_done': False,
            'session_id': None,  'pdf_path': None,
        }
        current_mode = 'front'
        return jsonify({'success': True})

    # --- Servir fichiers ---
    @app.route('/captures/<path:filename>')
    def serve_capture(filename):
        return send_from_directory('captures', filename)

    @app.route('/results/<path:filename>')
    def serve_result(filename):
        return send_from_directory('results', filename)

    # --- Capture simple (mode face unique) ---
    @app.route('/capture', methods=['POST'])
    def capture_simple():
        scanner = get_scanner()
        chemin = scanner.capturer_image()
        if chemin and os.path.exists(chemin):
            return jsonify({'success': True, 'filename': os.path.basename(chemin)})
        return jsonify({'success': False, 'message': 'Échec capture'}), 500

    print("[POKIA] Routes scanner v3 enregistrées")
