from flask import Flask, render_template, request, redirect, send_file, url_for, flash, send_from_directory, jsonify
import os
import cv2
import qrcode
import numpy as np
from werkzeug.utils import secure_filename
import base64
import uuid
import time
import threading
import json
from flask_paginate import Pagination, get_page_parameter
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime as dt
from mondial_relay import MondialRelayService
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_login import current_user, login_required
from io import BytesIO

# Importer les routes du scanner caméra Raspberry Pi
from scanner_routes import register_scanner_routes

# Importer les routes de calibration
from calibration_routes import register_calibration_routes

# Importer les fonctions pour l'analyse du devant de la carte
from pokia import (
    scanner_carte_pokemon, 
    analyser_qualite_carte, 
    generer_rapport_qualite,
    obtenir_description_psa, 
    init_model, 
    scanner_et_analyser_carte_ml,
    ML_DISPONIBLE
)

# Importer les fonctions pour l'analyse du dos de la carte
from pokia_back import (
    scanner_carte_pokemon_ameliore,
    analyser_qualite_carte as analyser_qualite_carte_dos,
    generer_rapport_qualite as generer_rapport_qualite_dos,
    obtenir_description_psa as obtenir_description_psa_dos
)

app = Flask(__name__)
app.secret_key = "pokemon_scanner_secret_key"

# Configuration pour l'analyse des cartes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['USE_ML'] = True  # Option pour activer ou désactiver l'analyse ML

# Configuration de la base de données MySQL pour la marketplace
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'pokemon_marketplace.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuration Mondial Relay (à ajouter dans votre configuration)
MONDIAL_RELAY_ENSEIGNE = "BDTEST13"  # À remplacer par votre code enseigne réel
MONDIAL_RELAY_PRIVATE_KEY = "TestAPI1key"  # À remplacer par votre clé privée réelle

# Initialiser le service Mondial Relay
mondial_relay_service = MondialRelayService(
    enseigne=MONDIAL_RELAY_ENSEIGNE,
    private_key=MONDIAL_RELAY_PRIVATE_KEY
)

# Initialisation de SQLAlchemy
db = SQLAlchemy(app)

# S'assurer que les dossiers nécessaires existent
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/images/cards', exist_ok=True)
os.makedirs('captures', exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Veuillez vous connecter pour accéder à cette page."
login_manager.login_message_category = "info"

# Vérifier la disponibilité du modèle ML au démarrage de l'application
if app.config['USE_ML'] and ML_DISPONIBLE:
    init_model()
    print("Modèle d'apprentissage automatique initialisé")
else:
    print("Analyse par apprentissage automatique non disponible ou désactivée")

# Définition du modèle de données pour les cartes Pokémon (marketplace)
class Card(db.Model):
    __tablename__ = 'cards'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    rarity = db.Column(db.String(50), nullable=False)
    psa_grade = db.Column(db.String(20))
    price = db.Column(db.Float, nullable=False)
    old_price = db.Column(db.Float)
    description = db.Column(db.Text)
    image_path = db.Column(db.String(255), nullable=False)
    date_added = db.Column(db.DateTime, default=dt.utcnow)
    
    def __repr__(self):
        return f'<Card {self.name}>'

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    date_registered = db.Column(db.DateTime, default=dt.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
# Ajoutez ces modèles à votre app.py, après les autres modèles existants

class CartItem(db.Model):
    __tablename__ = 'cart_items'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    card_id = db.Column(db.Integer, db.ForeignKey('cards.id'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    grading_service = db.Column(db.Boolean, default=False)  # Nouveau champ
    date_added = db.Column(db.DateTime, default=dt.utcnow)
    
    # Relations
    user = db.relationship('User', backref=db.backref('cart_items', lazy=True))
    card = db.relationship('Card', backref=db.backref('cart_items', lazy=True))
    
    def __repr__(self):
        return f'<CartItem: {self.quantity} x Card {self.card_id} for User {self.user_id}>'
    
class Order(db.Model):
    __tablename__ = 'orders'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')
    total_amount = db.Column(db.Float, nullable=False)
    shipping_address = db.Column(db.Text, nullable=False)
    order_date = db.Column(db.DateTime, default=dt.utcnow)
    
    # Relation
    user = db.relationship('User', backref=db.backref('orders', lazy=True))
    
    def __repr__(self):
        return f'<Order {self.id} by User {self.user_id}: {self.status}>'

class OrderItem(db.Model):
    __tablename__ = 'order_items'
    
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    card_id = db.Column(db.Integer, db.ForeignKey('cards.id'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    price_at_purchase = db.Column(db.Float, nullable=False)
    
    # Relations
    order = db.relationship('Order', backref=db.backref('items', lazy=True))
    card = db.relationship('Card', backref=db.backref('order_items', lazy=True))
    
    def __repr__(self):
        return f'<OrderItem: {self.quantity} x Card {self.card_id} in Order {self.order_id}>'

class Favorite(db.Model):
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    card_id = db.Column(db.Integer, db.ForeignKey('cards.id'), nullable=False)
    date_added = db.Column(db.DateTime, default=dt.utcnow)
    
    # Relations
    user = db.relationship('User', backref=db.backref('favorites', lazy=True))
    card = db.relationship('Card', backref=db.backref('favorited_by', lazy=True))
    
    # Contrainte unique pour éviter les doublons
    __table_args__ = (db.UniqueConstraint('user_id', 'card_id', name='uq_user_card_favorite'),)
    
    def __repr__(self):
        return f'<Favorite {self.user_id}:{self.card_id}>'

def generate_card_qr_code(card_id, verification_url=None):
    """
    Génère un QR code pour l'authenticité d'une carte Pokémon
    """
    # Si aucune URL n'est fournie, utiliser l'URL de vérification par défaut
    if verification_url is None:
        verification_url = f"/verify-card/{card_id}"
    
    # Créer l'objet QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(verification_url)
    qr.make(fit=True)
    
    # Créer une image du QR code
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Enregistrer dans un buffer
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return img_io

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def encode_image_to_base64(image_path):
    """Convertit une image en base64 pour l'affichage HTML"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def encode_cv2_image_to_base64(cv2_image):
    """Convertit une image CV2 en base64 pour l'affichage HTML"""
    _, buffer = cv2.imencode('.jpg', cv2_image)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return encoded_string

@app.template_filter('nl2br')
def nl2br_filter(text):
    if not text:
        return ""
    return text.replace('\n', '<br>')

@app.route('/')
def index():
    return render_template('index.html', ml_available=ML_DISPONIBLE and app.config['USE_ML'])

@app.route('/toggle_favorite/<int:card_id>', methods=['POST'])
@login_required
def toggle_favorite(card_id):
    card = Card.query.get_or_404(card_id)
    
    # Vérifier si la carte est déjà dans les favoris
    existing_favorite = Favorite.query.filter_by(
        user_id=current_user.id, 
        card_id=card_id
    ).first()
    
    if existing_favorite:
        # Si la carte est déjà en favoris, on la retire
        db.session.delete(existing_favorite)
        flash(f'"{card.name}" a été retiré de vos favoris.', 'info')
    else:
        # Sinon, on l'ajoute aux favoris
        new_favorite = Favorite(user_id=current_user.id, card_id=card_id)
        db.session.add(new_favorite)
        flash(f'"{card.name}" a été ajouté à vos favoris !', 'success')
    
    db.session.commit()
    
    # Redirect back to the page the user was on
    return redirect(request.referrer or url_for('marketplace'))

@app.route('/all-listings')
@login_required
def all_listings():
    # Récupérer les paramètres de l'URL
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 12  # Nombre de cartes par page
    
    # Filtres
    card_type = request.args.get('type')
    rarity = request.args.get('rarity')
    grade = request.args.get('grade')
    query = request.args.get('query')
    sort = request.args.get('sort', 'newest')
    
    # Construire la requête de base
    cards_query = Card.query
    
    # Appliquer les filtres
    if card_type:
        cards_query = cards_query.filter(Card.type.ilike(f'%{card_type}%'))
    if rarity:
        cards_query = cards_query.filter(Card.rarity.ilike(f'%{rarity}%'))
    if grade:
        if grade == 'non-grade':
            cards_query = cards_query.filter(Card.psa_grade.is_(None))
        else:
            # Extraire le numéro PSA (par exemple, "psa-10" -> "10")
            psa_number = grade.split('-')[1] if '-' in grade else None
            if psa_number:
                cards_query = cards_query.filter(Card.psa_grade.ilike(f'%{psa_number}%'))
    if query:
        cards_query = cards_query.filter(Card.name.ilike(f'%{query}%'))
    
    # Appliquer le tri
    if sort == 'price-asc':
        cards_query = cards_query.order_by(Card.price.asc())
    elif sort == 'price-desc':
        cards_query = cards_query.order_by(Card.price.desc())
    elif sort == 'alpha':
        cards_query = cards_query.order_by(Card.name.asc())
    else:  # par défaut, tri par date (plus récent d'abord)
        cards_query = cards_query.order_by(Card.date_added.desc())
    
    # Paginer les résultats
    total = cards_query.count()
    cards_paginated = cards_query.offset((page - 1) * per_page).limit(per_page).all()
    
    pagination = Pagination(page=page, per_page=per_page, total=total, 
                           css_framework='bootstrap4')
    
    return render_template('all_listings.html', 
                          cards=cards_paginated,
                          pagination=pagination,
                          page=page,
                          per_page=per_page)

@app.route('/favorites')
@login_required
def favorites():
    # Récupérer toutes les cartes favorites de l'utilisateur
    favorite_cards = db.session.query(Card).\
        join(Favorite, Card.id == Favorite.card_id).\
        filter(Favorite.user_id == current_user.id).\
        all()
    
    return render_template('favorites.html', favorite_cards=favorite_cards)

    

@app.route('/upload', methods=['POST'])
def upload_file():
    # Vérifier si un fichier a été envoyé
    if 'file' not in request.files:
        flash('Aucun fichier trouvé', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Vérifier si un fichier a été sélectionné
    if file.filename == '':
        flash('Aucun fichier sélectionné', 'error')
        return redirect(request.url)
    
    # Vérifier si le fichier est autorisé
    if not allowed_file(file.filename):
        flash('Format de fichier non pris en charge. Utilisez JPG, JPEG, PNG ou BMP.', 'error')
        return redirect(request.url)
    
    # Récupérer les options d'analyse
    use_ml = 'use_ml' in request.form and request.form['use_ml'] == 'on' and ML_DISPONIBLE and app.config['USE_ML']
    card_side = request.form.get('card_side', 'front')  # Par défaut, analyse du devant de la carte
    
    # Sauvegarder le fichier
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    base_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], base_filename)
    file.save(file_path)
    
    # Traiter l'image selon la face de la carte sélectionnée
    try:
        if card_side == 'back':
            # Analyse du dos de la carte
            carte_redressee, resultats_analyse = scanner_carte_pokemon_ameliore(
                file_path, afficher_etapes=False, sauvegarder_rapport=False)
            
            if carte_redressee is None or resultats_analyse is None:
                flash("Impossible de détecter correctement le dos de la carte Pokémon dans l'image.", 'error')
                return redirect(url_for('index'))
            
            # Extraire les résultats
            note_globale = resultats_analyse["score_total"]
            note_psa = resultats_analyse["note_Pokia"]
            img_annotee = resultats_analyse["image_annotee"]
            
            # Préparer les données pour le template
            resultats = {
                'coins': resultats_analyse["coins"]["score"],
                'bords': resultats_analyse["bords"]["score"],
                'centrage': resultats_analyse["centrage"]["score"],
            }
            
            analyse_type = "dos de carte"
            
            # Générer et sauvegarder le rapport
            result_filename = f"rapport_dos_{timestamp}_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            with open(result_path, 'wb') as f:
                cv2.imwrite(result_path, cv2.cvtColor(img_annotee, cv2.COLOR_RGB2BGR))
            
        else:
            # Analyse du devant de la carte (code existant)
            if use_ml:
                # Numériser et analyser avec ML
                note_globale, note_psa, resultats, img_annotee = scanner_et_analyser_carte_ml(
                    file_path, afficher_etapes=False, sauvegarder_rapport=False)
                analyse_type = "machine learning"
            else:
                # Numériser la carte d'abord
                carte_redressee = scanner_carte_pokemon(file_path, afficher_etapes=False)
                
                if carte_redressee is None:
                    flash("Impossible de détecter correctement la carte Pokémon dans l'image.", 'error')
                    return redirect(url_for('index'))
                
                # Puis analyser la qualité
                note_globale, note_psa, resultats, img_annotee = analyser_qualite_carte(carte_redressee, debug=False)
                analyse_type = "traditionnelle"
            
            # Générer et sauvegarder le rapport
            result_filename = f"rapport_{timestamp}_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            rapport = generer_rapport_qualite(note_globale, note_psa, resultats, img_annotee, result_path)
        
        # Convertir les images en base64 pour l'affichage
        original_image_b64 = encode_image_to_base64(file_path)
        
        # Pour le rapport, on gère différemment selon la face analysée
        if card_side == 'back':
            rapport_b64 = encode_cv2_image_to_base64(img_annotee)
        else:
            # Pour le devant, on utilise rapport produit par generer_rapport_qualite
            rapport_b64 = encode_cv2_image_to_base64(rapport if 'rapport' in locals() else img_annotee)
        
        # Préparer les détails pour le modèle
        details = {
            'note_globale': round(note_globale, 1),
            'note_psa': note_psa,
            'description_psa': obtenir_description_psa(note_psa),
            'coins': round(resultats['coins'], 1),
            'bords': round(resultats['bords'], 1),
            'centrage': round(resultats['centrage'], 1),
            'analyse_type': analyse_type,
            'original_image': original_image_b64,
            'rapport': rapport_b64,
            'result_filename': result_filename,
            'card_side': card_side
        }
        
        # Si nous avons une carte redressée
        if 'carte_redressee' in locals():
            if isinstance(carte_redressee, tuple) and len(carte_redressee) > 0:
                # Si c'est un tuple (cas du dos de carte)
                details['carte_redressee'] = encode_cv2_image_to_base64(carte_redressee[0])
            else:
                # Cas normal
                details['carte_redressee'] = encode_cv2_image_to_base64(carte_redressee)
        else:
            # En mode ML ou si pas disponible, utiliser l'image originale
            details['carte_redressee'] = original_image_b64
        
        # Si l'analyse du dos fournit des données de centrage BLM (Black Label Metrics)
        if card_side == 'back' and 'centrage' in resultats_analyse:
            if 'marges' in resultats_analyse['centrage']:
                marges = resultats_analyse['centrage']['marges']
                details['blm_grade'] = 'A'  # Valeur par défaut
                details['blm_ratio_horizontal'] = f"{max(marges['gauche'], marges['droite']) / min(marges['gauche'], marges['droite']):.2f}"
                details['blm_ratio_vertical'] = f"{max(marges['haut'], marges['bas']) / min(marges['haut'], marges['bas']):.2f}"
                details['blm_grade_horizontal'] = 'A' if float(details['blm_ratio_horizontal']) <= 1.5 else 'B'
                details['blm_grade_vertical'] = 'A' if float(details['blm_ratio_vertical']) <= 1.5 else 'B'
                details['blm_acceptable_h'] = float(details['blm_ratio_horizontal']) <= 1.5
                details['blm_acceptable_v'] = float(details['blm_ratio_vertical']) <= 1.5
                details['blm_acceptable'] = details['blm_acceptable_h'] and details['blm_acceptable_v']
                details['blm_marge_gauche'] = int(marges['gauche'])
                details['blm_marge_droite'] = int(marges['droite'])
                details['blm_marge_haut'] = int(marges['haut'])
                details['blm_marge_bas'] = int(marges['bas'])
        
        return render_template('result.html', details=details)
    
    except Exception as e:
        flash(f"Une erreur s'est produite lors du traitement de l'image: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/analyse_complete', methods=['POST'])
def analyse_complete():
    """Analyse à la fois le devant et le dos d'une carte Pokémon"""
    # Vérifier si les fichiers ont été envoyés
    if 'file_front' not in request.files or 'file_back' not in request.files:
        flash('Veuillez télécharger à la fois le devant et le dos de la carte', 'error')
        return redirect(url_for('index'))
    
    file_front = request.files['file_front']
    file_back = request.files['file_back']
    
    # Vérifier si les fichiers ont été sélectionnés
    if file_front.filename == '' or file_back.filename == '':
        flash('Veuillez sélectionner les deux images', 'error')
        return redirect(url_for('index'))
    
    # Vérifier si les fichiers sont autorisés
    if not (allowed_file(file_front.filename) and allowed_file(file_back.filename)):
        flash('Format de fichier non pris en charge. Utilisez JPG, JPEG, PNG ou BMP.', 'error')
        return redirect(url_for('index'))
    
    # Récupérer les options d'analyse
    use_ml = 'use_ml' in request.form and request.form['use_ml'] == 'on' and ML_DISPONIBLE and app.config['USE_ML']
    
    # Sauvegarder les fichiers
    timestamp = int(time.time())
    
    # Sauvegarder le devant de la carte
    filename_front = secure_filename(file_front.filename)
    base_filename_front = f"{timestamp}_front_{filename_front}"
    file_path_front = os.path.join(app.config['UPLOAD_FOLDER'], base_filename_front)
    file_front.save(file_path_front)
    
    # Sauvegarder le dos de la carte
    filename_back = secure_filename(file_back.filename)
    base_filename_back = f"{timestamp}_back_{filename_back}"
    file_path_back = os.path.join(app.config['UPLOAD_FOLDER'], base_filename_back)
    file_back.save(file_path_back)
    
    try:
        # Analyse du devant de la carte
        if use_ml:
            note_globale_front, note_psa_front, resultats_front, img_annotee_front = scanner_et_analyser_carte_ml(
                file_path_front, afficher_etapes=False, sauvegarder_rapport=False)
        else:
            carte_redressee_front = scanner_carte_pokemon(file_path_front, afficher_etapes=False)
            if carte_redressee_front is None:
                flash("Impossible de détecter correctement le devant de la carte.", 'error')
                return redirect(url_for('index'))
            note_globale_front, note_psa_front, resultats_front, img_annotee_front = analyser_qualite_carte(
                carte_redressee_front, debug=False)
        
        # Analyse du dos de la carte
        carte_redressee_back, resultats_back = scanner_carte_pokemon_ameliore(
            file_path_back, afficher_etapes=False, sauvegarder_rapport=False)
        
        if carte_redressee_back is None or resultats_back is None:
            flash("Impossible de détecter correctement le dos de la carte.", 'error')
            return redirect(url_for('index'))
        
        note_globale_back = resultats_back["score_total"]
        note_psa_back = resultats_back["note_psa"]
        img_annotee_back = resultats_back["image_annotee"]
        
        # Calcul de la note combinée (moyenne pondérée)
        # La note du devant compte pour 60%, celle du dos pour 40%
        note_combinee = (note_globale_front * 0.6) + (note_globale_back * 0.4)
        note_psa_combinee = min(note_psa_front, note_psa_back)  # On prend la note la plus basse
        
        # Générer et sauvegarder les rapports
        result_filename_front = f"rapport_front_{timestamp}.jpg"
        result_path_front = os.path.join(app.config['RESULTS_FOLDER'], result_filename_front)
        rapport_front = generer_rapport_qualite(note_globale_front, note_psa_front, resultats_front, img_annotee_front, result_path_front)
        
        result_filename_back = f"rapport_back_{timestamp}.jpg"
        result_path_back = os.path.join(app.config['RESULTS_FOLDER'], result_filename_back)
        with open(result_path_back, 'wb') as f:
            cv2.imwrite(result_path_back, cv2.cvtColor(img_annotee_back, cv2.COLOR_RGB2BGR))
        
        # Convertir les images en base64 pour l'affichage
        original_front_b64 = encode_image_to_base64(file_path_front)
        original_back_b64 = encode_image_to_base64(file_path_back)
        rapport_front_b64 = encode_cv2_image_to_base64(rapport_front)
        rapport_back_b64 = encode_cv2_image_to_base64(img_annotee_back)
        
        # Si nous avons une carte redressée pour le devant
        if 'carte_redressee_front' in locals():
            carte_redressee_front_b64 = encode_cv2_image_to_base64(carte_redressee_front)
        else:
            carte_redressee_front_b64 = original_front_b64
        
        # Si nous avons une carte redressée pour le dos
        if isinstance(carte_redressee_back, tuple) and len(carte_redressee_back) > 0:
            carte_redressee_back_b64 = encode_cv2_image_to_base64(carte_redressee_back[0])
        else:
            carte_redressee_back_b64 = encode_cv2_image_to_base64(carte_redressee_back)
        
        # Préparer les détails pour le modèle
        details_front = {
            'note_globale': round(note_globale_front, 1),
            'note_psa': note_psa_front,
            'description_psa': obtenir_description_psa(note_psa_front),
            'coins': round(resultats_front['coins'], 1),
            'bords': round(resultats_front['bords'], 1),
            'centrage': round(resultats_front['centrage'], 1),
            'original_image': original_front_b64,
            'rapport': rapport_front_b64,
            'carte_redressee': carte_redressee_front_b64,
            'result_filename': result_filename_front
        }
        
        details_back = {
            'note_globale': round(note_globale_back, 1),
            'note_psa': note_psa_back,
            'description_psa': obtenir_description_psa_dos(note_psa_back),
            'coins': round(resultats_back["coins"]["score"], 1),
            'bords': round(resultats_back["bords"]["score"], 1),
            'centrage': round(resultats_back["centrage"]["score"], 1),
            'original_image': original_back_b64,
            'rapport': rapport_back_b64,
            'carte_redressee': carte_redressee_back_b64,
            'result_filename': result_filename_back
        }
        
        # Si l'analyse du dos fournit des données de centrage BLM
        if 'centrage' in resultats_back:
            if 'marges' in resultats_back['centrage']:
                marges = resultats_back['centrage']['marges']
                details_back['blm_grade'] = 'A'  # Valeur par défaut
                details_back['blm_ratio_horizontal'] = f"{max(marges['gauche'], marges['droite']) / min(marges['gauche'], marges['droite']):.2f}"
                details_back['blm_ratio_vertical'] = f"{max(marges['haut'], marges['bas']) / min(marges['haut'], marges['bas']):.2f}"
                details_back['blm_grade_horizontal'] = 'A' if float(details_back['blm_ratio_horizontal']) <= 1.5 else 'B'
                details_back['blm_grade_vertical'] = 'A' if float(details_back['blm_ratio_vertical']) <= 1.5 else 'B'
                details_back['blm_acceptable_h'] = float(details_back['blm_ratio_horizontal']) <= 1.5
                details_back['blm_acceptable_v'] = float(details_back['blm_ratio_vertical']) <= 1.5
                details_back['blm_acceptable'] = details_back['blm_acceptable_h'] and details_back['blm_acceptable_v']
                details_back['blm_marge_gauche'] = int(marges['gauche'])
                details_back['blm_marge_droite'] = int(marges['droite'])
                details_back['blm_marge_haut'] = int(marges['haut'])
                details_back['blm_marge_bas'] = int(marges['bas'])
        
        # Résultat combiné
        details_combined = {
            'note_globale': round(note_combinee, 1),
            'note_psa': note_psa_combinee,
            'description_psa': obtenir_description_psa(note_psa_combinee)
        }
        
        return render_template(
            'result_complete.html', 
            details_front=details_front,
            details_back=details_back,
            details_combined=details_combined
        )
    
    except Exception as e:
        flash(f"Une erreur s'est produite lors du traitement des images: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/batch', methods=['GET', 'POST'])
def batch_upload():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('Aucun fichier trouvé', 'error')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            flash('Aucun fichier sélectionné', 'error')
            return redirect(request.url)
        
        # Récupérer les options d'analyse
        use_ml = 'use_ml' in request.form and request.form['use_ml'] == 'on' and ML_DISPONIBLE
        card_side = request.form.get('card_side', 'front')  # Face de la carte à analyser
        
        # Créer un dossier de lot avec un timestamp
        batch_id = f"batch_{int(time.time())}"
        batch_folder = os.path.join(app.config['RESULTS_FOLDER'], batch_id)
        os.makedirs(batch_folder, exist_ok=True)
        
        # Traiter chaque fichier
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                try:
                    if card_side == 'back':
                        # Analyse du dos de la carte
                        carte_redressee, resultats_analyse = scanner_carte_pokemon_ameliore(
                            file_path, afficher_etapes=False, sauvegarder_rapport=True)
                        
                        if carte_redressee is not None and resultats_analyse is not None:
                            note_globale = resultats_analyse["score_total"]
                            note_psa = resultats_analyse["note_psa"]
                            
                            results.append({
                                'filename': filename,
                                'success': True,
                                'note_globale': round(note_globale, 1),
                                'note_psa': note_psa,
                                'description_psa': obtenir_description_psa_dos(note_psa)
                            })
                        else:
                            results.append({
                                'filename': filename,
                                'success': False,
                                'error': "Impossible de détecter la carte"
                            })
                    else:
                        # Analyse du devant de la carte
                        if use_ml:
                            note_globale, note_psa, _, _ = scanner_et_analyser_carte_ml(
                                file_path, afficher_etapes=False, sauvegarder_rapport=True)
                        else:
                            carte_redressee = scanner_carte_pokemon(file_path, afficher_etapes=False)
                            if carte_redressee is not None:
                                note_globale, note_psa, _, _ = analyser_qualite_carte(carte_redressee, debug=False)
                            else:
                                note_globale, note_psa = None, None
                        
                        results.append({
                            'filename': filename,
                            'success': note_globale is not None,
                            'note_globale': round(note_globale, 1) if note_globale else None,
                            'note_psa': note_psa if note_psa else None,
                            'description_psa': obtenir_description_psa(note_psa) if note_psa else "Échec"
                        })
                    
                except Exception as e:
                    results.append({
                        'filename': filename,
                        'success': False,
                        'error': str(e)
                    })
        
        # Générer un rapport CSV des résultats
        csv_path = os.path.join(batch_folder, 'resultats.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Fichier,Succès,Note globale,Note PSA,Description\n")
            for r in results:
                f.write(f"{r['filename']},{r['success']},{r.get('note_globale', '')},{r.get('note_psa', '')},{r.get('description_psa', '')}\n")
        
        return render_template('batch_results.html', results=results, batch_id=batch_id)
    
    return render_template('batch.html', ml_available=ML_DISPONIBLE and app.config['USE_ML'])

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/batch_download/<batch_id>')
def batch_download(batch_id):
    return send_from_directory(os.path.join(app.config['RESULTS_FOLDER'], batch_id), 'resultats.csv')

# Ajoutez ces routes à votre app.py

@app.route('/add-to-cart/<int:card_id>', methods=['POST'])
@login_required
def add_to_cart(card_id):
    # Vérifier si la carte existe
    card = Card.query.get_or_404(card_id)

    #service de gradation
    grading_service = request.form.get('grading_service', 'no')

    
    # Vérifier si l'article est déjà dans le panier
    cart_item = CartItem.query.filter_by(
        user_id=current_user.id, 
        card_id=card_id, 
        grading_service=(grading_service == 'yes')
    ).first()

    if cart_item:
        # Si l'article est déjà dans le panier, augmenter la quantité
        cart_item.quantity += 1
        db.session.commit()
        flash(f"La quantité de '{card.name}' a été augmentée dans votre panier.", 'success')
    else:
        # Sinon, ajouter l'article au panier
        new_cart_item = CartItem(
            user_id=current_user.id, 
            card_id=card_id, 
            quantity=1,
            grading_service=(grading_service == 'yes')
        )
        db.session.add(new_cart_item)
        db.session.commit()
        
        service_msg = " avec option gradation" if grading_service == 'yes' else ""
        flash(f"'{card.name}'{service_msg} a été ajouté à votre panier.", 'success')
    
    # Rediriger selon la source de la demande
    if request.referrer and 'card-details' in request.referrer:
        return redirect(url_for('card_details', card_id=card_id))
    else:
        return redirect(url_for('marketplace'))

@app.route('/cart')
@login_required
def view_cart():
    # Récupérer les articles du panier de l'utilisateur
    cart_items = CartItem.query.filter_by(user_id=current_user.id).all()
    
    # Calculer le total du panier
    total = sum(item.card.price * item.quantity for item in cart_items)
    
    return render_template('cart.html', cart_items=cart_items, total=total)

@app.route('/update-cart/<int:item_id>', methods=['POST'])
@login_required
def update_cart(item_id):
    cart_item = CartItem.query.filter_by(id=item_id, user_id=current_user.id).first_or_404()
    
    action = request.form.get('action')
    
    if action == 'increase':
        cart_item.quantity += 1
    elif action == 'decrease':
        cart_item.quantity -= 1
        if cart_item.quantity <= 0:
            db.session.delete(cart_item)
    elif action == 'remove':
        db.session.delete(cart_item)
    
    db.session.commit()
    flash("Votre panier a été mis à jour.", 'success')
    return redirect(url_for('view_cart'))

@app.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    # Récupérer les articles du panier
    cart_items = CartItem.query.filter_by(user_id=current_user.id).all()
    
    if not cart_items:
        flash("Votre panier est vide.", 'error')
        return redirect(url_for('view_cart'))
    
    # Calculer le total
    total = sum(item.card.price * item.quantity for item in cart_items)
    
    if request.method == 'POST':
        # Récupérer les informations de livraison
        shipping_name = request.form.get('shipping_name')
        shipping_address = request.form.get('shipping_address')
        shipping_city = request.form.get('shipping_city')
        shipping_postal_code = request.form.get('shipping_postal_code')
        shipping_country = request.form.get('shipping_country')
        
        # Créer une chaîne complète d'adresse
        full_address = f"{shipping_name}\n{shipping_address}\n{shipping_city}, {shipping_postal_code}\n{shipping_country}"
        
        # Créer une nouvelle commande
        new_order = Order(
            user_id=current_user.id,
            status='pending',
            total_amount=total,
            shipping_address=full_address
        )
        
        db.session.add(new_order)
        db.session.flush()  # Pour obtenir l'ID de la commande
        
        # Ajouter les articles à la commande
        for item in cart_items:
            order_item = OrderItem(
                order_id=new_order.id,
                card_id=item.card_id,
                quantity=item.quantity,
                price_at_purchase=item.card.price
            )
            db.session.add(order_item)
            
            # Supprimer l'article du panier
            db.session.delete(item)
        
        db.session.commit()
        
        # Rediriger vers la confirmation de commande
        flash("Votre commande a été passée avec succès!", 'success')
        return redirect(url_for('order_confirmation', order_id=new_order.id))
    
    return render_template('checkout.html', cart_items=cart_items, total=total)

@app.route('/process_paypal_payment', methods=['POST'])
def process_paypal_payment():
    # Récupérer l'ID de commande PayPal
    paypal_order_id = request.form.get('paypal_order_id')
    
    # Autres détails d'adresse de livraison
    shipping_name = request.form.get('shipping_name')
    shipping_address = request.form.get('shipping_address')
    shipping_city = request.form.get('shipping_city')
    shipping_postal_code = request.form.get('shipping_postal_code')
    shipping_country = request.form.get('shipping_country')
    
    # Ici, vous pouvez vérifier l'ordre PayPal auprès de l'API PayPal
    # et enregistrer la commande dans votre base de données
    
    # Si tout est correct, redirigez vers une page de confirmation
    return jsonify({
        'success': True,
        'redirect_url': url_for('order_confirmation', order_id=new_order_id)
    })

@app.route('/orders')
@login_required
def order_history():
    # Récupérer l'historique des commandes de l'utilisateur
    orders = Order.query.filter_by(user_id=current_user.id).order_by(Order.order_date.desc()).all()
    return render_template('orders.html', orders=orders)

@app.route('/order/<int:order_id>')
@login_required
def order_details(order_id):
    # Récupérer les détails d'une commande spécifique
    order = Order.query.filter_by(id=order_id, user_id=current_user.id).first_or_404()
    return render_template('order_details.html', order=order)

@app.route('/order-confirmation/<int:order_id>')
@login_required
def order_confirmation(order_id):
    order = Order.query.filter_by(id=order_id, user_id=current_user.id).first_or_404()
    return render_template('order_confirmation.html', order=order)

#route pour le profile
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Si l'utilisateur est déjà connecté, rediriger vers la marketplace
    if current_user.is_authenticated:
        return redirect(url_for('marketplace'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Vérifier si le nom d'utilisateur existe déjà
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Ce nom d\'utilisateur est déjà pris.', 'error')
            return redirect(url_for('register'))
        
        # Vérifier si l'email existe déjà
        email_exists = User.query.filter_by(email=email).first()
        if email_exists:
            flash('Cette adresse email est déjà utilisée.', 'error')
            return redirect(url_for('register'))
        
        # Vérifier que les mots de passe correspondent
        if password != confirm_password:
            flash('Les mots de passe ne correspondent pas.', 'error')
            return redirect(url_for('register'))
        
        # Créer un nouvel utilisateur
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        # Ajouter à la base de données
        db.session.add(new_user)
        db.session.commit()
        
        flash('Votre compte a été créé avec succès! Vous pouvez maintenant vous connecter.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Si l'utilisateur est déjà connecté, rediriger vers la marketplace
    if current_user.is_authenticated:
        return redirect(url_for('marketplace'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        # Chercher l'utilisateur dans la base de données
        user = User.query.filter_by(username=username).first()
        
        # Vérifier si l'utilisateur existe et si le mot de passe est correct
        if not user or not user.check_password(password):
            flash('Nom d\'utilisateur ou mot de passe incorrect.', 'error')
            return redirect(url_for('login'))
        
        # Connecter l'utilisateur
        login_user(user, remember=remember)
        
        # Rediriger vers la page demandée initialement ou la marketplace
        next_page = request.args.get('next')
        return redirect(next_page or url_for('marketplace'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

# Modifiez votre route marketplace pour exiger une connexion
# ======= ROUTES POUR LA MARKETPLACE =======

@app.route('/marketplace')
@login_required
def marketplace():
    # Récupérer toutes les cartes de la base de données
    cards = Card.query.all()
    return render_template('market.html', cards=cards)


@app.route('/generate-auth-code/<int:card_id>')
@login_required
def generate_auth_code(card_id):
    """Génère un QR code d'authenticité pour une carte Pokémon"""
    card = Card.query.get_or_404(card_id)
    
    # Vérifier que l'utilisateur a le droit de générer un code (admin ou propriétaire)
    if not current_user.is_admin:
        flash("Vous n'avez pas l'autorisation de générer un code d'authenticité.", "error")
        return redirect(url_for('card_details', card_id=card_id))
    
    # Générer l'URL de vérification avec le domaine actuel
    verification_url = request.host_url.rstrip('/') + url_for('verify_card', card_id=card_id, _external=False)
    
    # Générer le QR code
    qr_io = generate_card_qr_code(card_id, verification_url)
    
    # Retourner l'image du QR code
    return send_file(qr_io, mimetype='image/png', 
                    as_attachment=True, 
                    download_name=f'auth_qr_pokia_{card_id}.png')

@app.route('/card-auth-label/<int:card_id>')
@login_required
def card_auth_label(card_id):
    """Page pour générer une étiquette avec QR code d'authenticité"""
    card = Card.query.get_or_404(card_id)
    
    # Vérifier que l'utilisateur a le droit de générer un code (admin ou propriétaire)
    if not current_user.is_admin:
        flash("Vous n'avez pas l'autorisation de générer une étiquette d'authenticité.", "error")
        return redirect(url_for('card_details', card_id=card_id))
    
    # Obtenir la date actuelle pour l'affichage
    now = dt.utcnow()
    
    # Générer l'URL de vérification avec le domaine actuel
    verification_url = request.host_url.rstrip('/') + url_for('verify_card', card_id=card.id, _external=False)
    
    return render_template('auth_label.html', card=card, verification_url=verification_url, now=now)

@app.route('/verify-card/<int:card_id>')
def verify_card(card_id):
    """Page de vérification d'authenticité d'une carte Pokémon"""
    card = Card.query.get_or_404(card_id)
    
    # Obtenir la date actuelle pour l'affichage
    now = dt.utcnow()
    
    # Ici, vous pouvez ajouter plus de logique pour vérifier l'authenticité
    # Par exemple, vérifier si la carte a été gradée par Pokia
    
    return render_template('verify_card.html', card=card, now=now)

@app.route('/add-card', methods=['GET', 'POST'])
def add_card():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        card_name = request.form.get('card_name')
        card_type = request.form.get('card_type')
        card_rarity = request.form.get('card_rarity')
        psa_grade = request.form.get('psa_grade')
        card_price = float(request.form.get('card_price'))
        old_price = request.form.get('old_price')
        card_description = request.form.get('card_description')
        
        # Convertir old_price en float s'il n'est pas vide
        if old_price:
            old_price = float(old_price)
        
        # Gérer l'upload de l'image
        if 'card_image' not in request.files:
            flash('Aucun fichier sélectionné', 'error')
            return redirect(request.url)
            
        file = request.files['card_image']
        
        if file.filename == '':
            flash('Aucun fichier sélectionné', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Générer un nom de fichier unique avec timestamp
            timestamp = dt.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{filename}"
            
            # Assurez-vous que le dossier existe
            cards_folder = 'static/images/cards'
            os.makedirs(cards_folder, exist_ok=True)
            
            file_path = os.path.join(cards_folder, filename)
            file.save(file_path)
            
            # Créer une nouvelle carte dans la base de données
            new_card = Card(
                name=card_name,
                type=card_type,
                rarity=card_rarity,
                psa_grade=psa_grade,
                price=card_price,
                old_price=old_price,
                description=card_description,
                image_path=f"images/cards/{filename}"
            )
            
            # Ajouter et commiter dans la base de données
            db.session.add(new_card)
            db.session.commit()
            
            flash('Carte ajoutée avec succès!', 'success')
            return redirect(url_for('marketplace'))
        else:
            flash('Type de fichier non autorisé', 'error')
            return redirect(request.url)
    
    # Méthode GET - Afficher le formulaire
    return render_template('add_card.html')

@app.route('/edit-card/<int:card_id>', methods=['GET', 'POST'])
def edit_card(card_id):
    # Récupérer la carte à modifier
    card = Card.query.get_or_404(card_id)
    
    if request.method == 'POST':
        # Mettre à jour les informations de la carte
        card.name = request.form.get('card_name')
        card.type = request.form.get('card_type')
        card.rarity = request.form.get('card_rarity')
        card.psa_grade = request.form.get('psa_grade')
        card.price = float(request.form.get('card_price'))
        
        # Gérer l'ancien prix (optionnel)
        old_price = request.form.get('old_price')
        card.old_price = float(old_price) if old_price else None
        
        card.description = request.form.get('card_description')
        
        # Gérer l'upload d'une nouvelle image (si fournie)
        if 'card_image' in request.files and request.files['card_image'].filename != '':
            file = request.files['card_image']
            
            if allowed_file(file.filename):
                # Supprimer l'ancienne image si elle existe
                try:
                    old_image_path = os.path.join('static', card.image_path)
                    if os.path.exists(old_image_path):
                        os.remove(old_image_path)
                except Exception as e:
                    print(f"Erreur lors de la suppression de l'ancienne image: {e}")
                
                # Enregistrer la nouvelle image
                filename = secure_filename(file.filename)
                timestamp = dt.now().strftime("%Y%m%d%H%M%S")
                filename = f"{timestamp}_{filename}"
                
                file_path = os.path.join('static/images/cards', filename)
                file.save(file_path)
                
                # Mettre à jour le chemin d'image
                card.image_path = f"images/cards/{filename}"
        
        # Enregistrer les modifications
        db.session.commit()
        
        flash('Carte mise à jour avec succès!', 'success')
        return redirect(url_for('marketplace'))
        
    # Méthode GET - Afficher le formulaire pré-rempli
    return render_template('edit_card.html', card=card)

@app.route('/delete-card/<int:card_id>', methods=['POST'])
def delete_card(card_id):
    # Récupérer la carte à supprimer
    card = Card.query.get_or_404(card_id)
    
    # Supprimer l'image associée
    try:
        image_path = os.path.join('static', card.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Image supprimée: {image_path}")  # Log pour débug

    except Exception as e:
        print(f"Erreur lors de la suppression de l'image: {e}")
    
    # Supprimer l'entrée de la base de données
    db.session.delete(card)
    db.session.commit()
    
    print(f"Carte {card_id} supprimée avec succès")  # Log pour débug

    
    flash('Carte supprimée avec succès!', 'success')
    return redirect(url_for('marketplace'))

@app.route('/card-details/<int:card_id>')
def card_details(card_id):
    card = Card.query.get_or_404(card_id)
    
    # Créer une liste des IDs de cartes en favoris si l'utilisateur est connecté
    user_favorites = []
    if current_user.is_authenticated:
        user_favorites = [fav.card_id for fav in Favorite.query.filter_by(user_id=current_user.id).all()]
    
    return render_template('card_details.html', card=card, user_favorites=user_favorites)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help_page():
    return render_template('help.html')


# Enregistrer les routes du scanner caméra POKIA
register_scanner_routes(app)

# Enregistrer les routes de calibration
register_calibration_routes(app)

# Point d'entrée de l'application
if __name__ == '__main__':
    # Créer les tables si elles n'existent pas
    with app.app_context():
        db.create_all()   
    app.run(host='0.0.0.0', port=5001, debug=True)
