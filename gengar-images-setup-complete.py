import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter

# Créer les dossiers nécessaires
os.makedirs('static/images', exist_ok=True)

# URLs de quelques images Ectoplasma que vous pourriez utiliser
# Remarque: Assurez-vous d'utiliser des images libres de droits ou de respecter les droits d'auteur
GENGAR_ICON_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/94.png"
PLACEHOLDER_URLS = {
    "gengar-icon.png": GENGAR_ICON_URL,
}

def download_image(url, file_path):
    """Télécharger une image depuis l'URL et la sauvegarder"""
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(file_path)
        print(f"✓ Image téléchargée: {file_path}")
    except Exception as e:
        print(f"⚠️ Erreur lors du téléchargement de {url}: {str(e)}")

def create_gengar_logo():
    """Créer un logo Pokémon avec couleurs Ectoplasma"""
    # Taille de l'image
    width, height = 300, 100
    
    # Créer une image avec un fond transparent
    logo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)
    
    # Dessiner le texte "Pokémon" avec dégradé violet
    # Ceci est un placeholder - dans une vraie application, vous auriez besoin d'une police personnalisée
    draw.rectangle([0, 0, width, height], fill=(74, 43, 107, 180), outline=(123, 98, 163), width=2)
    draw.text((width//2-60, height//2-10), "POKEMON", fill=(255, 255, 255))
    
    # Sauvegarder l'image
    logo.save('static/images/pokemon-logo.png')
    print("✓ Logo Pokémon créé")

def create_gengar_pattern():
    """Créer un motif de fond avec des silhouettes d'Ectoplasma"""
    # Taille de l'image pour le pattern
    width, height = 300, 300
    
    # Créer une image avec un fond transparent
    pattern = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pattern)
    
    # Dessiner plusieurs silhouettes d'Ectoplasma
    for i in range(5):
        for j in range(5):
            x = i * 60 + (j % 2) * 30
            y = j * 60
            
            # Dessiner une silhouette simple d'Ectoplasma
            r = 10  # rayon
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(123, 98, 163, 30))
    
    # Appliquer un flou pour adoucir le pattern
    pattern = pattern.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Sauvegarder l'image
    pattern.save('static/images/gengar-pattern.png')
    print("✓ Motif Ectoplasma créé")

def create_gengar_silhouette():
    """Créer une silhouette d'Ectoplasma"""
    # Taille de l'image
    width, height = 100, 100
    
    # Créer une image avec un fond transparent
    silhouette = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(silhouette)
    
    # Dessiner une silhouette simple d'Ectoplasma
    # Corps
    draw.ellipse([20, 30, 80, 90], fill=(74, 43, 107, 200))
    # Tête
    draw.ellipse([10, 10, 90, 70], fill=(74, 43, 107, 200))
    # Yeux
    draw.ellipse([30, 30, 40, 40], fill=(255, 56, 96, 200))
    draw.ellipse([60, 30, 70, 40], fill=(255, 56, 96, 200))
    # Sourire
    draw.arc([30, 40, 70, 60], 0, 180, fill=(255, 56, 96, 200), width=2)
    
    # Sauvegarder l'image
    silhouette.save('static/images/gengar-silhouette.png')
    print("✓ Silhouette Ectoplasma créée")

def create_gengar_bullet():
    """Créer une petite icône d'Ectoplasma pour les puces de liste"""
    # Taille de l'image
    width, height = 20, 20
    
    # Créer une image avec un fond transparent
    bullet = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bullet)
    
    # Dessiner une petite silhouette d'Ectoplasma
    draw.ellipse([5, 5, 15, 15], fill=(123, 98, 163, 255))
    draw.ellipse([7, 7, 9, 9], fill=(255, 56, 96, 255))
    draw.ellipse([11, 7, 13, 9], fill=(255, 56, 96, 255))
    
    # Sauvegarder l'image
    bullet.save('static/images/gengar-bullet.png')
    print("✓ Puce Ectoplasma créée")

def create_analysis_icons():
    """Créer des icônes pour les fonctionnalités d'analyse"""
    # Liste des icônes à créer
    icons = [
        ("icon-card.png", (160, 64, 160)),  # Violet pour carte
        ("icon-analyse.png", (255, 56, 96)),  # Rouge pour analyse
        ("icon-grade.png", (123, 98, 163))  # Violet clair pour notation
    ]
    
    for name, color in icons:
        # Taille de l'image
        width, height = 60, 60
        
        # Créer une image avec un fond transparent
        icon = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        
        # Dessiner un cercle avec la couleur spécifiée
        draw.ellipse([5, 5, 55, 55], fill=color + (180,))
        
        # Ajouter un effet brillant
        draw.ellipse([15, 15, 25, 25], fill=(255, 255, 255, 100))
        
        # Sauvegarder l'image
        icon.save(f'static/images/{name}')
        print(f"✓ Icône {name} créée")

def create_placeholder_photos():
    """Créer des exemples de photos pour la page d'aide"""
    # Liste des photos à créer
    photos = [
        ("good-photo.jpg", (100, 200, 100)),  # Vert pour bonne photo
        ("bad-photo.jpg", (200, 100, 100))  # Rouge pour mauvaise photo
    ]
    
    for name, color in photos:
        # Taille de l'image
        width, height = 200, 300
        
        # Créer une image avec un fond coloré
        photo = Image.new('RGB', (width, height), color)
        draw = ImageDraw.Draw(photo)
        
        # Ajouter un rectangle pour simuler une carte
        draw.rectangle([50, 100, 150, 200], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        
        # Sauvegarder l'image
        photo.save(f'static/images/{name}')
        print(f"✓ Photo exemple {name} créée")

def create_favicon():
    """Créer une favicon simple"""
    # Taille de l'image
    width, height = 32, 32
    
    # Créer une image avec un fond transparent
    favicon = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(favicon)
    
    # Dessiner une petite silhouette d'Ectoplasma
    draw.ellipse([0, 0, 32, 32], fill=(74, 43, 107, 255))
    draw.ellipse([8, 8, 14, 14], fill=(255, 56, 96, 255))
    draw.ellipse([18, 8, 24, 14], fill=(255, 56, 96, 255))
    draw.arc([8, 14, 24, 24], 0, 180, fill=(255, 56, 96, 255), width=2)
    
    # Sauvegarder l'image
    favicon.save('static/images/favicon.ico')
    print("✓ Favicon créée")

def main():
    """Fonction principale"""
    print("=== Création des images pour le thème Ectoplasma ===")
    
    # Télécharger l'image de l'API Pokémon
    for name, url in PLACEHOLDER_URLS.items():
        download_image(url, f'static/images/{name}')
    
    # Créer nos propres images
    create_gengar_logo()
    create_gengar_pattern()
    create_gengar_silhouette()
    create_gengar_bullet()
    create_analysis_icons()
    create_placeholder_photos()
    create_favicon()
    
    print("\n✅ Création d'images terminée!")

if __name__ == "__main__":
    main()