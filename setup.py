import os
import shutil
import sys

def create_directory_structure():
    """Crée la structure de dossiers pour l'application"""
    directories = [
        'static',
        'static/css',
        'static/images',
        'templates',
        'uploads',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Dossier créé: {directory}")

def create_placeholder_images():
    """Crée des fichiers image de base pour les exemples"""
    # Ce code est un placeholder - dans une vraie application, vous devriez 
    # télécharger ou préparer de vraies images
    placeholder_files = [
        ('static/images/pokemon-logo.png', 'Créez ou téléchargez un logo Pokémon'),
        ('static/images/favicon.ico', 'Créez ou téléchargez une favicon'),
        ('static/images/pokeball-pattern.png', 'Créez ou téléchargez un motif de Pokéball'),
        ('static/images/pokeball-bullet.png', 'Créez ou téléchargez une icône de Pokéball'),
        ('static/images/icon-card.png', 'Créez ou téléchargez une icône de carte'),
        ('static/images/icon-analyse.png', 'Créez ou téléchargez une icône d\'analyse'),
        ('static/images/icon-grade.png', 'Créez ou téléchargez une icône de note'),
        ('static/images/good-photo.jpg', 'Créez ou téléchargez un exemple de bonne photo'),
        ('static/images/bad-photo.jpg', 'Créez ou téléchargez un exemple de mauvaise photo')
    ]
    
    for file_path, message in placeholder_files:
        # Juste créer un fichier vide pour l'instant
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('')
            print(f"✓ Fichier placeholder créé: {file_path} - {message}")

def install_requirements():
    """Installe les dépendances requises"""
    requirements = [
        'flask',
        'opencv-python',
        'numpy',
    ]
    
    try:
        import pip
        for package in requirements:
            print(f"Installation de {package}...")
            pip.main(['install', package])
        print("✓ Dépendances installées avec succès")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'installation des dépendances: {e}")
        print("Veuillez installer manuellement les packages suivants: flask, opencv-python, numpy")

def main():
    """Fonction principale de configuration"""
    print("=== Configuration de l'Analyseur de Cartes Pokémon ===")
    
    create_directory_structure()
    create_placeholder_images()
    install_requirements()
    
    print("\n✅ Configuration terminée!")
    print("\nPour lancer l'application, exécutez:")
    print("python app.py")

if __name__ == "__main__":
    main()