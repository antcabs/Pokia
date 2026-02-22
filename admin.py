from app import app, db, User

# Utilisez un context d'application Flask
with app.app_context():
    # Remplacez "votre_nom_utilisateur" par votre nom d'utilisateur réel
    user = User.query.filter_by(username="antcabs").first()
    if user:
        print(f"Utilisateur trouvé: {user.username}")
        print(f"Statut admin actuel: {user.is_admin}")
        
        # Promouvoir l'utilisateur en administrateur
        user.is_admin = True
        db.session.commit()
        
        print(f"Nouveau statut admin: {user.is_admin}")
        print("L'utilisateur est maintenant administrateur!")
    else:
        print("Utilisateur non trouvé.")