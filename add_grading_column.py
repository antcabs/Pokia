from app import app, db
import sqlalchemy as sa
from sqlalchemy import inspect
import datetime

class Favorite(db.Model):
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    card_id = db.Column(db.Integer, db.ForeignKey('cards.id'), nullable=False)
    date_added = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Relations
    user = db.relationship('User', backref=db.backref('favorites', lazy=True))
    card = db.relationship('Card', backref=db.backref('favorited_by', lazy=True))
    
    # Contrainte unique pour éviter les doublons
    __table_args__ = (
        db.UniqueConstraint('user_id', 'card_id', name='uq_user_card_favorite'),
    )
    
    def __repr__(self):
        return f'<Favorite {self.user_id}:{self.card_id}>'

def create_favorites_table():
    with app.app_context():
        inspector = inspect(db.engine)
        if 'favorites' not in inspector.get_table_names():
            db.create_all()  # Crée la table si elle n'existe pas
            print("Table 'favorites' créée avec succès.")
        else:
            print("La table 'favorites' existe déjà.")

if __name__ == "__main__":
    create_favorites_table()
