import pymysql

try:
    # Connexion à MySQL
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',  # Ajoutez votre mot de passe si nécessaire
        database='pokemon_marketplace'
    )
    
    print("Connexion à MySQL réussie!")
    
    # Créer un curseur
    cursor = connection.cursor()
    
    # Fonction pour vérifier si un index existe déjà et le créer s'il n'existe pas
    def create_index_if_not_exists(cursor, table_name, index_name, column_name):
        cursor.execute(f"SHOW INDEX FROM {table_name} WHERE Key_name = '{index_name}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE INDEX {index_name} ON {table_name}({column_name})")
            print(f"Index '{index_name}' créé avec succès.")
        else:
            print(f"L'index '{index_name}' existe déjà.")
    
    # Créer les tables d'abord
    
    # 1. Table pour les cartes Pokémon
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        type VARCHAR(50) NOT NULL,
        rarity VARCHAR(50) NOT NULL,
        psa_grade VARCHAR(20),
        price DECIMAL(10, 2) NOT NULL,
        old_price DECIMAL(10, 2),
        description TEXT,
        image_path VARCHAR(255) NOT NULL,
        date_added DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    print("Table 'cards' créée avec succès.")
    
    # 2. Table pour les utilisateurs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) NOT NULL UNIQUE,
        email VARCHAR(100) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        is_admin BOOLEAN DEFAULT FALSE,
        date_registered DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    print("Table 'users' créée avec succès.")
    
    # 3. Table pour les commandes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        total_amount DECIMAL(10, 2) NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        shipping_address TEXT NOT NULL,
        order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    print("Table 'orders' créée avec succès.")
    
    # 4. Table pour les détails de commande
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        id INT AUTO_INCREMENT PRIMARY KEY,
        order_id INT NOT NULL,
        card_id INT NOT NULL,
        quantity INT NOT NULL DEFAULT 1,
        price_at_purchase DECIMAL(10, 2) NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (card_id) REFERENCES cards(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    print("Table 'order_items' créée avec succès.")
    
    # 5. Table pour les évaluations/avis des cartes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        id INT AUTO_INCREMENT PRIMARY KEY,
        card_id INT NOT NULL,
        user_id INT NOT NULL,
        rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
        comment TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(id),
        FOREIGN KEY (user_id) REFERENCES users(id),
        UNIQUE KEY unique_review (card_id, user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    print("Table 'reviews' créée avec succès.")
    
    # 6. Table pour les favoris
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS favorites (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        card_id INT NOT NULL,
        date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (card_id) REFERENCES cards(id),
        UNIQUE KEY unique_favorite (user_id, card_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    print("Table 'favorites' créée avec succès.")
    
    # Maintenant, créer les index après que toutes les tables existent
    
    # Index pour la table cards
    create_index_if_not_exists(cursor, "cards", "idx_cards_type", "type")
    create_index_if_not_exists(cursor, "cards", "idx_cards_rarity", "rarity")
    create_index_if_not_exists(cursor, "cards", "idx_cards_price", "price")
    create_index_if_not_exists(cursor, "cards", "idx_cards_psa_grade", "psa_grade")
    print("Vérification des index pour la table 'cards' terminée.")
    
    # Index pour la table orders
    create_index_if_not_exists(cursor, "orders", "idx_orders_user_id", "user_id")
    create_index_if_not_exists(cursor, "orders", "idx_orders_status", "status")
    print("Vérification des index pour la table 'orders' terminée.")
    
    # Index pour la table order_items
    create_index_if_not_exists(cursor, "order_items", "idx_order_items_order_id", "order_id")
    create_index_if_not_exists(cursor, "order_items", "idx_order_items_card_id", "card_id")
    print("Vérification des index pour la table 'order_items' terminée.")
    
    # Index pour la table reviews
    create_index_if_not_exists(cursor, "reviews", "idx_reviews_card_id", "card_id")
    create_index_if_not_exists(cursor, "reviews", "idx_reviews_user_id", "user_id")
    create_index_if_not_exists(cursor, "reviews", "idx_reviews_rating", "rating")
    print("Vérification des index pour la table 'reviews' terminée.")
    
    # Index pour la table favorites
    create_index_if_not_exists(cursor, "favorites", "idx_favorites_user_id", "user_id")
    create_index_if_not_exists(cursor, "favorites", "idx_favorites_card_id", "card_id")
    print("Vérification des index pour la table 'favorites' terminée.")
    
    # Vérifier si des cartes existent déjà
    cursor.execute("SELECT COUNT(*) FROM cards")
    card_count = cursor.fetchone()[0]
    
    # N'insérer des données d'exemple que si la table est vide
    if card_count == 0:
        # Insérer des données d'exemple dans la table cards
        cursor.execute("""
        INSERT INTO cards (name, type, rarity, psa_grade, price, old_price, description, image_path, date_added) VALUES
        ('Dracaufeu GX Hyper Rare', 'Feu', 'Hyper Rare', 'PSA 9', 399.99, 499.99, 'Carte Dracaufeu GX Hyper Rare en excellent état avec note PSA 9.', 'images/cards/default_card1.jpg', NOW()),
        ('Mewtwo GX Secret Rare', 'Psy', 'Secrète', 'PSA 10', 599.99, 699.99, 'Carte Mewtwo GX Secret Rare parfaite avec note PSA 10.', 'images/cards/default_card2.jpg', NOW()),
        ('Pikachu VMAX Rainbow', 'Électrique', 'Rainbow', NULL, 289.99, NULL, 'Magnifique carte Pikachu VMAX Rainbow en très bon état.', 'images/cards/default_card3.jpg', NOW()),
        ('Ectoplasma GX Shiny', 'Spectre', 'Shiny', 'Rare', 349.99, NULL, 'Carte Ectoplasma GX Shiny avec détails holographiques impressionnants.', 'images/cards/default_card4.jpg', NOW())
        """)
        print("Données d'exemple insérées dans la table 'cards'.")
    else:
        print(f"La table 'cards' contient déjà {card_count} enregistrements, aucune donnée d'exemple n'a été insérée.")
    
    # Valider les changements
    connection.commit()
    
    # Vérifier les tables créées
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    print("\nTables dans la base de données 'pokemon_marketplace':")
    for table in tables:
        print(f"- {table[0]}")
    
    # Fermer la connexion
    connection.close()
    print("\nToutes les opérations ont été effectuées avec succès!")
    
except Exception as e:
    print(f"Erreur: {e}")