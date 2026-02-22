# Intégration de la Détection et Gradation de Cartes Pokémon

## Résumé des Modifications

Ce document décrit l'intégration complète de la détection de cartes avec gradation ML dans le système de scanner POKIA.

## Fichiers Modifiés

### 1. `scanner_routes.py`

**Modifications apportées:**
- Ajout de l'import de `scanner_et_analyser_carte_ml`, `obtenir_description_psa` et `ML_DISPONIBLE` depuis le module `pokia`
- Ajout de l'import de `base64` pour encoder les images
- Nouvelle route `/analyze_capture` qui:
  - Prend le chemin d'une image capturée en POST (JSON)
  - Appelle `scanner_et_analyser_carte_ml()` pour analyser la carte avec le modèle ML
  - Retourne les résultats en JSON incluant:
    - Note globale (sur 80)
    - Grade PSA (1-10)
    - Description du grade PSA
    - Détails (coins, bords, centrage)
    - Image annotée en base64

**Code de la nouvelle route:**
```python
@app.route('/analyze_capture', methods=['POST'])
def analyze_capture():
    """Analyse une image capturée avec la gradation ML."""
    data = request.get_json()
    chemin_image = data.get('path')
    
    # Validation et analyse...
    resultats = scanner_et_analyser_carte_ml(chemin_image, ...)
    
    # Retourne les résultats en JSON
```

### 2. `templates/scanner.html`

**Modifications apportées:**

#### A. CSS
- Ajout de styles pour la section de résultats (`.results-section`)
- Styles pour l'affichage du grade PSA (`.grade-display`, `.grade-psa`)
- Grille de détails (`.details-grid`, `.detail-item`)
- Animation de slide-in pour l'apparition des résultats

#### B. HTML
- Nouvelle section `<div class="results-section">` qui affiche:
  - Grade PSA en grand avec description
  - Note globale sur 80 points
  - Détails des scores (coins, bords, centrage)
  - Image annotée avec les zones d'analyse

#### C. JavaScript
- Modification de la fonction `captureImage()`:
  - **Étape 1:** Capture l'image via `/capture`
  - **Étape 2:** Envoie le chemin vers `/analyze_capture` pour l'analyse ML
  - **Étape 3:** Affiche les résultats dans la nouvelle section
  - Gestion d'erreurs améliorée
  - Scroll automatique vers les résultats

## Flux de Fonctionnement

```
┌─────────────────┐
│  Utilisateur    │
│  clique sur     │
│  "Capturer et   │
│   Analyser"     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  POST /capture  │
│  (Capture la    │
│   photo)        │
└────────┬────────┘
         │
         ▼ retourne chemin
┌──────────────────────┐
│ POST /analyze_capture│
│ - Détecte la carte   │
│ - Redresse l'image   │
│ - Analyse avec ML    │
│   (coins, bords,     │
│    centrage)         │
│ - Calcule grade PSA  │
└────────┬─────────────┘
         │
         ▼ retourne JSON
┌──────────────────────┐
│  Affichage dans UI   │
│  - Grade PSA         │
│  - Note /80          │
│  - Détails           │
│  - Image annotée     │
└──────────────────────┘
```

## Utilisation

1. **Démarrer l'application:**
   ```bash
   python app.py
   ```

2. **Accéder au scanner:**
   - Ouvrir http://localhost:5000/scanner
   - La caméra démarre automatiquement
   - La détection ML en temps réel affiche:
     - Contour de la carte (violet)
     - Zones des coins C1-C4 (vert)
     - Zones des bords (orange)
     - Lignes de centrage (magenta)

3. **Capturer et analyser:**
   - Placer la carte dans le cadre
   - Aligner selon les guides
   - Cliquer sur "📸 Capturer et Analyser"
   - Attendre l'analyse (quelques secondes)
   - Les résultats apparaissent en dessous:
     - Grade PSA (ex: PSA 8)
     - Description (ex: "Near Mint-Mint")
     - Note totale (ex: 72.5 / 80 points)
     - Détails par catégorie

4. **Options d'affichage:**
   - Cocher/décocher les options pour voir:
     - Zones de coins
     - Zones de bords  
     - Lignes de centrage

## Dépendances

Le système nécessite:
- Flask
- OpenCV (cv2)
- NumPy
- PyTorch (pour le modèle ML)
- Picamera2 (pour Raspberry Pi)
- Le modèle entraîné: `modele_carte_pokemon.pth`

## Modules Utilisés

### Du fichier `pokia.py`:
- `scanner_et_analyser_carte_ml()` - Fonction principale qui:
  - Scanne la carte (détection de contour)
  - Redresse l'image
  - Analyse avec le modèle ML
  - Retourne: note_globale, note_psa, résultats, img_annotée

- `obtenir_description_psa()` - Convertit un grade PSA (1-10) en description textuelle
  - PSA 10: "Gem Mint"
  - PSA 9: "Mint"
  - PSA 8: "Near Mint-Mint"
  - etc.

### Du fichier `model_pokemon.py`:
- `CardGraderModel` - Réseau de neurones (ResNet18) entraîné pour:
  - Évaluer les coins (sur 30)
  - Évaluer les bords (sur 30)
  - Évaluer le centrage (sur 20)
  - Calculer un grade PSA (1-10)

- `analyze_card()` - Fonction qui prépare l'image et fait l'inférence

### Du fichier `scanner_rpi.py`:
- `get_scanner()` - Retourne l'instance du scanner hardware
- `PokiaScanner.capturer_image()` - Capture une photo HD et la sauvegarde

## Tests

Pour tester l'intégration:

1. **Test manuel:**
   ```bash
   # Démarrer l'application
   python app.py
   
   # Ouvrir dans le navigateur
   firefox http://localhost:5000/scanner
   
   # Placer une carte et capturer
   ```

2. **Test de la route d'analyse:**
   ```bash
   # Capturer d'abord une image
   curl -X POST http://localhost:5000/capture
   
   # Analyser l'image capturée
   curl -X POST http://localhost:5000/analyze_capture \
     -H "Content-Type: application/json" \
     -d '{"path": "captures/capture_1234567890.jpg"}'
   ```

3. **Vérifications:**
   - ✅ La caméra affiche le flux vidéo
   - ✅ Les overlays de détection apparaissent
   - ✅ La capture fonctionne
   - ✅ L'analyse retourne des résultats valides
   - ✅ Les résultats s'affichent correctement

## Exemple de Résultat JSON

```json
{
  "success": true,
  "note_globale": 72.5,
  "note_psa": 8,
  "description_psa": "Near Mint-Mint",
  "details": {
    "coins": 25.2,
    "bords": 26.8,
    "centrage": 18.5
  },
  "image_annotee": "base64_encoded_image...",
  "message": "Analyse terminée avec succès"
}
```

## Troubleshooting

### Problème: "Module model_pokemon non trouvé"
**Solution:** Vérifier que `model_pokemon.py` est dans le même dossier

### Problème: "Modèle non disponible"
**Solution:** Vérifier que `modele_carte_pokemon.pth` existe et est accessible

### Problème: "Impossible d'analyser la carte"
**Causes possibles:**
- Carte pas détectée (mauvais éclairage, fond inadapté)
- Image floue
- Carte trop petite ou trop grande dans le cadre

**Solutions:**
- Améliorer l'éclairage
- Utiliser un fond blanc uni
- Ajuster la position de la carte selon les guides

### Problème: Caméra non disponible
**Solution sur Raspberry Pi:**
```bash
# Vérifier que picamera2 est installé
pip install picamera2

# Vérifier les permissions
sudo usermod -a -G video $USER
```

## Améliorations Futures

- [ ] Ajouter l'analyse du dos de la carte
- [ ] Historique des scans
- [ ] Export PDF des résultats
- [ ] Comparaison avant/après
- [ ] Détection automatique du trigger (sans clic)
- [ ] Support de plusieurs cartes simultanées
- [ ] Base de données des cartes scannées

## Auteur

Intégration réalisée pour le projet POKIA - Scanner de cartes Pokémon avec gradation ML
