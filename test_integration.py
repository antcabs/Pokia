#!/usr/bin/env python3
"""
Test de l'intégration de la détection et gradation de cartes POKIA
Ce script teste les différents composants de l'intégration.
"""

import os
import sys
import time
import requests
import json

# Configuration
BASE_URL = "http://localhost:5000"
TEST_IMAGE_PATH = "captures/test_card.jpg"  # À remplacer par votre chemin

def print_section(title):
    """Affiche un titre de section."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def test_1_scanner_status():
    """Test 1: Vérifier le statut du scanner."""
    print_section("Test 1: Statut du scanner")
    
    try:
        response = requests.get(f"{BASE_URL}/scanner/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Scanner accessible")
            print(f"   - Caméra disponible: {data.get('camera_available', False)}")
            print(f"   - LED disponible: {data.get('led_available', False)}")
            print(f"   - Initialisé: {data.get('initialized', False)}")
            return True
        else:
            print(f"❌ Erreur: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        print("   Assurez-vous que l'application Flask est en cours d'exécution.")
        return False

def test_2_capture_image():
    """Test 2: Capturer une image."""
    print_section("Test 2: Capture d'image")
    
    try:
        print("⏳ Envoi de la requête de capture...")
        response = requests.post(f"{BASE_URL}/capture")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Capture réussie")
                path = data.get('path')
                print(f"   - Chemin: {path}")
                
                # Vérifier que le fichier existe
                if os.path.exists(path):
                    print(f"   - Fichier trouvé: {os.path.getsize(path)} bytes")
                    return path
                else:
                    print(f"   ⚠️  Fichier non trouvé: {path}")
                    return None
            else:
                print(f"❌ Capture échouée: {data.get('message')}")
                return None
        else:
            print(f"❌ Erreur: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def test_3_analyze_capture(image_path):
    """Test 3: Analyser l'image capturée."""
    print_section("Test 3: Analyse de la carte")
    
    if not image_path:
        print("❌ Pas de chemin d'image fourni")
        return False
    
    try:
        print(f"⏳ Analyse de l'image: {image_path}")
        print("   (Cela peut prendre quelques secondes...)")
        
        payload = {"path": image_path}
        response = requests.post(
            f"{BASE_URL}/analyze_capture",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30  # 30 secondes de timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print("✅ Analyse réussie\n")
                print("📊 RÉSULTATS:")
                print(f"   Grade PSA: {data.get('note_psa')}")
                print(f"   Description: {data.get('description_psa')}")
                print(f"   Note globale: {data.get('note_globale')} / 80 points\n")
                
                details = data.get('details', {})
                print("   Détails:")
                print(f"   - Coins: {details.get('coins')} / 30")
                print(f"   - Bords: {details.get('bords')} / 30")
                print(f"   - Centrage: {details.get('centrage')} / 20")
                
                if data.get('image_annotee'):
                    print("\n   ✅ Image annotée disponible (base64)")
                
                return True
            else:
                print(f"❌ Analyse échouée: {data.get('message')}")
                return False
        else:
            print(f"❌ Erreur: Status code {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Message: {error_data.get('message')}")
            except:
                pass
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Timeout: L'analyse prend trop de temps")
        print("   Vérifiez que le modèle ML est bien chargé")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_4_config_detection():
    """Test 4: Configuration de la détection."""
    print_section("Test 4: Configuration de la détection")
    
    try:
        config = {
            "show_corners": True,
            "show_edges": True,
            "show_centering": True
        }
        
        response = requests.post(
            f"{BASE_URL}/scanner/config",
            json=config,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Configuration mise à jour")
                print(f"   Config actuelle: {data.get('config')}")
                return True
            else:
                print("❌ Échec de la configuration")
                return False
        else:
            print(f"❌ Erreur: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "="*60)
    print("  TESTS D'INTÉGRATION - POKIA SCANNER")
    print("="*60)
    
    results = []
    
    # Test 1: Statut
    results.append(("Statut du scanner", test_1_scanner_status()))
    
    if not results[0][1]:
        print("\n❌ Le scanner n'est pas accessible. Arrêt des tests.")
        print("   Lancez l'application avec: python app.py")
        return
    
    time.sleep(1)
    
    # Test 2: Capture
    image_path = test_2_capture_image()
    results.append(("Capture d'image", image_path is not None))
    
    time.sleep(1)
    
    # Test 3: Analyse (seulement si la capture a réussi)
    if image_path:
        results.append(("Analyse de la carte", test_3_analyze_capture(image_path)))
    else:
        print("\n⚠️  Test d'analyse ignoré (pas d'image capturée)")
        results.append(("Analyse de la carte", False))
    
    time.sleep(1)
    
    # Test 4: Configuration
    results.append(("Configuration", test_4_config_detection()))
    
    # Résumé
    print_section("RÉSUMÉ DES TESTS")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    print(f"\n{passed}/{total} tests réussis")
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés! L'intégration fonctionne correctement.")
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez la configuration.")

if __name__ == "__main__":
    print("POKIA - Test d'intégration de la détection et gradation")
    print("========================================================\n")
    print("Ce script teste l'intégration complète du système.")
    print("Assurez-vous que l'application Flask est en cours d'exécution.\n")
    
    input("Appuyez sur Entrée pour commencer les tests...")
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
    
    print("\n" + "="*60)
    print("  FIN DES TESTS")
    print("="*60 + "\n")
