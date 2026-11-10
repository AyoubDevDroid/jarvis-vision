# 🏍️ Jarvis Vision — HUD trafic & lecture de plaques en temps réel

Détection de véhicules (YOLO11), suivi multi-objets et lecture de plaques (FastALPR) en direct depuis une webcam, une vidéo ou un flux RTSP, **100 % en local**.

> Version améliorée du projet [Orsy-git/jarvis](https://github.com/Orsy-git/jarvis).

## 🎬 Démo

<video src="https://media.githubusercontent.com/media/AyoubDevDroid/jarvis-vision/main/docs/jarvis_demo_flotte.mp4" controls width="100%"></video>

[▶️ Voir la vidéo de démo du suivi de flotte](docs/jarvis_demo_flotte.mp4) (plaques floutées)

## ✨ Fonctionnalités

- Voitures, motos, bus, camions et vélos détectés et colorés par type
- Lecture des plaques **fiabilisée par vote** sur plusieurs images
- Zone ROI avec compteur lissé et alerte « RALENTIR » quand le trafic est dense
- Flèche vers la section la plus chargée de la zone
- Compteur FPS en direct
- **Mode léger sans PyTorch** (ONNX Runtime) : installation de ~100 Mo au lieu de plusieurs Go

## 🚀 Ce qui a changé par rapport à la v1

| Problème v1 | Solution v2 |
|---|---|
| Saccades : toutes les plaques lues sur la même image | OCR étalé dans le temps, 2 lectures max par image, les véhicules proches en priorité |
| Une mauvaise lecture écrasait la bonne | Vote majoritaire : plaque verrouillée après 3 lectures identiques, puis l'OCR s'arrête |
| Seules les voitures étaient lues | Voitures, motos, bus et camions |
| Plaque attribuée au mauvais véhicule sans ID | OCR uniquement sur les véhicules suivis |
| Fuite mémoire du cache de plaques | Purge automatique des véhicules disparus |
| Latence qui s'accumule en webcam/RTSP | Thread de capture qui ne garde que la dernière image |
| Texte OCR brut | Validation SIV (AA-123-AA) et FNI + correction 0/D, 8/B, 1/I selon la position |
| Alerte qui clignote | Comptage lissé sur 10 images |
| Nécessite PyTorch | Nouveau backend ONNX + tracker IoU, aucun PyTorch requis |
| — | `--anonymize` masque les plaques pour publier une vidéo (RGPD) |
| — | Suivi de flotte : base SQLite, alertes, journal, Telegram |

## 🚐 Suivi de flotte

Jarvis peut signaler le passage des véhicules de **votre propre flotte** : cadre jaune autour du véhicule, bandeau d'alerte, journal des passages, photo et notification Telegram en option.

```powershell
# 1. Créer la liste (une fois)
python -m jarvis_vision.watchlist add AB-123-CD "Ambulance 1"
python -m jarvis_vision.watchlist add EF-456-GH "VSL 2" --note "Skoda Octavia"
# ou en masse depuis un CSV (plaque;libelle;note)
python -m jarvis_vision.watchlist import flotte_exemple.csv

# 2. Lancer la surveillance
python -m jarvis_vision.app --backend onnx --watchlist flotte.db --snapshots snapshots

# 3. Consulter / exporter les passages
python -m jarvis_vision.watchlist passages
python -m jarvis_vision.watchlist export passages.csv
```

- Une plaque n'est comparée qu'une fois **verrouillée** par le vote (3 lectures identiques par défaut).
- 1 erreur OCR tolérée (`--tolerance`), seulement si une seule plaque de la flotte est assez proche.
- Le même véhicule n'est pas re-signalé avant 60 s (`--cooldown`).
- Notification Telegram : définir `TELEGRAM_BOT_TOKEN` et `TELEGRAM_CHAT_ID` avant de lancer.

```powershell
$env:TELEGRAM_BOT_TOKEN = "123456:ABC..."
$env:TELEGRAM_CHAT_ID = "123456789"
```

> Ce module est prévu pour des véhicules dont vous êtes responsable. Si la caméra filme la voie publique, les autres plaques restent des données personnelles : pensez à `--anonymize` et à l'information des salariés concernés (CNIL).

## 📦 Installation (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt            # mode léger
# ou : python -m pip install -r requirements-ultralytics.txt   (mode complet)
python -m pip install --force-reinstall --no-deps opencv-python
```

Les modèles se téléchargent automatiquement au premier lancement.

## ▶️ Utilisation

```powershell
# Webcam, mode léger
python -m jarvis_vision.app --backend onnx

# Vidéo, plaques masquées, export annoté
python -m jarvis_vision.app --backend onnx --source trajet.mp4 --anonymize --output sortie.mp4

# Flux RTSP, mode complet (ByteTrack)
python -m jarvis_vision.app --source "rtsp://user:pass@ip/stream"

# Plaques européennes et non françaises acceptées
python -m jarvis_vision.app --ocr-model european-plates-mobile-vit-v2-model --any-plate
```

| Option | Rôle | Défaut |
|---|---|---|
| `--backend` | `onnx` (léger) ou `ultralytics` | `ultralytics` |
| `--threshold` | Véhicules dans la ROI avant alerte | 4 |
| `--ocr-interval` | Images entre 2 lectures d'un même véhicule | 8 |
| `--max-ocr-per-frame` | Lectures OCR max par image | 2 |
| `--votes` | Lectures identiques pour verrouiller une plaque | 3 |
| `--min-vehicle-width` | Taille minimale (px) pour tenter l'OCR | 100 |
| `--any-plate` | Désactive la validation du format français | off |
| `--anonymize` | Masque le centre des plaques à l'écran | off |
| `--watchlist` | Base SQLite de la flotte à signaler | — |
| `--tolerance` | Erreurs OCR tolérées pour la flotte | 1 |
| `--cooldown` | Secondes avant de re-signaler un véhicule | 60 |
| `--snapshots` | Dossier des photos de passage | — |
| `--verbose` | Affiche chaque plaque verrouillée | off |

Touche `q` ou `Échap` pour quitter.

## 🧪 Tests

```powershell
python -m unittest discover -s tests -v
```

## ⚠️ Avertissement

L'analyse de la ROI est une estimation visuelle, pas une mesure de distance ni de vitesse. L'alerte invite à ralentir et à garder ses distances ; elle ne conseille aucune manœuvre. Ne remplace pas l'attention du conducteur.

Les plaques d'immatriculation sont des données personnelles : utilisez `--anonymize` avant de publier une vidéo filmée sur la voie publique.
