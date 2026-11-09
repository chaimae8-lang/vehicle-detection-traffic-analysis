# Intelligent Traffic Monitoring - Vehicle Detection and Model Optimization

Ce projet est une application de **suivi intelligent du trafic** qui utilise le modèle de deep learning **YOLOv8** pour détecter, compter et catégoriser les véhicules en temps réel à partir d'un flux vidéo. Il intègre également des fonctionnalités d'estimation de la densité du trafic ainsi que des outils d'exportation et de benchmark au format **ONNX** pour l'optimisation des performances.

## 🚀 Fonctionnalités

- **Détection multi-classes** : Identification des voitures, motos, bus et camions.
- **Analyse du trafic** : Calcul en temps réel du nombre total de véhicules et classification de la densité du trafic (`LOW`, `MEDIUM`, `HIGH`).
- **Superposition visuelle** : Affichage des boîtes de détection (bounding boxes) et des statistiques directement sur la vidéo.
- **Export de données** : Sauvegarde des métriques par image (nombre de véhicules, temps d'inférence) dans un fichier CSV.
- **Optimisation ONNX** : Module d'exportation du modèle YOLOv8 vers le format ONNX pour améliorer le déploiement.
- **Benchmarking** : Évaluation comparative des performances d'inférence (Moyenne, P95, Maximum).

## 📦 Prérequis

Avant de lancer le script, assurez-vous d'avoir installé les bibliothèques suivantes :

```bash
pip install ultralytics opencv-python numpy pandas torch
```

*Note : Si vous disposez d'un GPU compatible CUDA, assurez-vous que `torch` est correctement configuré pour l'utiliser afin d'accélérer l'inférence.*

## 🛠️ Structure des Fichiers Générés

Lors de l'exécution, le script génère automatiquement les fichiers suivants :
- `traffic_annotated.mp4` : La vidéo de sortie avec les détections et les statistiques incrustées.
- `traffic_results.csv` : Un rapport d'analyse image par image contenant l'historique du trafic.
- `traffic_benchmark_frame.jpg` : Une image de référence extraite de la vidéo pour servir aux tests de performance.
- `yolov8n.onnx` : Le modèle converti (si l'option d'export est activée).

## 💻 Utilisation

Le script s'exécute en ligne de commande et accepte plusieurs arguments pour adapter son comportement.

### 1. Exécution standard (Analyse vidéo seule)
Pour traiter la vidéo par défaut (`Cars_driving_at_night.webm`) :
```bash
python main.py
```

### 2. Spécifier une autre vidéo
```bash
python main.py --video chemin/vers/votre_video.mp4
```

### 3. Exporter le modèle en ONNX
Ajoutez l'argument `--export-onnx` pour convertir le modèle après le traitement de la vidéo :
```bash
python main.py --export-onnx
```

### 4. Lancer un Benchmark complet
Pour analyser la vidéo, exporter le modèle au format ONNX, puis mesurer précisément le temps d'inférence sur l'image de test sur CPU :
```bash
python main.py --benchmark --runs 50
```

### 5. Limiter le nombre d'images à analyser (Test rapide)
Si vous souhaitez tester le script rapidement sur les 100 premières images de la vidéo :
```bash
python main.py --max-frames 100
```

## 📊 Options de la ligne de commande

| Argument | Type | Par défaut | Description |
| :--- | :--- | :--- | :--- |
| `--video` | `str` | `Cars_driving_at_night.webm` | Chemin vers la vidéo de trafic à analyser. |
| `--export-onnx`| `flag`| *False* | Exporte le modèle YOLO au format ONNX. |
| `--benchmark` | `flag`| *False* | Exporte et lance le benchmark de performance ONNX. |
| `--runs` | `int` | `30` | Nombre d'itérations pour les mesures du benchmark. |
| `--max-frames` | `int` | `0` | Nombre maximal d'images à traiter (0 = toute la vidéo). |

## ⏱️ Seuils de Densité du Trafic

Le niveau de trafic est déterminé en fonction du nombre total de véhicules visibles sur l'image :
- **LOW** : $\le$ 5 véhicules.
- **MEDIUM** : Entre 6 et 15 véhicules.
- **HIGH** : $>$ 15 véhicules.

*(Ces valeurs sont illustratives et peuvent être calibrées dans la fonction `estimate_traffic_density` selon l'angle de votre caméra).*
