# DiscoSMILES

Initial context: Drug discovery projects entail cycles of design, synthesis, and testing that
yield a series of chemically related small molecules whose properties, such as
binding affinity to a given target protein, are progressively tailored to a
particular drug discovery goal. The use of deep-learning technologies could
augment the typical practice of using human intuition in the design cycle,
and thereby expedite drug discovery projects. Here, we present DESMILES,
a deep neural network model that advances the state of the art in
machine learning approaches to molecular design. We applied DESMILES to a
previously published benchmark that assesses the ability of a method to modify
input molecules to inhibit the dopamine receptor D2, and DESMILES yielded a
77% lower failure rate compared to state-of-the-art models.

To explain the ability of DESMILES to hone molecular properties,
we visualize a layer of the DESMILES network, and further demonstrate
this ability by using DESMILES to tailor the same molecules used in the D2
benchmark test to dock more potently against seven different receptors.

Dataset took from https://www.kaggle.com/datasets/imtkaggleteam/drug-discovery-chemical-space-design/data

## Génération et optimisation de molécules par Deep Learning

DiscoSMILES est un projet de recherche appliquée inspiré de **DESMILES**, un modèle de deep learning conçu pour assister la découverte de médicaments (drug discovery). Le projet vise à construire une plateforme complète permettant :

* de générer de nouvelles molécules à partir de structures existantes ;
* d’optimiser certaines propriétés chimiques ou biologiques ;
* d’explorer l’espace chimique via des représentations moléculaires SMILES ;
* de déployer un modèle IA utilisable depuis une interface web moderne.

Le projet est structuré avec :

* un **backend Python** pour l’entraînement, l’inférence et les traitements chimiques ;
* un **frontend Streamlit** pour la visualisation et l’interaction utilisateur ;
* un **déploiement cloud** pour exposer le modèle via API.

---

# Table des matières

1. [Objectifs du projet](#objectifs-du-projet)
2. [Dataset](#dataset)
3. [Architecture globale](#architecture-globale)
4. [Stack technique](#stack-technique)
5. [Structure du projet](#structure-du-projet)
6. [Pipeline Machine Learning](#pipeline-machine-learning)
7. [Backend](#backend)
8. [Frontend Streamlit](#frontend-streamlit)
9. [Déploiement Cloud](#déploiement-cloud)
10. [Monitoring et MLOps](#monitoring-et-mlops)
11. [Roadmap](#roadmap)
12. [Installation locale](#installation-locale)
13. [Commandes utiles](#commandes-utiles)
14. [Améliorations futures](#améliorations-futures)

---

# Objectifs du projet

Le projet DiscoSMILES a pour but de reproduire puis étendre les concepts de DESMILES.

## Fonctionnalités principales

### Génération moléculaire

* Générer de nouvelles molécules à partir d’un SMILES d’entrée.
* Produire des molécules chimiquement valides.
* Explorer l’espace latent des molécules.

### Optimisation de propriétés

Le modèle doit pouvoir optimiser des propriétés comme :

* affinité potentielle ;
* docking score ;
* logP ;
* toxicité ;
* solubilité ;
* drug-likeness (Lipinski).

### Visualisation

Le frontend permettra :

* d’afficher les structures chimiques ;
* de comparer les molécules ;
* d’afficher les embeddings ;
* de suivre l’évolution des propriétés.

### Déploiement IA

Le modèle sera exposé via une API cloud afin de permettre :

* l’inférence distante ;
* le scaling ;
* la séparation frontend/backend ;
* une architecture production-ready.

---

# Dataset

Dataset utilisé :

* Kaggle :
  [https://www.kaggle.com/datasets/imtkaggleteam/drug-discovery-chemical-space-design/data](https://www.kaggle.com/datasets/imtkaggleteam/drug-discovery-chemical-space-design/data)

## Contenu du dataset

Le dataset contient :

* des représentations SMILES ;
* des propriétés chimiques ;
* des informations d’activité biologique ;
* des molécules utilisées pour le drug design.

## Préprocessing prévu

### Étapes

* suppression des doublons ;
* validation des SMILES ;
* normalisation ;
* tokenization des séquences ;
* génération du vocabulaire ;
* création des datasets train/val/test.

### Librairies utilisées

* RDKit
* Pandas
* NumPy
* Scikit-learn

---

# Architecture globale

```text
                ┌────────────────────┐
                │   Frontend UI      │
                │     Streamlit      │
                └─────────┬──────────┘
                          │ HTTP
                          ▼
                ┌────────────────────┐
                │   FastAPI Backend  │
                │  Inference Server  │
                └─────────┬──────────┘
                          │
          ┌───────────────┼────────────────┐
          │                                │
          ▼                                ▼
┌──────────────────┐            ┌──────────────────┐
│ Deep Learning ML │            │ PostgreSQL / DB  │
│ PyTorch Model    │            │ Experiments      │
└──────────────────┘            └──────────────────┘
          │
          ▼
┌──────────────────┐
│ Cloud Deployment │
│ Docker + GPU     │
└──────────────────┘
```

---

# Stack technique

## Machine Learning

| Domaine                 | Technologie                |
| ----------------------- | -------------------------- |
| Deep Learning           | PyTorch                    |
| Chimie computationnelle | RDKit                      |
| NLP moléculaire         | Tokenizers / SentencePiece |
| Tracking                | MLflow                     |
| Expérimentation         | Weights & Biases           |

## Backend

| Domaine    | Technologie |
| ---------- | ----------- |
| API        | FastAPI     |
| Serveur    | Uvicorn     |
| Validation | Pydantic    |
| Async      | asyncio     |
| Auth       | JWT         |

## Frontend

| Domaine                   | Technologie |
| ------------------------- | ----------- |
| UI                        | Streamlit   |
| Visualisation moléculaire | py3Dmol     |
| Graphiques                | Plotly      |
| Data viz                  | Altair      |

## Infrastructure

| Domaine            | Technologie                  |
| ------------------ | ---------------------------- |
| Containerisation   | Docker                       |
| Orchestration      | Docker Compose               |
| CI/CD              | GitHub Actions               |
| Cloud              | AWS / GCP / Azure            |
| Déploiement modèle | Hugging Face / AWS SageMaker |
| Reverse proxy      | Nginx                        |

---

# Structure du projet

```text
DiscoSMILES/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── services/
│   │   ├── schemas/
│   │   └── main.py
│   │
│   ├── training/
│   │   ├── datasets/
│   │   ├── preprocessing/
│   │   ├── models/
│   │   ├── trainers/
│   │   └── utils/
│   │
│   ├── notebooks/
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│   ├── pages/
│   ├── components/
│   ├── utils/
│   └── app.py
│
├── deployment/
│   ├── docker/
│   ├── nginx/
│   ├── terraform/
│   └── github-actions/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── experiments/
├── docs/
├── README.md
└── docker-compose.yml
```

---

# Pipeline Machine Learning

## Étape 1 — Préprocessing

### Objectif

Transformer les SMILES en séquences exploitables par le modèle.

### Pipeline

```text
SMILES → Cleaning → Tokenization → Vocabulary → Tensorization
```

### Possibilités

* character-level tokenization ;
* BPE tokenization ;
* SELFIES representation ;
* graph representation.

---

## Étape 2 — Entraînement

### Architectures possibles

#### Option 1 — LSTM (simple baseline)

* Encoder/Decoder
* Seq2Seq
* Teacher forcing

#### Option 2 — Transformer

* meilleure scalabilité ;
* meilleure génération ;
* attention sur séquences longues.

#### Option 3 — Variational AutoEncoder

* espace latent moléculaire ;
* interpolation ;
* optimisation guidée.

#### Option 4 — Diffusion Model

* génération moderne ;
* exploration chimique avancée.

---

## Étape 3 — Optimisation moléculaire

Objectif : modifier une molécule existante pour améliorer ses propriétés.

### Méthodes

* reinforcement learning ;
* latent space optimization ;
* Bayesian optimization ;
* genetic algorithms.

---

## Étape 4 — Évaluation

### Métriques

| Métrique      | Description                    |
| ------------- | ------------------------------ |
| Validity      | Molécules chimiquement valides |
| Novelty       | Nouvelles molécules            |
| Diversity     | Diversité chimique             |
| QED           | Drug-likeness                  |
| Docking score | Affinité potentielle           |
| Similarity    | Similarité moléculaire         |

---

# Backend

Le backend est responsable de :

* charger les modèles ;
* effectuer l’inférence ;
* gérer les requêtes utilisateur ;
* exposer les endpoints ;
* gérer les expériences.

## API prévue

### Génération moléculaire

```http
POST /generate
```

Body :

```json
{
  "smiles": "CCO",
  "num_samples": 10
}
```

---

### Optimisation

```http
POST /optimize
```

Body :

```json
{
  "smiles": "CCO",
  "target_property": "docking"
}
```

---

### Visualisation

```http
GET /molecule/{id}
```

---

## Technologies backend

### FastAPI

Pourquoi FastAPI :

* performant ;
* async natif ;
* documentation Swagger automatique ;
* excellent pour les APIs ML.

### Pydantic

* validation stricte ;
* sérialisation ;
* robustesse des endpoints.

---

# Frontend Streamlit

Le frontend Streamlit sert de dashboard interactif.

## Fonctionnalités UI

### Upload et saisie

* saisie d’un SMILES ;
* import CSV ;
* batch generation.

### Visualisation moléculaire

* rendu 2D ;
* rendu 3D ;
* comparaison moléculaire.

### Dashboard IA

* scores de propriétés ;
* probabilités ;
* embeddings ;
* statistiques.

### Historique

* expériences sauvegardées ;
* tracking des générations.

---

## Exemple de flow utilisateur

```text
Utilisateur → Streamlit UI → FastAPI → Model → Résultat → UI
```

---

# Déploiement Cloud

## Objectif

Rendre le modèle accessible depuis internet.

---

## Architecture de déploiement

```text
Frontend Streamlit
        │
        ▼
Load Balancer / Nginx
        │
        ▼
FastAPI Inference API
        │
        ▼
GPU Instance
        │
        ▼
PyTorch Model
```

---

## Options de déploiement

### Option 1 — Hugging Face Spaces

Avantages :

* simple ;
* rapide ;
* gratuit pour prototype ;
* support Streamlit.

Bon pour :

* MVP ;
* démonstration ;
* portfolio.

---

### Option 2 — AWS SageMaker

Avantages :

* scalable ;
* GPU ;
* monitoring ;
* production.

Bon pour :

* gros modèles ;
* trafic réel ;
* MLOps.

---

### Option 3 — Railway / Render

Avantages :

* très simple ;
* CI/CD rapide ;
* faible coût.

Bon pour :

* projet étudiant ;
* prototype.

---

# Monitoring et MLOps

## Tracking d’expériences

### MLflow

Tracking :

* hyperparamètres ;
* métriques ;
* checkpoints ;
* modèles.

---

## Logging

### Logs backend

* erreurs ;
* temps d’inférence ;
* requêtes ;
* monitoring API.

---

## CI/CD

### GitHub Actions

Automatisation :

* tests ;
* lint ;
* build Docker ;
* déploiement.

---

# Roadmap

## Phase 1 — Base projet

* [ ] Structure repository
* [ ] Setup backend FastAPI
* [ ] Setup frontend Streamlit
* [ ] Setup Docker
* [ ] Setup CI/CD

---

## Phase 2 — Data pipeline

* [ ] Download dataset
* [ ] Cleaning
* [ ] Tokenization
* [ ] Dataset split
* [ ] DataLoader PyTorch

---

## Phase 3 — Baseline model

* [ ] LSTM model
* [ ] Training loop
* [ ] Evaluation
* [ ] Metrics

---

## Phase 4 — Advanced model

* [ ] Transformer
* [ ] Latent space
* [ ] Property optimization
* [ ] Docking integration

---

## Phase 5 — Frontend

* [ ] Molecule rendering
* [ ] Dashboard
* [ ] Interactive inference
* [ ] Experiment history

---

## Phase 6 — Deployment

* [ ] Dockerization
* [ ] Cloud deployment
* [ ] GPU serving
* [ ] Monitoring

---

# Installation locale

## Cloner le repository

```bash
git clone https://github.com/your-username/discosmiles.git
cd discosmiles
```

---

## Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## Docker

```bash
docker-compose up --build
```

---

# Commandes utiles

## Lancer le backend

```bash
uvicorn app.main:app --reload
```

---

## Lancer Streamlit

```bash
streamlit run app.py
```

---

## Entraîner le modèle

```bash
python training/train.py
```

---

# Améliorations futures

## Recherche avancée

* docking réel avec AutoDock Vina ;
* reinforcement learning ;
* diffusion models ;
* graph neural networks.

---

## Production

* Kubernetes ;
* inference batching ;
* autoscaling GPU ;
* monitoring Prometheus/Grafana.

---

## Scientifique

* benchmark GuacaMol ;
* benchmark MOSES ;
* comparaison DESMILES ;
* publication scientifique.

---

# Références

## Article DESMILES

* Moret et al., DESMILES: Deep neural network for molecular design.

---

## Librairies utiles

* RDKit
* PyTorch
* DeepChem
* Hugging Face Transformers
* PyTorch Lightning

---

# Auteur

Projet personnel de recherche IA appliquée à la découverte de médicaments.

Objectif : construire une plateforme moderne combinant :

* IA générative ;
* chimie computationnelle ;
* visualisation interactive ;
* MLOps et déploiement cloud.
