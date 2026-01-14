# ⚖️ Code du Travail Tunisien - RAG System with Multi-Stage Reasoning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

Un système RAG (Retrieval-Augmented Generation) intelligent pour le Code du Travail Tunisien, utilisant un raisonnement multi-étapes et une mémoire conversationnelle pour fournir des réponses juridiques précises et contextuelles.

## 🌟 Fonctionnalités Principales

### 🧠 Raisonnement Multi-Étapes (3 Stages)
1. **Reformulation de la question** : Transforme la question utilisateur en requêtes de recherche optimales
2. **Analyse juridique approfondie** : Analyse les articles du Code du Travail dans leur contexte
3. **Réponse humaine et actionnable** : Génère une réponse claire avec conseils pratiques

### 💭 Mémoire Conversationnelle
- **Mémoire court-terme** : Maintient le contexte de la session active
- **Mémoire long-terme** : Sauvegarde l'historique dans Qdrant pour référence future
- **Recherche contextuelle** : Récupère les conversations pertinentes passées

### 🔍 Retrieval Avancé
- Recherche vectorielle avec Qdrant Cloud
- Multi-query retrieval avec déduplication
- Re-ranking basé sur la pertinence
- Support des filtres hiérarchiques (Livre, Titre, Chapitre, Section, Article)

### 🎨 Interface Utilisateur Moderne
- Interface chat intuitive avec Streamlit
- Affichage des sources juridiques avec scores de pertinence
- Visualisation de la chaîne de réflexion (optionnel)
- Statistiques en temps réel

## 📋 Table des Matières

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [API Documentation](#-api-documentation)
- [Exemples](#-exemples)
- [Technologies](#-technologies)
- [Contribuer](#-contribuer)
- [License](#-license)

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   FastAPI API   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ↓         ↓
┌─────────┐ ┌──────────────┐
│ Memory  │ │   Reasoning  │
│ System  │ │    Engine    │
└────┬────┘ └──────┬───────┘
     │             │
     │         ┌───┴────┐
     │         │        │
     ↓         ↓        ↓
┌─────────┐ ┌──────┐ ┌──────┐
│ Qdrant  │ │Groq  │ │Search│
│  Cloud  │ │ LLM  │ │Vector│
└─────────┘ └──────┘ └──────┘
```

### Pipeline de Traitement

```
User Question
     │
     ↓
[Memory Context Retrieval]
     │
     ↓
[Stage 1: Query Rewriting] → Optimized Queries
     │
     ↓
[Vector Search in Qdrant] → Relevant Articles
     │
     ↓
[Stage 2: Legal Analysis] → Deep Analysis
     │
     ↓
[Stage 3: Human Response] → Final Answer
     │
     ↓
[Save to Memory]
     │
     ↓
Response to User
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- Compte Qdrant Cloud (gratuit)
- Clé API Groq (gratuit)

### Étapes d'Installation

1. **Cloner le repository**
```bash
git clone https://github.com/yourusername/code-travail-rag.git
cd code-travail-rag
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configuration des clés API**

Créez un fichier `.env` à la racine :
```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

Ou modifiez directement dans `api.py` et les modules concernés.

5. **Préparer les données**

```bash
# Chunking du PDF
python src/modules/chunking.py

# Embedding et upload vers Qdrant
python src/modules/embedding.py
```

## ⚙️ Configuration

### Structure du Projet

```
code-travail-rag/
├── src/
│   └── modules/
│       ├── chunking.py          # Extraction et découpage du PDF
│       ├── embedding.py         # Génération des embeddings
│       ├── retrieval.py         # Système de recherche
│       ├── reasoning.py         # Moteur de raisonnement 3 étapes
│       ├── memory.py            # Mémoire conversationnelle
│       └── ingestion.py         # Ingestion de documents
├── api.py                       # API FastAPI
├── app.py                       # Interface Streamlit
├── data/
│   └── TN_Code_du_Travail.pdf  # PDF source
├── requirements.txt
├── .env
└── README.md
```

### Configuration des Modules

#### Chunking
```python
@dataclass
class ChunkingConfig:
    # 1 article = 1 chunk
    # Les sous-articles (5-2, 5-3) sont des chunks séparés
```

#### Embedding
```python
@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_size: int = 384
    batch_size: int = 100
```

#### Reasoning
```python
@dataclass
class ThinkingConfig:
    model_name: str = "llama-3.3-70b-versatile"
    temperature_query_rewrite: float = 0.1
    temperature_reasoning: float = 0.2
    temperature_response: float = 0.3
```

#### Memory
```python
@dataclass
class MemoryConfig:
    short_term_limit: int = 10
    long_term_retrieval_limit: int = 3
    relevance_threshold: float = 0.6
```

## 📖 Utilisation

### 1. Démarrer l'API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API disponible sur : `http://localhost:8000`
Documentation interactive : `http://localhost:8000/docs`

### 2. Lancer l'Interface Streamlit

```bash
streamlit run app.py
```

Interface disponible sur : `http://localhost:8501`


## 🛠️ Technologies

### Backend
- **FastAPI** : Framework API moderne et performant
- **Groq** : Inference LLM ultra-rapide (Llama 3.3 70B)
- **Qdrant Cloud** : Base de données vectorielle
- **Sentence Transformers** : Génération d'embeddings

### Frontend
- **Streamlit** : Interface utilisateur interactive

### Traitement
- **PyPDF2** / **pdfplumber** : Extraction de PDF
- **LangDetect** : Détection de langue
- **Python-docx** : Support DOCX

### Models
- **all-MiniLM-L6-v2** : Embeddings (384 dimensions)
- **Llama 3.3 70B** : Génération de réponses



</div>
