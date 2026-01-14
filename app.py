"""
Interface Streamlit Chat avec Reasoning Engine
==============================================

Installation:
    pip install streamlit requests

Usage:
    streamlit run app.py
"""

import streamlit as st
import requests
from typing import Dict
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
API_BASE_URL = "http://localhost:8000"

# Configuration de la page
st.set_page_config(
    page_title="Code du Travail Tunisien - Chat Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STYLES CSS
# ============================================
st.markdown("""
<style>
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Messages du chat */
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-right: 20%;
    }
    .message-header {
        font-weight: 600;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .user-message .message-header {
        color: white;
    }
    .assistant-message .message-header {
        color: #1f77b4;
    }
    .message-content {
        line-height: 1.6;
        font-size: 1rem;
        white-space: pre-wrap;
    }
    
    /* Badge pour reasoning */
    .reasoning-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    /* Source card */
    .source-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .article-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .score-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Thinking chain */
    .thinking-stage {
        background-color: #f8f9fa;
        border-left: 3px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .thinking-stage-title {
        color: #667eea;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Welcome message */
    .welcome-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Stats box */
    .stat-box {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FONCTIONS API
# ============================================

def check_api_health() -> bool:
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def query_rag(
    question: str,
    user_id: str,
    session_id: str,
    top_k: int = 8, 
    enable_thinking: bool = True,
    enable_memory: bool = True,
    show_thinking_chain: bool = False
) -> Dict:
    """Envoie une requête RAG à l'API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={
                "question": question,
                "user_id": user_id,
                "session_id": session_id,
                "top_k": top_k,
                "enable_thinking": enable_thinking,
                "enable_memory": enable_memory,
                "show_thinking_chain": show_thinking_chain
            },
            timeout=90
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_api_stats() -> Dict:
    """Récupère les statistiques de l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=5)
        return response.json()
    except:
        return {}

# ============================================
# COMPOSANTS UI
# ============================================

def display_chat_message(role: str, content: str, sources: Dict = None, metadata: Dict = None):
    """Affiche un message dans le style chatbot"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">
                <span style="font-size: 1.3rem;">👤</span>
                <span>Vous</span>
            </div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Badge pour indiquer le reasoning
        reasoning_badge = ""
        if metadata and metadata.get('used_reasoning'):
            reasoning_badge = '<span class="reasoning-badge">🧠 Multi-stage Reasoning</span>'
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">
                <span style="font-size: 1.3rem;">⚖️</span>
                <span>Assistant Juridique</span>
                {reasoning_badge}
            </div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chaîne de réflexion (si disponible)
        if metadata and metadata.get('thinking_chain'):
            with st.expander("🧠 Voir la chaîne de réflexion", expanded=False):
                chain = metadata['thinking_chain']
                
                if chain.get('query_rewriting'):
                    st.markdown("""
                    <div class="thinking-stage">
                        <div class="thinking-stage-title">1️⃣ Reformulation de la question</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.text(chain['query_rewriting'])
                
                if chain.get('legal_analysis'):
                    st.markdown("""
                    <div class="thinking-stage">
                        <div class="thinking-stage-title">2️⃣ Analyse juridique</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.text(chain['legal_analysis'][:500] + "...")
                
                if chain.get('final_answer'):
                    st.markdown("""
                    <div class="thinking-stage">
                        <div class="thinking-stage-title">3️⃣ Réponse finale</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.text(chain['final_answer'][:500] + "...")
        
        # Requêtes optimisées
        if metadata and metadata.get('optimized_queries'):
            with st.expander("🔍 Requêtes de recherche utilisées", expanded=False):
                for i, query in enumerate(metadata['optimized_queries'], 1):
                    st.write(f"{i}. `{query}`")
        
        # Sources
        if sources and len(sources) > 0:
            with st.expander(f"📚 Voir les {len(sources)} source(s) juridiques", expanded=False):
                for src_id, src_info in sources.items():
                    st.markdown(f"""
                    <div class="source-card">
                        <div>
                            <span class="article-badge">{src_info['article']}</span>
                            <span class="score-badge">Pertinence: {src_info['score']:.1%}</span>
                        </div>
                        <p style="margin-top: 0.8rem; font-size: 0.9rem; color: #555; line-height: 1.5;">
                        {src_info['text'][:400]}{'...' if len(src_info['text']) > 400 else ''}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Statistiques
        if metadata and any(metadata.values()):
            with st.expander("📊 Statistiques de la réponse", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📚 Sources", metadata.get('num_sources', 0))
                with col2:
                    st.metric("🧠 Étapes", "3" if metadata.get('used_reasoning') else "1")
                with col3:
                    st.metric("⏱️ Temps", f"{metadata.get('response_time', 0):.1f}s" if 'response_time' in metadata else "N/A")

# ============================================
# INITIALISATION
# ============================================

# Header
st.markdown('<h1 class="main-header">⚖️ Code du Travail Tunisien</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Assistant Juridique Intelligent avec Raisonnement Multi-Étapes</p>', unsafe_allow_html=True)

# Vérification API
api_status = check_api_health()
if api_status:
    st.success("✅ API connectée - Reasoning Engine actif")
else:
    st.error("❌ Impossible de se connecter à l'API. Assurez-vous que le serveur FastAPI est démarré.")
    st.code("uvicorn api:app --reload --host 0.0.0.0 --port 8000", language="bash")
    st.stop()

# Session State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'message_count' not in st.session_state:
    st.session_state.message_count = 0

# Identifiant utilisateur unique (persistent)
if 'user_id' not in st.session_state:
    import hashlib
    import time
    # Génère un user_id unique basé sur le timestamp et un random
    st.session_state.user_id = hashlib.md5(
        f"{time.time()}".encode()
    ).hexdigest()[:12]

# Identifiant de session (change à chaque "Nouveau")
if 'session_id' not in st.session_state:
    import hashlib
    import time
    st.session_state.session_id = hashlib.md5(
        f"{st.session_state.user_id}_{time.time()}".encode()
    ).hexdigest()[:12]

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Paramètres
    st.markdown("### 🔧 Paramètres de recherche")
    top_k = st.slider(
        "📊 Nombre d'articles",
        min_value=3,
        max_value=15,
        value=8,
        help="Nombre d'articles à récupérer"
    )
    
    # Activer/désactiver le reasoning
    enable_thinking = st.checkbox(
        "🧠 Activer le raisonnement multi-étapes",
        value=True,
        help="Active l'analyse approfondie avec 3 étapes de réflexion (recommandé)"
    )
    
    # Activer/désactiver la mémoire
    enable_memory = st.checkbox(
        "💭 Activer la mémoire conversationnelle",
        value=True,
        help="Se souvient des conversations précédentes pour des réponses contextuelles"
    )
    
    if enable_thinking:
        show_thinking_chain = st.checkbox(
            "🔍 Afficher la chaîne de réflexion",
            value=False,
            help="Montre les étapes de raisonnement de l'assistant"
        )
    else:
        show_thinking_chain = False
    
    st.markdown("---")
    
    # Actions
    st.markdown("### 🎛️ Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            # Effacer aussi la mémoire côté serveur
            try:
                requests.post(
                    f"{API_BASE_URL}/api/memory/clear",
                    params={"user_id": st.session_state.user_id, "session_id": st.session_state.session_id},
                    timeout=5
                )
            except:
                pass
            st.rerun()
    
    with col2:
        if st.button("🔄 Nouveau", use_container_width=True):
            # Sauvegarder la session actuelle en long-term
            try:
                requests.post(
                    f"{API_BASE_URL}/api/memory/save",
                    params={"user_id": st.session_state.user_id, "session_id": st.session_state.session_id},
                    timeout=5
                )
            except:
                pass
            
            # Nouvelle session
            import hashlib
            import time
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.session_state.session_id = hashlib.md5(
                f"{st.session_state.user_id}_{time.time()}".encode()
            ).hexdigest()[:12]
            st.rerun()
    
    st.markdown("---")
    
    # Statistiques
    st.markdown("### 📊 Statistiques")
    
    # Stats de session
    st.markdown(f"""
    <div class="stat-box">
        <strong>💬 Messages:</strong> {len(st.session_state.chat_history)}
    </div>
    <div class="stat-box">
        <strong>❓ Questions:</strong> {st.session_state.message_count}
    </div>
    """, unsafe_allow_html=True)
    
    # Stats API
    api_stats = get_api_stats()
    if api_stats and 'total_articles' in api_stats:
        st.markdown(f"""
        <div class="stat-box">
            <strong>📚 Articles totaux:</strong> {api_stats['total_articles']}
        </div>
        <div class="stat-box">
            <strong>🧠 Étapes de raisonnement:</strong> {api_stats.get('reasoning_stages', 3)}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Info
    st.markdown("### ℹ️ À propos")
    st.info("""
    **Nouveau !** Raisonnement multi-étapes :
    
    1️⃣ Reformulation de la question
    2️⃣ Analyse juridique approfondie  
    3️⃣ Réponse humaine et actionnable
    
    **Modèle:** Llama 3.3 70B
    **Sources:** Code du Travail Tunisien
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.85rem; color: #666;">
    <strong>Version 2.0</strong><br>
    © 2024 - RAG System + Reasoning
    </div>
    """, unsafe_allow_html=True)

# ============================================
# ZONE PRINCIPALE - CHAT
# ============================================

# Conteneur pour les messages
chat_container = st.container()

# Affichage des messages
with chat_container:
    if not st.session_state.chat_history:
        # Message de bienvenue
        st.markdown("""
        <div class="welcome-box">
            <h2 style="margin: 0 0 1rem 0;">👋 Bienvenue!</h2>
            <p style="font-size: 1.1rem; margin: 0;">
                Je suis votre assistant juridique spécialisé dans le Code du Travail Tunisien.
                <br><br>
                <strong>✨ Nouveau :</strong> Je raisonne maintenant en plusieurs étapes pour vous donner
                les réponses les plus précises et actionnables !
                <br><br>
                <strong>Posez-moi vos questions</strong> sur vos droits, obligations, congés, salaires, licenciement, etc.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Affichage de l'historique
        for message in st.session_state.chat_history:
            display_chat_message(
                role=message['role'],
                content=message['content'],
                sources=message.get('sources'),
                metadata=message.get('metadata')
            )

# Séparateur
st.markdown("---")

# ============================================
# ZONE DE SAISIE
# ============================================

# Formulaire de saisie
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "💭 Posez votre question ici...",
            placeholder="Ex: Mon employeur peut-il me licencier sans préavis?",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col2:
        submit_button = st.form_submit_button(
            "Envoyer 🚀",
            use_container_width=True,
            type="primary"
        )

# ============================================
# TRAITEMENT DE LA QUESTION
# ============================================

if submit_button and user_input and user_input.strip():
    # Incrémenter le compteur
    st.session_state.message_count += 1
    
    # Ajouter le message utilisateur
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Traiter la requête
    with st.spinner("🧠 Analyse en cours avec raisonnement multi-étapes..." if enable_thinking else "🔍 Recherche en cours..."):
        start_time = datetime.now()
        result = query_rag(
            user_input,
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            top_k=top_k,
            enable_thinking=enable_thinking,
            enable_memory=enable_memory,
            show_thinking_chain=show_thinking_chain
        )
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Gérer les erreurs
        if 'error' in result and not result.get('answer'):
            assistant_message = f"❌ Désolé, j'ai rencontré une erreur : {result['error']}"
            sources = {}
            metadata = {}
        else:
            assistant_message = result.get('answer', 'Je n\'ai pas pu générer une réponse appropriée.')
            sources = result.get('sources', {})
            metadata = {
                'num_sources': result.get('num_sources', 0),
                'response_time': response_time,
                'used_reasoning': enable_thinking,
                'used_memory': enable_memory,
                'optimized_queries': result.get('optimized_queries'),
                'thinking_chain': result.get('thinking_chain')
            }
        
        # Ajouter la réponse de l'assistant
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': assistant_message,
            'sources': sources,
            'metadata': metadata,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Recharger
    st.rerun()

# ============================================
# FOOTER
# ============================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem; padding: 1rem 0;">
    <p style="margin: 0.5rem 0;">
        <strong>⚖️ Code du Travail Tunisien - Assistant Intelligent avec Reasoning</strong>
    </p>
    <p style="margin: 0.5rem 0;">
        Propulsé par <strong>FastAPI</strong> • <strong>Qdrant</strong> • <strong>Groq</strong> • <strong>Llama 3.3 70B</strong> • <strong>Streamlit</strong>
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem;">
        Les réponses sont générées par IA et doivent être vérifiées par un professionnel du droit.
    </p>
</div>
""", unsafe_allow_html=True)