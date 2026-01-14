from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import sys
from pathlib import Path

# Ajouter le chemin des modules
sys.path.append(str(Path(__file__).parent))

from src.modules.retrieval import CodeTravailRetriever
from src.modules.reasoning import LegalThinkingEngine, ThinkingConfig
from src.modules.memory import ConversationMemory, MemoryConfig

# ============================================
# CONFIGURATION
# ============================================
GROQ_API_KEY = "API_KEY_GROQ_ICI"  # Remplacer par votre clé API Groq

# ============================================
# MODÈLES PYDANTIC
# ============================================
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question de l'utilisateur")
    user_id: str = Field(..., description="Identifiant unique de l'utilisateur")
    session_id: Optional[str] = Field(None, description="Identifiant de session (auto-généré si absent)")
    top_k: Optional[int] = Field(8, ge=1, le=20, description="Nombre de résultats")
    enable_thinking: Optional[bool] = Field(True, description="Activer le raisonnement multi-étapes")
    enable_memory: Optional[bool] = Field(True, description="Activer la mémoire conversationnelle")
    show_thinking_chain: Optional[bool] = Field(False, description="Inclure la chaîne de réflexion dans la réponse")

class Source(BaseModel):
    article: str
    text: str
    score: float
    hierarchy: Optional[str] = None

class ThinkingChain(BaseModel):
    """Chaîne de réflexion complète"""
    original_query: Optional[str] = None
    query_rewriting: Optional[str] = None
    legal_analysis: Optional[str] = None
    final_answer: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Dict[str, Source]
    num_sources: int
    user_id: str
    session_id: str
    optimized_queries: Optional[List[str]] = None
    thinking_chain: Optional[ThinkingChain] = None
    memory_used: Optional[bool] = None
    conversation_context: Optional[Dict] = None
    error: Optional[str] = None

# ============================================
# INITIALISATION FASTAPI
# ============================================
app = FastAPI(
    title="Code du Travail Tunisien - API RAG avec Reasoning",
    description="API pour interroger le Code du Travail Tunisien via RAG avec raisonnement multi-étapes",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# INITIALISATION DES COMPOSANTS
# ============================================
retriever = None
reasoning_engine = None
# Dict pour stocker les mémoires par user_id + session_id
active_memories: Dict[str, ConversationMemory] = {}

def get_or_create_memory(user_id: str, session_id: Optional[str] = None) -> ConversationMemory:
    """Récupère ou crée une mémoire pour un utilisateur/session"""
    memory_key = f"{user_id}_{session_id}" if session_id else user_id
    
    if memory_key not in active_memories:
        active_memories[memory_key] = ConversationMemory(
            user_id=user_id,
            session_id=session_id
        )
    
    return active_memories[memory_key]

@app.on_event("startup")
async def startup_event():
    """Initialise les composants au démarrage"""
    global retriever, reasoning_engine
    print("🚀 Initialisation du système RAG avec Reasoning Engine...")
    try:
        # Retriever
        retriever = CodeTravailRetriever()
        print("✓ Retriever initialisé")
        
        # Reasoning Engine (remplace le generator)
        config = ThinkingConfig(
            enable_verbose=False  # Désactiver en production
        )
        reasoning_engine = LegalThinkingEngine(
            groq_api_key=GROQ_API_KEY,
            config=config
        )
        print("✓ Reasoning Engine initialisé")
        
        print("✅ API prête avec Reasoning multi-étapes!")
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        raise

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API Code du Travail Tunisien - Reasoning Engine",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Multi-stage reasoning",
            "Query rewriting",
            "Legal analysis",
            "Human-like responses"
        ],
        "endpoints": {
            "docs": "/docs",
            "query": "/api/query",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "retriever": retriever is not None,
        "reasoning_engine": reasoning_engine is not None,
        "reasoning_stages": 3
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Requête RAG avec raisonnement multi-étapes + mémoire conversationnelle
    
    **Pipeline:**
    1. Récupération du contexte mémoire (short-term + long-term)
    2. Reformulation de la query avec contexte
    3. Retrieval des articles pertinents
    4. Analyse juridique + Réponse humaine
    5. Sauvegarde dans la mémoire
    
    **Paramètres:**
    - **question**: Question de l'utilisateur
    - **user_id**: Identifiant unique de l'utilisateur (requis)
    - **session_id**: Identifiant de session (optionnel, auto-généré si absent)
    - **top_k**: Nombre d'articles à récupérer (1-20)
    - **enable_thinking**: Activer le raisonnement multi-étapes
    - **enable_memory**: Activer la mémoire conversationnelle
    - **show_thinking_chain**: Inclure la chaîne de réflexion complète
    """
    try:
        # ============================================
        # 1. GESTION DE LA MÉMOIRE
        # ============================================
        memory = None
        memory_context = ""
        
        if request.enable_memory:
            memory = get_or_create_memory(request.user_id, request.session_id)
            
            # Ajouter le message utilisateur à la mémoire
            memory.add_message("user", request.question)
            
            # Récupérer le contexte conversationnel
            memory_context = memory.format_context_for_llm(
                current_query=request.question,
                include_long_term=True
            )
        
        # ============================================
        # 2. MODE SANS REASONING (Simple)
        # ============================================
        if not request.enable_thinking:
            retrieved_chunks = retriever.retrieve(
                query=request.question,
                top_k=request.top_k
            )
            
            if not retrieved_chunks:
                answer = "Désolé, je n'ai trouvé aucune information pertinente dans le Code du Travail."
                
                if memory:
                    memory.add_message("assistant", answer)
                
                return QueryResponse(
                    question=request.question,
                    answer=answer,
                    sources={},
                    num_sources=0,
                    user_id=request.user_id,
                    session_id=memory.session_id if memory else "no-session",
                    memory_used=request.enable_memory,
                    error="No relevant chunks found"
                )
            
            context = "\n\n".join([
                f"{chunk.get('metadata', {}).get('article', 'N/A')}: {chunk['text']}"
                for chunk in retrieved_chunks[:3]
            ])
            
            answer = f"Voici les articles pertinents trouvés:\n\n{context}"
            
            if memory:
                memory.add_message("assistant", answer)
            
            sources = {
                str(i): Source(
                    article=chunk.get('metadata', {}).get('article', 'N/A'),
                    text=chunk['text'],
                    score=chunk.get('score', 0.0)
                )
                for i, chunk in enumerate(retrieved_chunks, 1)
            }
            
            return QueryResponse(
                question=request.question,
                answer=answer,
                sources=sources,
                num_sources=len(sources),
                user_id=request.user_id,
                session_id=memory.session_id if memory else "no-session",
                memory_used=request.enable_memory
            )
        
        # ============================================
        # 3. MODE AVEC REASONING + MEMORY
        # ============================================
        result = reasoning_engine.process_query(
            user_query=request.question,
            retriever=retriever,
            top_k=request.top_k,
            memory_context=memory_context  # Inject memory context
        )
        
        answer = result.get('answer', '')
        
        # Sauvegarder la réponse dans la mémoire
        if memory:
            memory.add_message(
                "assistant",
                answer,
                metadata={
                    "sources": result.get('sources', {}),
                    "optimized_queries": result.get('optimized_queries', [])
                }
            )
        
        # Formater les sources
        formatted_sources = {}
        for src_id, src_info in result.get('sources', {}).items():
            formatted_sources[src_id] = Source(
                article=src_info['article'],
                text=src_info['text'][:500],
                score=src_info.get('score', 0.0),
                hierarchy=src_info.get('hierarchy', '')
            )
        
        # Préparer la chaîne de réflexion si demandée
        thinking_chain = None
        if request.show_thinking_chain:
            chain = result.get('thinking_chain', {})
            thinking_chain = ThinkingChain(
                original_query=chain.get('original_query'),
                query_rewriting=chain.get('query_rewriting'),
                legal_analysis=chain.get('legal_analysis'),
                final_answer=chain.get('final_answer')
            )
        
        # Contexte conversationnel (pour debug/info)
        conversation_context = None
        if memory and request.show_thinking_chain:
            conversation_context = {
                "short_term_messages": len(memory.short_term_memory),
                "session_id": memory.session_id
            }
        
        return QueryResponse(
            question=result.get('question', request.question),
            answer=answer,
            sources=formatted_sources,
            num_sources=result.get('num_sources', 0),
            user_id=request.user_id,
            session_id=memory.session_id if memory else "no-session",
            optimized_queries=result.get('optimized_queries'),
            thinking_chain=thinking_chain,
            memory_used=request.enable_memory,
            conversation_context=conversation_context,
            error=result.get('error')
        )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Erreur dans /api/query: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )

@app.get("/api/stats")
async def get_stats():
    """Statistiques du système"""
    try:
        # Compter les documents dans Qdrant
        collection_info = retriever.qdrant.get_collection(
            collection_name=retriever.config.collection_name
        )
        
        return {
            "total_articles": collection_info.points_count,
            "collection_name": retriever.config.collection_name,
            "reasoning_stages": 3,
            "model": reasoning_engine.config.model_name,
            "active_sessions": len(active_memories),
            "memory_enabled": True
        }
    except Exception as e:
        return {
            "error": str(e)
        }

@app.post("/api/memory/clear")
async def clear_memory(user_id: str, session_id: Optional[str] = None):
    """Efface la mémoire court-terme d'une session"""
    memory_key = f"{user_id}_{session_id}" if session_id else user_id
    
    if memory_key in active_memories:
        active_memories[memory_key].clear_short_term()
        return {"status": "success", "message": "Mémoire effacée"}
    
    return {"status": "not_found", "message": "Session non trouvée"}

@app.post("/api/memory/save")
async def save_session(user_id: str, session_id: Optional[str] = None):
    """Sauvegarde la session actuelle en mémoire long-terme"""
    memory_key = f"{user_id}_{session_id}" if session_id else user_id
    
    if memory_key in active_memories:
        memory = active_memories[memory_key]
        memory.save_to_long_term()
        return {
            "status": "success",
            "message": f"{len(memory.short_term_memory)} messages sauvegardés"
        }
    
    return {"status": "not_found", "message": "Session non trouvée"}

@app.get("/api/memory/history")
async def get_history(user_id: str, limit: int = 20):
    """Récupère l'historique complet d'un utilisateur"""
    try:
        # Créer une mémoire temporaire pour accéder à l'historique
        temp_memory = ConversationMemory(user_id=user_id)
        history = temp_memory.get_user_history(limit=limit)
        
        return {
            "user_id": user_id,
            "total_messages": len(history),
            "history": history
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================
# POINT D'ENTRÉE
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )