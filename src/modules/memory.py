"""
Module de Mémoire pour RAG Juridique - VERSION CORRIGÉE
======================================================
Corrections:
1. Ajout d'index Qdrant sur user_id pour les filtres
2. Correction de la méthode search_long_term
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from sentence_transformers import SentenceTransformer
import hashlib


@dataclass
class MemoryConfig:
    """Configuration du système de mémoire"""
    short_term_collection: str = "chat_short_term_memory"
    long_term_collection: str = "chat_long_term_memory"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_size: int = 384
    short_term_limit: int = 10
    long_term_retrieval_limit: int = 3
    relevance_threshold: float = 0.6
    qdrant_url: str = "https://81cb2705-ec2f-4b4e-9603-88368c5abb3d.europe-west3-0.gcp.cloud.qdrant.io:6333"
    qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ynh283IPcSfyUZZp55FGDuqtG0sAoiMdfVW3UYF_iRQ"


@dataclass
class Message:
    """Représente un message dans la conversation"""
    role: str
    content: str
    timestamp: str
    message_id: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ConversationMemory:
    """Gère la mémoire court-terme et long-terme des conversations"""
    
    def __init__(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        config: Optional[MemoryConfig] = None
    ):
        self.user_id = user_id
        self.session_id = session_id or self._generate_session_id()
        self.config = config or MemoryConfig()
        
        # Connexion Qdrant
        self.qdrant = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            timeout=60
        )
        
        # Modèle d'embedding
        self.model = SentenceTransformer(self.config.model_name)
        
        # Short-term memory (RAM)
        self.short_term_memory: List[Message] = []
        
        # Initialiser les collections avec index
        self._init_collections()
        
        print(f"✓ Memory initialisée pour user={user_id}, session={self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Génère un ID de session unique"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{self.user_id}_{timestamp}".encode()).hexdigest()[:12]
    
    def _init_collections(self):
        """Initialise les collections Qdrant avec index sur user_id"""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        
        # Short-term collection
        if self.config.short_term_collection not in collections:
            self.qdrant.create_collection(
                collection_name=self.config.short_term_collection,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance.COSINE
                )
            )
            # Créer l'index sur user_id
            self.qdrant.create_payload_index(
                collection_name=self.config.short_term_collection,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"✓ Collection créée: {self.config.short_term_collection} (avec index)")
        
        # Long-term collection
        if self.config.long_term_collection not in collections:
            self.qdrant.create_collection(
                collection_name=self.config.long_term_collection,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance.COSINE
                )
            )
            # Créer l'index sur user_id - CRITIQUE pour les filtres
            self.qdrant.create_payload_index(
                collection_name=self.config.long_term_collection,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"✓ Collection créée: {self.config.long_term_collection} (avec index)")
    
    def _embed_text(self, text: str) -> List[float]:
        """Génère l'embedding d'un texte"""
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def _create_message_id(self) -> str:
        """Crée un ID unique pour un message"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(
            f"{self.user_id}_{self.session_id}_{timestamp}".encode()
        ).hexdigest()[:16]
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Message:
        """Ajoute un message à la mémoire court-terme"""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            message_id=self._create_message_id(),
            metadata=metadata or {}
        )
        
        self.short_term_memory.append(message)
        
        if len(self.short_term_memory) > self.config.short_term_limit * 2:
            self.short_term_memory = self.short_term_memory[-self.config.short_term_limit * 2:]
        
        print(f"📝 Message ajouté: {role} ({len(content)} chars)")
        return message
    
    def get_short_term_context(self, max_messages: Optional[int] = None) -> List[Dict]:
        """Récupère le contexte de la session actuelle"""
        limit = max_messages or self.config.short_term_limit
        recent_messages = self.short_term_memory[-limit:]
        return [msg.to_dict() for msg in recent_messages]
    
    def clear_short_term(self):
        """Efface la mémoire court-terme"""
        self.short_term_memory = []
        self.session_id = self._generate_session_id()
        print("🗑️ Mémoire court-terme effacée, nouvelle session créée")
    
    def save_to_long_term(self):
        """Sauvegarde la session actuelle dans la mémoire long-terme"""
        if not self.short_term_memory:
            print("⚠️ Aucun message à sauvegarder")
            return
        
        print(f"💾 Sauvegarde de {len(self.short_term_memory)} messages en long-term...")
        
        points = []
        for message in self.short_term_memory:
            embedding = self._embed_text(message.content)
            
            point = PointStruct(
                id=hash(message.message_id) % (10**10),
                vector=embedding,
                payload={
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp,
                    "message_id": message.message_id,
                    "metadata": message.metadata
                }
            )
            points.append(point)
        
        # Upload par batch
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.qdrant.upsert(
                collection_name=self.config.long_term_collection,
                points=batch
            )
        
        print(f"✅ {len(points)} messages sauvegardés en long-term")
    
    def search_long_term(
        self,
        query: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Recherche dans la mémoire long-terme - VERSION CORRIGÉE"""
        limit = limit or self.config.long_term_retrieval_limit
        
        query_vector = self._embed_text(query)
        
        # Filtre avec index
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=self.user_id)
                )
            ]
        )
        
        try:
            # Vérifier d'abord si la collection existe et a des points
            collection_info = self.qdrant.get_collection(self.config.long_term_collection)
            if collection_info.points_count == 0:
                print("ℹ️ Aucun historique long-term disponible")
                return []
            
            results = self.qdrant.query_points(
                collection_name=self.config.long_term_collection,
                query=query_vector,
                query_filter=user_filter,
                limit=limit,
                score_threshold=self.config.relevance_threshold,
                with_payload=True
            )
            
            formatted_results = []
            for hit in results.points:
                formatted_results.append({
                    "role": hit.payload["role"],
                    "content": hit.payload["content"],
                    "timestamp": hit.payload["timestamp"],
                    "session_id": hit.payload["session_id"],
                    "score": hit.score,
                    "metadata": hit.payload.get("metadata", {})
                })
            
            print(f"🔍 {len(formatted_results)} messages pertinents trouvés dans l'historique")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Erreur search_long_term: {e}")
            return []
    
    def get_user_history(self, limit: int = 50) -> List[Dict]:
        """Récupère tout l'historique d'un utilisateur"""
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=self.user_id)
                )
            ]
        )
        
        try:
            results, _ = self.qdrant.scroll(
                collection_name=self.config.long_term_collection,
                scroll_filter=user_filter,
                limit=limit,
                with_payload=True
            )
            
            history = []
            for point in results:
                history.append({
                    "role": point.payload["role"],
                    "content": point.payload["content"],
                    "timestamp": point.payload["timestamp"],
                    "session_id": point.payload["session_id"]
                })
            
            history.sort(key=lambda x: x["timestamp"], reverse=True)
            return history
            
        except Exception as e:
            print(f"❌ Erreur get_user_history: {e}")
            return []
    
    def get_context_for_query(
        self,
        current_query: str,
        include_long_term: bool = True
    ) -> Dict:
        """Récupère le contexte complet pour une nouvelle query"""
        context = {
            "short_term": [],
            "long_term": [],
            "formatted_context": ""
        }
        
        # Short-term
        short_term = self.get_short_term_context()
        context["short_term"] = short_term
        
        # Long-term
        if include_long_term:
            long_term = self.search_long_term(current_query)
            context["long_term"] = long_term
        
        # Formater
        formatted_parts = []
        
        if context["long_term"]:
            formatted_parts.append("📚 **Contexte des conversations précédentes:**\n")
            for msg in context["long_term"]:
                role_emoji = "👤" if msg["role"] == "user" else "⚖️"
                formatted_parts.append(
                    f"{role_emoji} [{msg['timestamp'][:10]}] {msg['content'][:150]}...\n"
                )
            formatted_parts.append("\n")
        
        if context["short_term"]:
            formatted_parts.append("💬 **Conversation actuelle:**\n")
            for msg in context["short_term"]:
                role_emoji = "👤" if msg["role"] == "user" else "⚖️"
                formatted_parts.append(f"{role_emoji} {msg['content']}\n")
        
        context["formatted_context"] = "".join(formatted_parts)
        return context
    
    def format_context_for_llm(
        self,
        current_query: str,
        include_long_term: bool = True
    ) -> str:
        """Formate le contexte mémoire pour injection dans le prompt LLM"""
        context = self.get_context_for_query(current_query, include_long_term)
        
        if not context["short_term"] and not context["long_term"]:
            return ""
        
        prompt_context = "\n" + "="*70 + "\n"
        prompt_context += "📝 CONTEXTE DE LA CONVERSATION\n"
        prompt_context += "="*70 + "\n\n"
        
        if context["long_term"]:
            prompt_context += "**Historique pertinent des conversations passées:**\n\n"
            for msg in context["long_term"]:
                role = "Utilisateur" if msg["role"] == "user" else "Assistant"
                prompt_context += f"• [{role}] {msg['content'][:200]}...\n"
            prompt_context += "\n"
        
        if context["short_term"]:
            prompt_context += "**Messages récents de cette conversation:**\n\n"
            for msg in context["short_term"]:
                role = "Utilisateur" if msg["role"] == "user" else "Assistant"
                prompt_context += f"• [{role}] {msg['content']}\n"
        
        prompt_context += "\n" + "="*70 + "\n"
        prompt_context += "La question actuelle de l'utilisateur doit être comprise dans ce contexte.\n"
        prompt_context += "="*70 + "\n\n"
        
        return prompt_context
    
    def end_session(self, save_to_long_term: bool = True):
        """Termine la session actuelle"""
        if save_to_long_term:
            self.save_to_long_term()
        self.clear_short_term()
        print(f"🔚 Session {self.session_id} terminée")
    
    def get_session_summary(self) -> Dict:
        """Résumé de la session actuelle"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages_count": len(self.short_term_memory),
            "short_term_messages": self.get_short_term_context()
        }