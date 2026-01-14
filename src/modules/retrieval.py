import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
@dataclass
class RetrievalConfig:
    collection_name: str = "code_travail_tunisien"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_top_k: int = 5
    min_score_threshold: float = 0.3
    rerank_enabled: bool = True

# -----------------------------
# CLIENT QDRANT
# -----------------------------
qdrant = QdrantClient(
    url="https://81cb2705-ec2f-4b4e-9603-88368c5abb3d.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ynh283IPcSfyUZZp55FGDuqtG0sAoiMdfVW3UYF_iRQ",
    timeout=120
)


# -----------------------------
# RETRIEVER PRINCIPAL
# -----------------------------
class CodeTravailRetriever:
    """Système de retrieval avancé pour le Code du Travail Tunisien"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.qdrant = qdrant
        self.model = SentenceTransformer(self.config.model_name)
        print(f"✓ Retriever initialized with collection: {self.config.collection_name}")
    
    def _normalize_query(self, query: str) -> str:
        """Normalise la requête utilisateur"""
        query = query.strip().lower()
        query = re.sub(r'\s+', ' ', query)
        return query
    
    def _embed_query(self, query: str) -> List[float]:
        """Génère l'embedding de la requête"""
        return self.model.encode(query, convert_to_numpy=True).tolist()
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Recherche les chunks les plus pertinents
        
        Args:
            query: Question ou requête de l'utilisateur
            top_k: Nombre de résultats à retourner
            filters: Filtres Qdrant (ex: {'article': 'Article 1'})
            score_threshold: Score minimum de pertinence
            
        Returns:
            Liste de chunks avec scores et métadonnées
        """
        top_k = top_k or self.config.default_top_k
        score_threshold = score_threshold or self.config.min_score_threshold
        
        normalized_query = self._normalize_query(query)
        query_vector = self._embed_query(normalized_query)
        
        search_params = {
            'collection_name': self.config.collection_name,
            'query': query_vector,
            'limit': top_k * 2 if self.config.rerank_enabled else top_k,
            'with_payload': True
        }
        
        if filters:
            search_params['query_filter'] = filters
        
        results = self.qdrant.query_points(**search_params)
        
        formatted_results = []
        for hit in results.points:
            score = hit.score
            if score < score_threshold:
                continue
                
            result = {
                'id': hit.id,
                'score': score,
                'text': hit.payload.get('text', ''),
                'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
            }
            formatted_results.append(result)
        
        if self.config.rerank_enabled and len(formatted_results) > top_k:
            formatted_results = self._rerank(normalized_query, formatted_results)
        
        return formatted_results[:top_k]
    
    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-classe les résultats en fonction de la pertinence contextuelle"""
        query_terms = set(query.split())
        
        for result in results:
            text_lower = result['text'].lower()
            text_terms = set(text_lower.split())
            
            term_overlap = len(query_terms & text_terms) / len(query_terms) if query_terms else 0
            
            text_length = len(result['text'])
            length_penalty = 1.0
            if text_length < 50:
                length_penalty = 0.8
            elif text_length > 2000:
                length_penalty = 0.9
            
            result['rerank_score'] = (
                result['score'] * 0.7 + 
                term_overlap * 0.3
            ) * length_penalty
        
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results
    
    def retrieve_by_article(self, article_number: str, top_k: int = 10) -> List[Dict]:
        """
        Récupère tous les chunks d'un article spécifique - VERSION CORRIGÉE
        
        Args:
            article_number: Numéro de l'article (ex: "Article 1", "1")
            top_k: Nombre maximum de résultats
        """
        # Normalisation du numéro d'article
        if not article_number.lower().startswith('article'):
            article_number = f"Article {article_number}"
        
        filters = {
            'must': [
                {
                    'key': 'article',
                    'match': {'value': article_number}
                }
            ]
        }
        
        # Utilisation de scroll au lieu de retrieve
        try:
            results = self.qdrant.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=filters,
                limit=top_k,
                with_payload=True
            )
            
            formatted_results = []
            for point in results[0]:  # results[0] contient les points
                formatted_results.append({
                    'id': point.id,
                    'text': point.payload.get('text', ''),
                    'metadata': {k: v for k, v in point.payload.items() if k != 'text'}
                })
            
            return formatted_results
        except Exception as e:
            print(f"❌ Erreur retrieve_by_article: {e}")
            return []
    
    def retrieve_by_chapter(self, chapter_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """Recherche dans un chapitre spécifique"""
        filters = {
            'must': [
                {
                    'key': 'chapitre',
                    'match': {'value': chapter_name}
                }
            ]
        }
        
        return self.retrieve(query, top_k=top_k, filters=filters)
    
    def multi_query_retrieve(
        self, 
        queries: List[str], 
        top_k_per_query: int = 3,
        deduplicate: bool = True
    ) -> List[Dict]:
        """
        Recherche avec plusieurs requêtes et fusion des résultats
        
        Args:
            queries: Liste de requêtes
            top_k_per_query: Nombre de résultats par requête
            deduplicate: Supprime les doublons
        """
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = self.retrieve(query, top_k=top_k_per_query)
            
            for result in results:
                if deduplicate:
                    if result['id'] in seen_ids:
                        continue
                    seen_ids.add(result['id'])
                
                all_results.append(result)
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results
    
    def get_context_window(
        self, 
        chunk_id: int, 
        window_size: int = 2
    ) -> List[Dict]:
        """
        Récupère un chunk avec son contexte (chunks avant/après)
        
        Args:
            chunk_id: ID du chunk central
            window_size: Nombre de chunks avant et après
        """
        results = []
        
        for offset in range(-window_size, window_size + 1):
            target_id = chunk_id + offset
            
            try:
                point = self.qdrant.retrieve(
                    collection_name=self.config.collection_name,
                    ids=[target_id],
                    with_payload=True
                )
                
                if point:
                    results.append({
                        'id': point[0].id,
                        'text': point[0].payload.get('text', ''),
                        'metadata': {k: v for k, v in point[0].payload.items() if k != 'text'},
                        'offset': offset
                    })
            except:
                continue
        
        return results

# -----------------------------
# CLASSE UTILITAIRE DE RÉSULTATS
# -----------------------------
class RetrievalResult:
    """Encapsule et formate les résultats de recherche"""
    
    def __init__(self, results: List[Dict]):
        self.results = results
    
    def get_texts(self) -> List[str]:
        """Retourne uniquement les textes"""
        return [r['text'] for r in self.results]
    
    def get_top_result(self) -> Optional[Dict]:
        """Retourne le résultat le plus pertinent"""
        return self.results[0] if self.results else None
    
    def format_for_llm(self, max_length: int = 2000) -> str:
        """
        Formate les résultats pour un prompt LLM
        
        Args:
            max_length: Longueur maximale du contexte
        """
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(self.results, 1):
            text = result['text']
            article = result['metadata'].get('article', 'N/A')
            
            chunk_text = f"[Source {i} - {article}]\n{text}\n"
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > max_length:
                break
            
            context_parts.append(chunk_text)
            current_length += chunk_length
        
        return "\n".join(context_parts)
    
    def to_json(self) -> List[Dict]:
        """Retourne les résultats en format JSON"""
        return self.results
    
    def print_summary(self):
        """Affiche un résumé des résultats"""
        print(f"\n📊 Found {len(self.results)} results\n")
        for i, result in enumerate(self.results, 1):
            article = result['metadata'].get('article', 'N/A')
            score = result.get('score', 0)
            text_preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
            
            print(f"{i}. [{article}] Score: {score:.3f}")
            print(f"   {text_preview}\n")


if __name__ == "__main__":
    # Test
    retriever = CodeTravailRetriever()
    
    # Test recherche simple
    results = retriever.retrieve("durée du congé annuel", top_k=3)
    print(f"✓ Test recherche: {len(results)} résultats")
    
    # Test retrieve_by_article
    article_results = retriever.retrieve_by_article("114", top_k=5)
    print(f"✓ Test article: {len(article_results)} résultats")