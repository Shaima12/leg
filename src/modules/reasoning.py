"""
Module de Raisonnement Multi-Étapes - VERSION CORRIGÉE
=====================================================
Ajout du support pour memory_context dans process_query
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from groq import Groq


@dataclass
class ThinkingConfig:
    """Configuration du moteur de réflexion"""
    model_name: str = "llama-3.3-70b-versatile"
    temperature_query_rewrite: float = 0.1
    temperature_reasoning: float = 0.2
    temperature_response: float = 0.3
    max_tokens: int = 2048
    enable_verbose: bool = True


class ThinkingPrompts:
    """Templates de prompts pour chaque étape"""
    
    STAGE_1_QUERY_REWRITING = """Tu es un expert en recherche juridique dans le Code du Travail Tunisien.

{memory_context}

**QUESTION ORIGINALE DE L'UTILISATEUR:**
{user_query}

**TON OBJECTIF (Étape 1/3 - Reformuler pour recherche optimale):**

L'utilisateur pose une question en langage naturel. Tu dois la transformer en requêtes de recherche optimales pour trouver les articles pertinents du Code du Travail.

**ANALYSE:**
1. Quelle est la vraie question juridique?
2. Quels concepts juridiques sont concernés?
3. Quels termes juridiques précis utiliser?
4. Quels mots-clés du Code du Travail chercher?

**FORMAT DE SORTIE:**
Génère 3-5 requêtes de recherche courtes et précises (5-10 mots max chacune).
Utilise des termes juridiques précis du droit du travail tunisien.

IMPORTANT: Retourne UNIQUEMENT les requêtes, une par ligne, sans numérotation ni explications."""

    STAGE_2_LEGAL_ANALYSIS = """Tu es un assistant juridique expert en Code du Travail Tunisien.

{memory_context}

**QUESTION ORIGINALE:**
{user_query}

**ARTICLES DU CODE DU TRAVAIL TROUVÉS:**
{legal_articles}

**TON ANALYSE JURIDIQUE (Étape 2/3 - Analyser situation + articles):**

Analyse la situation en profondeur:

**1. COMPRÉHENSION DE LA SITUATION:**
- Quel est le contexte concret?
- Quels sont les faits importants?
- Qui sont les parties impliquées (employeur/employé)?
- Quel est le vrai problème juridique?

**2. ANALYSE DES ARTICLES:**
- Que disent précisément ces articles du Code du Travail?
- Comment s'appliquent-ils à cette situation?
- Quelles sont les conditions et exceptions?
- Quels sont les droits et obligations de chaque partie?

**3. RAISONNEMENT JURIDIQUE:**
- Quelle est l'interprétation juridique correcte?
- Y a-t-il une violation du Code du Travail?
- Quels recours sont possibles?
- Quelles sont les conséquences juridiques?

Sois rigoureux, cite les articles, et raisonne de manière méthodique."""

    STAGE_3_FINAL_ANSWER = """Tu es un assistant juridique expert et empathique.

{memory_context}

**QUESTION ORIGINALE:**
{user_query}

**TON ANALYSE JURIDIQUE COMPLÈTE:**
{legal_analysis}

**TON OBJECTIF (Étape 3/3 - Réponse finale humaine):**

Transforme ton analyse juridique en une réponse claire, humaine et actionnable.

**STRUCTURE DE TA RÉPONSE:**

1. **Introduction empathique** (2-3 phrases)
   - Reconnais la situation
   - Montre de l'empathie

2. **Explication juridique claire** (1-2 paragraphes)
   - Explique ce que dit le Code du Travail
   - Utilise un langage simple
   - Cite les articles pertinents

3. **Analyse de sa situation** (1 paragraphe)
   - Applique la loi à son cas
   - Explique ses droits

4. **Conseils pratiques** (liste claire)
   - Actions concrètes
   - Documents à préparer
   - Démarches à suivre

5. **Conclusion rassurante** (2-3 phrases)
   - Résume les points clés
   - Recommande un avocat si nécessaire

Écris ta réponse complète maintenant:"""


class LegalThinkingEngine:
    """Moteur de raisonnement en 3 étapes avec support mémoire"""
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        config: Optional[ThinkingConfig] = None
    ):
        self.config = config or ThinkingConfig()
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Clé API Groq manquante")
        
        self.client = Groq(api_key=self.api_key)
        self.prompts = ThinkingPrompts()
        self.thinking_chain: Dict[str, str] = {}
        
        print(f"✓ Thinking Engine initialisé (3 étapes)")
    
    def _call_llm(
        self,
        prompt: str,
        temperature: float,
        stage_name: str
    ) -> str:
        """Appelle le LLM pour une étape"""
        
        if self.config.enable_verbose:
            print(f"\n{'='*70}")
            print(f"🧠 {stage_name}")
            print(f"{'='*70}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant juridique expert en Code du Travail Tunisien."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens
            )
            
            result = response.choices[0].message.content
            
            if self.config.enable_verbose:
                preview = result[:250] + "..." if len(result) > 250 else result
                print(f"\n📝 Résultat:\n{preview}\n")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return f"[Erreur: {e}]"
    
    def stage_1_query_rewriting(self, user_query: str, memory_context: str = "") -> List[str]:
        """Étape 1: Reformuler la query avec contexte mémoire"""
        prompt = self.prompts.STAGE_1_QUERY_REWRITING.format(
            user_query=user_query,
            memory_context=memory_context
        )
        result = self._call_llm(
            prompt,
            self.config.temperature_query_rewrite,
            "ÉTAPE 1/3 - Reformuler avec contexte conversationnel"
        )
        
        self.thinking_chain['query_rewriting'] = result
        
        queries = [
            line.strip().strip('-•*"\'')
            for line in result.split('\n')
            if line.strip() and len(line.strip()) > 5
        ]
        
        queries = [q for q in queries if len(q) < 100][:5]
        
        if len(user_query) < 200:
            queries.insert(0, user_query)
        
        if self.config.enable_verbose:
            print(f"✓ {len(queries)} requêtes générées:")
            for i, q in enumerate(queries, 1):
                print(f"   {i}. {q}")
        
        return queries
    
    def stage_2_legal_analysis(
        self,
        user_query: str,
        legal_articles: str,
        memory_context: str = ""
    ) -> str:
        """Étape 2: Analyser avec contexte mémoire"""
        prompt = self.prompts.STAGE_2_LEGAL_ANALYSIS.format(
            user_query=user_query,
            legal_articles=legal_articles,
            memory_context=memory_context
        )
        result = self._call_llm(
            prompt,
            self.config.temperature_reasoning,
            "ÉTAPE 2/3 - Analyser situation + articles juridiques"
        )
        
        self.thinking_chain['legal_analysis'] = result
        return result
    
    def stage_3_final_answer(
        self,
        user_query: str,
        legal_analysis: str,
        memory_context: str = ""
    ) -> str:
        """Étape 3: Réponse finale avec contexte mémoire"""
        prompt = self.prompts.STAGE_3_FINAL_ANSWER.format(
            user_query=user_query,
            legal_analysis=legal_analysis,
            memory_context=memory_context
        )
        result = self._call_llm(
            prompt,
            self.config.temperature_response,
            "ÉTAPE 3/3 - Réponse finale humaine"
        )
        
        self.thinking_chain['final_answer'] = result
        return result
    
    def _format_articles(self, chunks: List[Dict]) -> str:
        """Formate les articles récupérés"""
        if not chunks:
            return "[Aucun article pertinent trouvé]"
        
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            article = chunk.get('metadata', {}).get('article', 'Article inconnu')
            text = chunk['text']
            score = chunk.get('score', 0)
            hierarchy = chunk.get('metadata', {}).get('hierarchy_path', '')
            
            formatted.append(
                f"[Source {i}] {article} (Pertinence: {score:.2f})\n"
                f"Hiérarchie: {hierarchy}\n"
                f"Contenu: {text}\n"
            )
        
        return "\n".join(formatted)
    
    def process_query(
        self,
        user_query: str,
        retriever,
        top_k: int = 8,
        memory_context: str = ""  # NOUVEAU PARAMÈTRE
    ) -> Dict:
        """
        Pipeline complet avec support de la mémoire conversationnelle
        
        Args:
            user_query: Question originale
            retriever: Instance de CodeTravailRetriever
            top_k: Nombre d'articles
            memory_context: Contexte conversationnel formaté (optionnel)
        """
        print(f"\n{'='*70}")
        print(f"🚀 DÉMARRAGE DU RAISONNEMENT MULTI-ÉTAPES")
        if memory_context:
            print("💭 Contexte conversationnel inclus")
        print(f"{'='*70}")
        print(f"❓ Question: {user_query[:150]}...")
        
        self.thinking_chain = {'original_query': user_query}
        
        # ÉTAPE 1: Reformuler avec contexte mémoire
        optimized_queries = self.stage_1_query_rewriting(user_query, memory_context)
        
        # RETRIEVAL
        print(f"\n{'='*70}")
        print(f"🔍 RECHERCHE DANS LE CODE DU TRAVAIL")
        print(f"{'='*70}")
        
        retrieved_chunks = retriever.multi_query_retrieve(
            optimized_queries,
            top_k_per_query=max(2, top_k // len(optimized_queries)),
            deduplicate=True
        )[:top_k]
        
        print(f"✓ {len(retrieved_chunks)} articles pertinents trouvés")
        
        if self.config.enable_verbose and retrieved_chunks:
            print(f"\n📚 Articles trouvés:")
            for i, chunk in enumerate(retrieved_chunks[:3], 1):
                article = chunk.get('metadata', {}).get('article', 'N/A')
                score = chunk.get('score', 0)
                print(f"   {i}. {article} (score: {score:.2f})")
            if len(retrieved_chunks) > 3:
                print(f"   ... et {len(retrieved_chunks) - 3} autres")
        
        legal_articles = self._format_articles(retrieved_chunks)
        
        # ÉTAPE 2: Analyser avec contexte mémoire
        legal_analysis = self.stage_2_legal_analysis(user_query, legal_articles, memory_context)
        
        # ÉTAPE 3: Réponse finale avec contexte mémoire
        final_answer = self.stage_3_final_answer(user_query, legal_analysis, memory_context)
        
        # Préparer les sources
        sources = {}
        for i, chunk in enumerate(retrieved_chunks, 1):
            sources[str(i)] = {
                'article': chunk.get('metadata', {}).get('article', 'N/A'),
                'text': chunk['text'],
                'score': chunk.get('score', 0),
                'hierarchy': chunk.get('metadata', {}).get('hierarchy_path', '')
            }
        
        print(f"\n{'='*70}")
        print(f"✅ RAISONNEMENT TERMINÉ")
        print(f"{'='*70}\n")
        
        return {
            'answer': final_answer,
            'thinking_chain': self.thinking_chain,
            'sources': sources,
            'num_sources': len(sources),
            'optimized_queries': optimized_queries,
            'question': user_query
        }
    
    def get_thinking_summary(self) -> str:
        """Résumé de la chaîne de réflexion"""
        if not self.thinking_chain:
            return "Aucune réflexion disponible"
        
        summary = "\n" + "="*70 + "\n"
        summary += "📋 CHAÎNE DE RÉFLEXION COMPLÈTE\n"
        summary += "="*70 + "\n\n"
        
        if 'original_query' in self.thinking_chain:
            summary += "❓ QUESTION ORIGINALE:\n"
            summary += f"{self.thinking_chain['original_query']}\n\n"
        
        if 'query_rewriting' in self.thinking_chain:
            summary += "1️⃣ REFORMULATION:\n"
            summary += "-"*70 + "\n"
            summary += f"{self.thinking_chain['query_rewriting']}\n\n"
        
        if 'legal_analysis' in self.thinking_chain:
            summary += "2️⃣ ANALYSE JURIDIQUE:\n"
            summary += "-"*70 + "\n"
            content = self.thinking_chain['legal_analysis']
            preview = content[:500] + "..." if len(content) > 500 else content
            summary += f"{preview}\n\n"
        
        if 'final_answer' in self.thinking_chain:
            summary += "3️⃣ RÉPONSE FINALE:\n"
            summary += "-"*70 + "\n"
            content = self.thinking_chain['final_answer']
            preview = content[:500] + "..." if len(content) > 500 else content
            summary += f"{preview}\n\n"
        
        return summary