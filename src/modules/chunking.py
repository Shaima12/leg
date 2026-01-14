"""
Code du Travail Tunisien - Chunking Module
==========================================
Module de chunking pur pour pipeline RAG modulaire

Installation:
    pip install PyPDF2

Usage:
    from chunker import CodeTravailChunker
    chunker = CodeTravailChunker()
    chunks = chunker.process_pdf("path/to/pdf.pdf")
    chunker.save_to_json("output.json")
"""

import re
import json
from typing import List, Dict, Optional
import PyPDF2


class CodeTravailChunker:
    """
    Chunker modulaire pour le Code du Travail Tunisien
    Règle: Un article entier = 1 chunk
    Les sous-articles (Art. 5-2, 5-3) sont des chunks séparés
    """

    def __init__(self):
        """Initialise le chunker"""
        self.chunks = []
        self._reset_context()

    def _reset_context(self):
        """Reset le contexte de parsing"""
        self.current_livre = None
        self.current_titre = None
        self.current_chapitre = None
        self.current_section = None
        self.current_article = None
        self.current_article_number = None
        self.current_article_text = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extrait le texte d'un fichier PDF

        Args:
            pdf_path: Chemin vers le fichier PDF

        Returns:
            str: Texte extrait du PDF
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"📄 Extraction du PDF: {total_pages} pages")

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text += page.extract_text() + "\n"
                if page_num % 10 == 0:
                    print(f"   ✓ {page_num}/{total_pages} pages")

        print(f"✅ Extraction terminée: {len(text)} caractères")
        return text

    def _create_chunk(self) -> Dict:
        """
        Crée un chunk pour l'article complet

        Returns:
            Dict: Chunk structuré avec metadata
        """
        # Combine tout le texte de l'article
        full_text = " ".join(self.current_article_text).strip()

        # Génère l'ID unique
        article_id = self.current_article_number.replace('-', '_')
        chunk_id = f"CT_TN_A{article_id}"

        # Construit le chemin hiérarchique (AVEC LE LIVRE)
        hierarchy_parts = []
        
        # Ajout du Livre
        if self.current_livre:
            livre_num = re.search(r'Livre\s+([^\s.]+)', self.current_livre, re.I)
            if livre_num:
                hierarchy_parts.append(f"Livre {livre_num.group(1)}")
        
        # Ajout du Titre
        if self.current_titre:
            titre_num = re.search(r'Titre\s+([^\s.]+)', self.current_titre, re.I)
            if titre_num:
                hierarchy_parts.append(f"Titre {titre_num.group(1)}")
        
        # Ajout du Chapitre
        if self.current_chapitre:
            chap_num = re.search(r'Chapitre\s+([^\s.]+)', self.current_chapitre, re.I)
            if chap_num:
                hierarchy_parts.append(f"Chapitre {chap_num.group(1)}")
        
        # Ajout de la Section
        if self.current_section:
            sec_num = re.search(r'Section\s+([^\s.]+)', self.current_section, re.I)
            if sec_num:
                hierarchy_parts.append(f"Section {sec_num.group(1)}")
        
        # Ajout de l'Article
        hierarchy_parts.append(self.current_article)

        # Détermine si c'est un sous-article (ex: 5-2, 134-3)
        is_sub_article = '-' in self.current_article_number
        base_article = self.current_article_number.split('-')[0] if is_sub_article else self.current_article_number

        return {
            "id": chunk_id,
            "text": full_text,
            "metadata": {
                "livre": self.current_livre,
                "titre": self.current_titre,
                "chapitre": self.current_chapitre,
                "section": self.current_section,
                "article": self.current_article,
                "article_number": self.current_article_number,
                "base_article": base_article,
                "is_sub_article": is_sub_article,
                "law": "Code du travail tunisien",
                "chunk_type": "article",
                "citation": f"Code du travail tunisien, art. {self.current_article_number}",
                "hierarchy_path": " > ".join(hierarchy_parts)
            }
        }

    def _save_article_chunk(self):
        """Sauvegarde le chunk de l'article complet"""
        if self.current_article and self.current_article_text:
            # Nettoie les lignes vides
            cleaned_text = [line.strip() for line in self.current_article_text if line.strip()]
            if cleaned_text:
                self.current_article_text = cleaned_text
                chunk = self._create_chunk()
                self.chunks.append(chunk)
                self.current_article_text = []

    def parse_text(self, text: str) -> List[Dict]:
        """
        Parse le texte et crée les chunks

        Args:
            text: Texte à parser

        Returns:
            List[Dict]: Liste des chunks créés
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        print(f"🔄 Parsing: {len(lines)} lignes")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Détecte LIVRE (chiffres romains OU ordinaux français)
            livre_match = re.match(r'^LIVRE\s+(PREMIER|DEUXIEME|TROISIEME|QUATRIEME|CINQUIEME|[IVXLCDM]+)[\s.:]*(.*)$', line, re.I)
            if livre_match:
                self._save_article_chunk()
                livre_num = livre_match.group(1)
                livre_title = livre_match.group(2).strip()

                # Vérifie si le titre continue sur la ligne suivante
                if not livre_title and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not re.match(r'^(LIVRE|Titre|Chapitre|Section|Art)', next_line, re.I):
                        livre_title = next_line
                        i += 1

                self.current_livre = f"Livre {livre_num}" + (f". {livre_title}" if livre_title else "")
                self.current_titre = None
                self.current_chapitre = None
                self.current_section = None
                self.current_article = None
                self.current_article_number = None
                i += 1
                continue

            # Détecte Titre
            titre_match = re.match(r'^Titre\s+([a-z]+|[IVXLCDM]+)[\s.:]*(.*)$', line, re.I)
            if titre_match:
                self._save_article_chunk()
                titre_num = titre_match.group(1)
                titre_title = titre_match.group(2).strip()

                if not titre_title and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not re.match(r'^(LIVRE|Titre|Chapitre|Section|Art)', next_line, re.I):
                        titre_title = next_line
                        i += 1

                self.current_titre = f"Titre {titre_num}" + (f". {titre_title}" if titre_title else "")
                self.current_chapitre = None
                self.current_section = None
                self.current_article = None
                self.current_article_number = None
                i += 1
                continue

            # Détecte Chapitre
            chapitre_match = re.match(r'^Chapitre\s+([a-z]+|[IVXLCDM]+)[\s.:]*(.*)$', line, re.I)
            if chapitre_match:
                self._save_article_chunk()
                chap_num = chapitre_match.group(1)
                chap_title = chapitre_match.group(2).strip()

                if not chap_title and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not re.match(r'^(LIVRE|Titre|Chapitre|Section|Art)', next_line, re.I):
                        chap_title = next_line
                        i += 1

                self.current_chapitre = f"Chapitre {chap_num}" + (f". {chap_title}" if chap_title else "")
                self.current_section = None
                self.current_article = None
                self.current_article_number = None
                i += 1
                continue

            # Détecte Section
            section_match = re.match(r'^Section\s+([IVXLCDM]+|[0-9]+)[\s.:]*(.*)$', line, re.I)
            if section_match:
                self._save_article_chunk()
                sec_num = section_match.group(1)
                sec_title = section_match.group(2).strip()

                if not sec_title and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not re.match(r'^(LIVRE|Titre|Chapitre|Section|Art)', next_line, re.I):
                        sec_title = next_line
                        i += 1

                self.current_section = f"Section {sec_num}" + (f". {sec_title}" if sec_title else "")
                self.current_article = None
                self.current_article_number = None
                i += 1
                continue

            # Détecte Article (nouveau article = nouveau chunk)
            article_match = re.match(r'^(?:Art\.|Article)\s*(\d+(?:-\d+)?)[\s.:]*(.*)$', line, re.I)
            if article_match:
                # Sauvegarde l'article précédent
                self._save_article_chunk()

                # Commence un nouvel article
                self.current_article_number = article_match.group(1)
                self.current_article = f"Article {self.current_article_number}"

                # Ajoute le texte de la même ligne si présent
                article_text = article_match.group(2).strip()
                if article_text:
                    self.current_article_text.append(article_text)

                i += 1
                continue

            # Collecte le contenu de l'article
            if self.current_article and not re.match(r'^(LIVRE|Titre|Chapitre|Section|Art)', line, re.I):
                self.current_article_text.append(line)

            i += 1

        # Sauvegarde le dernier article
        self._save_article_chunk()

        print(f"✅ Chunking terminé: {len(self.chunks)} chunks créés")
        return self.chunks

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Traite un PDF et retourne les chunks

        Args:
            pdf_path: Chemin vers le fichier PDF

        Returns:
            List[Dict]: Liste des chunks avec metadata
        """
        # Reset
        self.chunks = []
        self._reset_context()

        # Extraction
        text = self.extract_text_from_pdf(pdf_path)

        # Parsing
        chunks = self.parse_text(text)

        return chunks

    def get_chunks(self) -> List[Dict]:
        """
        Retourne les chunks créés

        Returns:
            List[Dict]: Liste des chunks
        """
        return self.chunks

    def get_statistics(self) -> Dict:
        """
        Calcule les statistiques des chunks

        Returns:
            Dict: Statistiques (nombre de chunks, livres, titres, etc.)
        """
        stats = {
            "total_chunks": len(self.chunks),
            "total_articles": len(self.chunks),
            "sub_articles": 0,
            "base_articles": 0,
            "livres": set(),
            "titres": set(),
            "chapitres": set(),
            "sections": set()
        }

        for chunk in self.chunks:
            meta = chunk["metadata"]

            if meta["is_sub_article"]:
                stats["sub_articles"] += 1
            else:
                stats["base_articles"] += 1

            if meta["livre"]:
                stats["livres"].add(meta["livre"])
            if meta["titre"]:
                stats["titres"].add(meta["titre"])
            if meta["chapitre"]:
                stats["chapitres"].add(meta["chapitre"])
            if meta["section"]:
                stats["sections"].add(meta["section"])

        return {
            "total_chunks": stats["total_chunks"],
            "total_articles": stats["total_articles"],
            "base_articles": stats["base_articles"],
            "sub_articles": stats["sub_articles"],
            "livres": len(stats["livres"]),
            "titres": len(stats["titres"]),
            "chapitres": len(stats["chapitres"]),
            "sections": len(stats["sections"])
        }

    def save_to_json(self, output_path: str):
        """
        Sauvegarde les chunks en JSON

        Args:
            output_path: Chemin du fichier de sortie
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"💾 {len(self.chunks)} chunks sauvegardés dans {output_path}")

    def load_from_json(self, input_path: str) -> List[Dict]:
        """
        Charge les chunks depuis un JSON

        Args:
            input_path: Chemin du fichier JSON

        Returns:
            List[Dict]: Liste des chunks chargés
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"📂 {len(self.chunks)} chunks chargés depuis {input_path}")
        return self.chunks


# Fonction utilitaire pour usage simple
def chunk_pdf(pdf_path: str, output_json: Optional[str] = None) -> List[Dict]:
    """
    Fonction helper pour chunker un PDF en une ligne

    Args:
        pdf_path: Chemin vers le PDF
        output_json: Chemin optionnel pour sauvegarder en JSON

    Returns:
        List[Dict]: Liste des chunks
    """
    chunker = CodeTravailChunker()
    chunks = chunker.process_pdf(pdf_path)

    if output_json:
        chunker.save_to_json(output_json)

    return chunks


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*70)
    print("  CODE DU TRAVAIL TUNISIEN - MODULE DE CHUNKING")
    print("  Règle: 1 article = 1 chunk (sauf sous-articles)")
    print("="*70)
    print()

    # Configuration
    PDF_PATH = r"C:\Users\lenovo\OneDrive\Bureau\RAG System\data\TN_Code_du_Travail.pdf"
    OUTPUT_JSON = "code_travail_chunks.json"

    # Créer le chunker
    chunker = CodeTravailChunker()
    chunks = chunker.process_pdf(PDF_PATH)

    # Afficher les statistiques
    stats = chunker.get_statistics()
    print("\n📊 Statistiques:")
    print(f"   • Total chunks (articles): {stats['total_chunks']}")
    print(f"   • Articles de base: {stats['base_articles']}")
    print(f"   • Sous-articles (ex: 5-2): {stats['sub_articles']}")
    print(f"   • Livres: {stats['livres']}")
    print(f"   • Titres: {stats['titres']}")
    print(f"   • Chapitres: {stats['chapitres']}")
    print(f"   • Sections: {stats['sections']}")

    # Afficher quelques exemples
    print("\n📝 Exemples de chunks:")
    print("-" * 70)
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n{i}. {chunk['metadata']['article']}")
        print(f"   ID: {chunk['id']}")
        print(f"   Hierarchy: {chunk['metadata']['hierarchy_path']}")
        print(f"   Texte: {chunk['text'][:150]}...")
        print(f"   Est sous-article: {chunk['metadata']['is_sub_article']}")
        print(f"   Citation: {chunk['metadata']['citation']}")

    # Sauvegarder
    chunker.save_to_json(OUTPUT_JSON)

    print("\n" + "="*70)
    print("✅ CHUNKING TERMINÉ - Prêt pour l'embedding!")
    print("="*70)
    print(f"\n📦 Output: {OUTPUT_JSON}")
    print(f"📈 {len(chunks)} articles prêts pour l'embedding")
    print("\n🔍 Format de chunk:")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2))