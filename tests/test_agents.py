# tests/test_agents.py

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from langchain_core.documents import Document

# Mock out streamlit and langchain models to prevent network calls and heavy loads during test import
with patch('langchain_community.embeddings.HuggingFaceEmbeddings') as mock_embeddings_class, \
     patch('langchain_community.llms.Ollama') as mock_ollama_class, \
     patch('langchain_community.vectorstores.Chroma') as mock_chroma_class:
     
    from backend.agents import EmbeddingClassifier, AgentRouter, GlobalAgent, NeedleAgent
    from backend import config
    from backend.summary_hierarchical import generate_summary, generate_hierarchical_summary


class TestAgentRouterAndClassifier(unittest.TestCase):

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    def setUp(self, mock_emb_class):
        # Setup mock embeddings
        self.mock_emb = MagicMock()
        # Mock embed_documents to return a matrix of shape (n_examples, 384)
        # We make the mock centroids distinct by returning different values
        def side_effect_embed_docs(texts):
            # If they contain global keyword, make them tend positive, else negative
            embeds = []
            for t in texts:
                val = 0.5 if "global" in t.lower() or "summary" in t.lower() or "大綱" in t else -0.5
                embeds.append([val] * 384)
            return embeds
            
        self.mock_emb.embed_documents.side_effect = side_effect_embed_docs
        
        def side_effect_embed_query(text):
            val = 0.5 if "summary" in text.lower() or "大綱" in text else -0.5
            return [val] * 384
            
        self.mock_emb.embed_query.side_effect = side_effect_embed_query
        mock_emb_class.return_value = self.mock_emb

        # Initialize Classifier
        self.classifier = EmbeddingClassifier()

    def test_classifier_routing_global_by_keyword(self):
        # "summary" is in global keywords (config.GLOBAL_KEYWORDS)
        # Should route to global directly
        with patch('backend.config.GLOBAL_KEYWORDS', ["summary"]):
            query = "Give me a summary of the document"
            # We bypass semantic check if keyword matches
            router = AgentRouter(MagicMock())
            agent, reason = router.route(query)
            self.assertEqual(agent.name, "GlobalAgent")
            self.assertIn("keyword match", reason)

    def test_classifier_routing_needle(self):
        # Specific query should route to NeedleAgent
        query = "What is the exact room number of the CS department?"
        router = AgentRouter(MagicMock())
        # Mock classifer to return "needle"
        mock_clf = MagicMock()
        mock_clf.classify.return_value = "needle"
        router.classifier = mock_clf
        
        agent, reason = router.route(query)
        self.assertEqual(agent.name, "NeedleAgent")
        self.assertIn("classifier match", reason)

    def test_classifier_routing_forced(self):
        # Force mode in route
        router = AgentRouter(MagicMock())
        agent, reason = router.route("any query", force_mode="global")
        self.assertEqual(agent.name, "GlobalAgent")
        self.assertIn("forced via UI", reason)


class TestHierarchicalSummary(unittest.TestCase):

    @patch('backend.summary_hierarchical.Ollama')
    def test_generate_summary_success(self, mock_ollama_class):
        # Mock Ollama instance
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "This is a mock summary response."
        mock_ollama_class.return_value = mock_llm
        
        summary = generate_summary("Some text content to summarize", "chapter", "Introduction")
        self.assertEqual(summary, "This is a mock summary response.")
        mock_llm.invoke.assert_called_once()

    @patch('backend.summary_hierarchical.Ollama')
    def test_generate_summary_fallback(self, mock_ollama_class):
        # Simulate LLM crash
        mock_ollama_class.side_effect = Exception("Ollama connection failed")
        
        text = "This is a very long text containing important academic details about research."
        summary = generate_summary(text, "chapter", "Introduction")
        # Should fallback to truncation
        self.assertIn("[Fallback Summary for chapter 'Introduction']", summary)

    @patch('backend.summary_hierarchical._get_chapter_store')
    @patch('backend.summary_hierarchical._get_section_store')
    @patch('backend.summary_hierarchical.generate_chapter_and_sections_summaries')
    def test_generate_hierarchical_summary(self, mock_gen_batch, mock_sec_store_fn, mock_ch_store_fn):
        # Mock Chroma stores
        mock_ch_store = MagicMock()
        mock_sec_store = MagicMock()
        mock_ch_store_fn.return_value = mock_ch_store
        mock_sec_store_fn.return_value = mock_sec_store
        
        # Mock batch generator to return chapter and section summaries
        def side_effect_batch(ch_name, ch_text, ch_sections):
            return {
                "chapter_summary": f"Mocked Chapter Summary for {ch_name}",
                "sections": {sec: f"Mocked Section Summary for {sec}" for sec in ch_sections}
            }
        mock_gen_batch.side_effect = side_effect_batch

        # Mock Docs
        docs = [
            Document(page_content="Chapter 1 content line 1", metadata={"Header 1": "Book Title", "Header 2": "Chapter One", "Header 3": "Section A"}),
            Document(page_content="Chapter 1 content line 2", metadata={"Header 1": "Book Title", "Header 2": "Chapter One", "Header 3": "Section A"}),
            Document(page_content="Chapter 2 content line 1", metadata={"Header 1": "Book Title", "Header 2": "Chapter Two", "Header 3": "Section B"}),
        ]
        
        progress_calls = []
        def progress_callback(percent, text):
            progress_calls.append((percent, text))
            
        generate_hierarchical_summary(docs, doc_id=42, filename="test.pdf", progress_callback=progress_callback)
        
        # Verify chapter store saved 2 documents
        mock_ch_store.add_documents.assert_called_once()
        saved_ch_docs = mock_ch_store.add_documents.call_args[0][0]
        self.assertEqual(len(saved_ch_docs), 2)
        self.assertEqual(saved_ch_docs[0].metadata["chapter"], "Chapter One")
        self.assertEqual(saved_ch_docs[1].metadata["chapter"], "Chapter Two")

        # Verify section store saved 2 documents (Section A and Section B)
        mock_sec_store.add_documents.assert_called_once()
        saved_sec_docs = mock_sec_store.add_documents.call_args[0][0]
        self.assertEqual(len(saved_sec_docs), 2)
        
        # Verify progress callback was triggered
        self.assertTrue(len(progress_calls) > 0)
        self.assertEqual(progress_calls[-1][0], 100)


if __name__ == '__main__':
    unittest.main()
