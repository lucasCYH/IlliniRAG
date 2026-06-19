# tests/test_guardrail.py

import unittest
from unittest.mock import MagicMock, patch
from step3_query import parse_faithfulness_score, evaluate_faithfulness_online, execute_query_with_guardrail
from langchain_core.documents import Document

class TestGuardrailSystem(unittest.TestCase):

    def test_parse_faithfulness_score_explicit(self):
        # Case 1: Standard format
        text = "Statements Analysis:\n- Claim 1: Supported\n- Claim 2: Supported\n\nFaithfulness Score: 5"
        self.assertEqual(parse_faithfulness_score(text), 5)
        
        # Case 2: Score in the middle or other format
        text = "Statements Analysis:\n- Claim 1: Supported\n- Claim 2: Supported\n\nScore: 4"
        self.assertEqual(parse_faithfulness_score(text), 4)

        # Case 3: Mixed numbers in text but clear "Score" line
        text = "1. First statement is true.\n2. Second statement is true.\n\nScore: 2\n"
        self.assertEqual(parse_faithfulness_score(text), 2)

    def test_parse_faithfulness_score_fallback(self):
        # Case 4: No "Score:" prefix, fallback scans bottom up
        text = "Some random text explaining that the score should be 3.\n3"
        self.assertEqual(parse_faithfulness_score(text), 3)

        # Case 5: No score at all, fallback to 3
        text = "This response has no digits at all."
        self.assertEqual(parse_faithfulness_score(text), 3)

    @patch('langchain_community.llms.Ollama')
    def test_evaluate_faithfulness_online_success(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.return_value = "Here is the analysis:\n- Statement 1: Supported\n\nFaithfulness Score: 5"
        mock_llm.invoke.return_value = "Here is the analysis:\n- Statement 1: Supported\n\nFaithfulness Score: 5"
        
        score, analysis = evaluate_faithfulness_online(mock_llm, "Some context", "Some answer")
        self.assertEqual(score, 5)
        self.assertIn("Faithfulness Score: 5", analysis)

    @patch('langchain_community.llms.Ollama')
    def test_execute_query_with_guardrail_pass(self, mock_llm_class):
        # Setup mock RAG chain response
        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = {
            "answer": "This is a faithful answer supported by context.",
            "context": [Document(page_content="This is a faithful answer supported by context.")]
        }

        # Setup mock LLM that scores the answer as 5/5 (faithful)
        mock_llm = MagicMock()
        mock_llm.return_value = "Statements Analysis:\n- Claim: Supported\n\nFaithfulness Score: 5"
        mock_llm.invoke.return_value = "Statements Analysis:\n- Claim: Supported\n\nFaithfulness Score: 5"

        result = execute_query_with_guardrail(mock_rag_chain, mock_llm, "test query", [])
        
        self.assertEqual(result["guardrail_score"], 5)
        self.assertFalse(result["guardrail_breached"])
        self.assertEqual(result["answer"], "This is a faithful answer supported by context.")

    @patch('langchain_community.llms.Ollama')
    def test_execute_query_with_guardrail_breached(self, mock_llm_class):
        # Setup mock RAG chain response
        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = {
            "answer": "This is an unfaithful answer that hallucinates details.",
            "context": [Document(page_content="This is a simple unrelated topic.")]
        }

        # Setup mock LLM that scores the answer as 2/5 (hallucinated)
        mock_llm = MagicMock()
        mock_llm.return_value = "Statements Analysis:\n- Claim: Hallucination\n\nFaithfulness Score: 2"
        mock_llm.invoke.return_value = "Statements Analysis:\n- Claim: Hallucination\n\nFaithfulness Score: 2"

        result = execute_query_with_guardrail(mock_rag_chain, mock_llm, "test query", [])
        
        self.assertEqual(result["guardrail_score"], 2)
        self.assertTrue(result["guardrail_breached"])
        self.assertIn("[ONLINE GUARDRAIL BREACHED - FALLBACK ACTIVATED]", result["answer"])
        self.assertIn("faithfulness score: 2/5", result["answer"])

if __name__ == '__main__':
    unittest.main()
