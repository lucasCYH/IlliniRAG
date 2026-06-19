# backend/compression.py

import os
from typing import Sequence, Optional, Any
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import Field, PrivateAttr
import torch

class LLMLinguaDocumentCompressor(BaseDocumentCompressor):
    """
    Production-grade context compressor using LLMLingua.
    Compresses Top-3 documents at the token level by 30-50% while preserving high information entropy.
    """
    model_name: str = Field(default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    rate: float = Field(default=0.5, description="Target compression rate (0.5 means 50% compression)")
    device: str = Field(default="cpu")
    use_llmlingua2: bool = Field(default=True)
    
    _compressor: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Auto-detect best device if default is cpu
        if self.device == "cpu":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                
        print(f"[LLMLingua] Initializing PromptCompressor on device: {self.device} with model: {self.model_name}")
        from llmlingua import PromptCompressor
        self._compressor = PromptCompressor(
            model_name=self.model_name,
            device_map=self.device,
            use_llmlingua2=self.use_llmlingua2
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Any] = None
    ) -> Sequence[Document]:
        if not documents:
            return []

        # We take only top 3 reranked documents as per task specification
        target_docs = list(documents[:3])
        contexts = [doc.page_content for doc in target_docs]

        print(f"[LLMLingua] Compressing {len(target_docs)} documents for query: '{query[:40]}...' at rate {self.rate}")

        try:
            # Run collective compression
            if self.use_llmlingua2:
                compress_result = self._compressor.compress_prompt(
                    context=contexts,
                    instruction="",
                    question=query,
                    rate=self.rate
                )
            else:
                compress_result = self._compressor.compress_prompt(
                    context=contexts,
                    instruction="",
                    question=query,
                    rate=self.rate
                )
                
            compressed_prompt_list = compress_result.get("compressed_prompt_list")
            
            # If the model successfully returned split compressed chunks matching documents count
            if compressed_prompt_list and len(compressed_prompt_list) == len(target_docs):
                compressed_docs = []
                for doc, compressed_text in zip(target_docs, compressed_prompt_list):
                    compressed_docs.append(Document(
                        page_content=compressed_text,
                        metadata=doc.metadata
                    ))
                print(f"[LLMLingua] Collective compression complete. Retained metadata.")
                return compressed_docs
                
            # If collective list output is not supported/returned, fall back to individual compression
            else:
                print("[LLMLingua] Falling back to individual document compression to preserve metadata...")
                compressed_docs = []
                for doc in target_docs:
                    if self.use_llmlingua2:
                        res = self._compressor.compress_prompt(
                            context=[doc.page_content],
                            instruction="",
                            question=query,
                            rate=self.rate
                        )
                    else:
                        res = self._compressor.compress_prompt(
                            context=[doc.page_content],
                            instruction="",
                            question=query,
                            rate=self.rate
                        )
                    compressed_text = res.get("compressed_prompt", doc.page_content)
                    compressed_docs.append(Document(
                        page_content=compressed_text,
                        metadata=doc.metadata
                    ))
                print(f"[LLMLingua] Individual compression complete. Compressing by {self.rate:.0%}")
                return compressed_docs
                
        except Exception as e:
            print(f"[LLMLingua] Error during compression: {e}. Returning original documents.")
            return target_docs
