# step3_query.py

import re
from typing import Dict, Any, Tuple, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Lightweight LLM-as-a-judge prompt for online faithfulness verification
GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI guardrail system verifying RAG answer correctness.\n"
        "Your task is to analyze the generated Answer against the provided Context and evaluate its Faithfulness.\n\n"
        "Instructions:\n"
        "1. Extract all factual statements/claims made in the generated Answer.\n"
        "2. For each statement, verify if it is directly supported (entailed) by the Context.\n"
        "3. Output your statement-by-statement analysis.\n"
        "4. Assign a final Faithfulness score from 0 to 5 based on the proportion of supported statements:\n"
        "   - 5: The answer is entirely supported by the context with zero hallucinations.\n"
        "   - 4: The answer is mostly supported, but contains minor extrapolation or unsupported details.\n"
        "   - 0-3: The answer contains major hallucinations or contradicts the context.\n\n"
        "Provide your output exactly in this format:\n"
        "Statements Analysis:\n"
        "- [Statement]: [Supported / Hallucination] (Reason)\n"
        "...\n"
        "Faithfulness Score: [Score]\n"
        "Where [Score] is a single integer from 0 to 5."
    )),
    ("human", "Context:\n{context}\n\nGenerated Answer:\n{answer}\n\nPerform the faithfulness evaluation:")
])

def parse_faithfulness_score(response_text: str) -> int:
    """
    Parses the faithfulness score from the judge LLM response.
    Looks for the pattern 'Faithfulness Score: X' or 'Score: X' where X is a digit 0-5.
    If multiple matches are found, it takes the last one to avoid matching statement list indexes.
    """
    # Look for "Faithfulness Score: X" or "Score: X"
    matches = re.findall(r"(?:Faithfulness\s+)?Score:\s*(\d)", response_text, re.IGNORECASE)
    if matches:
        score = int(matches[-1])
        if 0 <= score <= 5:
            return score

    # Fallback: scan lines from bottom to top for any digit
    for line in reversed(response_text.splitlines()):
        # Exclude statement lists (lines starting with - or containing statement numbers)
        if line.strip().startswith("-") or "statement" in line.lower():
            continue
        digits = re.findall(r"\b([0-5])\b", line)
        if digits:
            return int(digits[-1])

    # Default fallback if parsing fails
    return 3

def evaluate_faithfulness_online(llm: Ollama, context: str, answer: str) -> Tuple[int, str]:
    """
    Evaluates faithfulness of an answer against the context.
    Returns (score, full_llm_response).
    """
    try:
        # Limit context to avoid context window overflow (8000 characters)
        truncated_context = context[:8000]
        chain = GUARDRAIL_PROMPT | llm
        response = chain.invoke({"context": truncated_context, "answer": answer})
        
        # Parse the score
        score = parse_faithfulness_score(response)
        return score, response
    except Exception as e:
        print(f"Error calling online judge LLM: {e}")
        return 3, f"Error calling online judge LLM: {e}"

def execute_query_with_guardrail(
    rag_chain: Any, 
    llm: Ollama,
    user_input: str, 
    chat_history: List[Any]
) -> Dict[str, Any]:
    """
    Executes RAG query and runs the online self-RAG guardrail.
    """
    # 1. Invoke raw RAG pipeline
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    raw_answer = response.get("answer", "")
    context_docs = response.get("context", [])
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    # 2. Run the faithfulness guardrail
    score, analysis = evaluate_faithfulness_online(llm, context_text, raw_answer)
    print(f"\n--- ONLINE GUARDRAIL EVALUATION ---")
    print(f"Faithfulness Score: {score}/5")
    print(analysis)
    print(f"-----------------------------------\n")
    
    # 3. Apply Fallback Mechanism if score is below 4.0/5.0
    if score < 4:
        print(f"[ONLINE GUARDRAIL BREACHED - FALLBACK ACTIVATED] Score: {score}/5")
        fallback_answer = (
            "⚠️ **[ONLINE GUARDRAIL BREACHED - FALLBACK ACTIVATED]**\n\n"
            "We apologize, but the generated answer did not pass our real-time faithfulness check "
            f"(faithfulness score: {score}/5, which is below our threshold of 4/5). "
            "To prevent presenting hallucinated or inaccurate information, we have blocked this response."
        )
        response["answer"] = fallback_answer
        response["guardrail_breached"] = True
    else:
        response["guardrail_breached"] = False
        
    response["guardrail_score"] = score
    response["guardrail_analysis"] = analysis
    
    return response
