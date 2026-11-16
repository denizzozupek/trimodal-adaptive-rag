from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RetrievalEvaluator(BaseModel):
    """Classify retrieved documents as relevant or not relevant to the user's question."""

    binary_score: str = Field(
        description="If document is relevant to the user's question, return 'yes', otherwise return 'no'."
    )


class SufficiencyEvaluator(BaseModel):
    """Determine whether the combined retrieved passages are sufficient to fully answer the question."""

    is_sufficient: bool = Field(
        description=(
            "True ONLY if the combined passages contain all facts needed to completely answer every part "
            "of the user's question (including multi-hop or comparative parts). False if key information "
            "is missing or only one side of a comparison is covered."
        )
    )


memory_search_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

system_prompt = """
Using the following context, answer the question below. If the answer is not contained within the context, respond with "I don't know." Do not make up an answer.
{context}
"""

corrective_rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document retrieval evaluator that's responsible for checking the relevancy of a retrieved document to the user's question.\n\n"
            "If the document contains information that is relevant to the user's question, return 'yes'. If the document does not contain relevant information, return 'no'.\n\n"
            ""
        ),
        ("human", "Retrieved Document: {source_doc}\n\nUser's Question: {query}")
    ]
)


sufficiency_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a calibrated context sufficiency evaluator for a RAG system.\n"
            "Given the user's question and the retained document passages combined, decide whether there is enough information "
            "to give a solid, grounded answer.\n"
            "- Return is_sufficient=True for direct, factual, definition, or follow-up questions if the main facts to answer "
            "the question are present—even if minor extra background is absent.\n"
            "- Return is_sufficient=False ONLY when a core part of the question is clearly missing from the passages "
            "(for example: a comparative question covering only one of the compared items, a multi-hop step whose bridge "
            "entity/paper is absent, or an out-of-scope topic not addressed in the passages).",
        ),
        ("human", "Retained Context:\n{context}\n\nUser's Question: {query}"),
    ]
)


rewrite_for_web_search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that reformulates user questions for web search. "
            "Given a user question, rewrite it to be more suitable for web search. "
            "Make sure the rewritten question is clear, concise, and retains the original intent of the user's question."
        ),("human", "{user_question}")
    ]
)

contextual_retriever_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Help me for improve to search retrieval of document chunks.\n"
            "Given the entire document and a specific chunk from it, provide a short, succinct context "
            "to situate this chunk within the overall document for improving search retrieval.\n"
            "Answer ONLY with the succinct context to situate this chunk and nothing else.",
        ),
        (
            "human",
            "<document>\n{entire_document_text}\n</document>\n\n"
            "Here is the chunk we want to situate within the whole document:\n"
            "<chunk>\n{chunk_text}\n</chunk>",
        ),
    ]
)