"""Turn a follow-up question into a standalone one, for code search.

Search sees only the question, so "why does it do that?" finds nothing useful. When a question
looks like it leans on earlier turns, one extra model call rewrites it; otherwise no call is made.
"""

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from codeatlas.services.memory.conversation import Turn, format_history

logger = logging.getLogger(__name__)

_REFERRING_WORDS = {
    "it", "its", "it's", "that", "this", "these", "those", "they", "them", "their",
    "he", "she", "one", "ones", "same", "above", "previous", "earlier", "former", "latter",
    "instead", "else", "again", "also",
}  # fmt: skip
_OPENERS = {"and", "but", "so", "or", "then"}
# "this repo" and similar are standalone; they don't point back at the conversation.
_SELF_CONTAINED = re.compile(r"\bthis (repo|repository|project|codebase|code base|library|package)\b")

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's last question so it can be understood without the conversation. "
            "Replace words like 'it' or 'that' with what they refer to. Keep it short. "
            "Return only the rewritten question.",
        ),
        ("human", "Conversation:\n{history}\n\nLast question: {question}"),
    ]
)


def needs_rewrite(question: str, history: list[Turn]) -> bool:
    if not history:
        return False
    words = re.findall(r"[a-z']+", question.lower())
    if not words:
        return False
    # Very short questions ("why?", "and the tests?") only make sense with the conversation.
    if len(words) <= 3 or words[0] in _OPENERS and len(words) <= 5:
        return True
    if words[:2] in (["what", "about"], ["how", "about"]):
        return True
    referring = re.findall(r"[a-z']+", _SELF_CONTAINED.sub(" ", question.lower()))
    return any(w in _REFERRING_WORDS for w in referring)


def rewrite_question(llm: BaseChatModel, question: str, history: list[Turn]) -> str:
    """The standalone question, or the original if the model fails or answers with something odd."""
    try:
        response = (_PROMPT | llm).invoke({"history": format_history(history), "question": question})
    except Exception as exc:
        logger.warning("Follow-up rewrite failed, searching with the original question (%s)", type(exc).__name__)
        return question
    rewritten = str(response.content).strip().strip('"').strip()
    if not rewritten or len(rewritten) > 4 * len(question) + 200:
        return question
    return rewritten
