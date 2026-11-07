"""
prompts.py — building the messages we send to the LLM.

The system prompt is the most important part of a harness: it tells the
model who it is, what it may and may not do, and what facts it should use.
Everything else (history, retrieved facts) is appended after it.
"""

from __future__ import annotations

from .knowledge_base import retrieve
from .llm_client import Message

# The base system prompt. Written in plain language; the model follows it
# because it appears first with role="system".
SYSTEM_PROMPT = (
    "You are HelpBot, a friendly helpdesk assistant for a small company. "
    "Answer questions clearly and concisely. Use only the facts provided "
    "below when they are relevant; if you don't know something, say so "
    "instead of guessing. Never invent prices, policies, or links."
)


def build_messages(question: str, history: list[Message], top_k: int = 2) -> list[Message]:
    """Assemble the full message list for one turn of the conversation.

    Order matters: system prompt first, then any retrieved facts (also as a
    system message, so the model treats them as ground truth), then the
    conversation history, and finally the new user question.
    """
    messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Look up relevant facts and inject them, clearly delimited, so the
    # model can quote them but knows where they came from.
    facts = retrieve(question, top_k=top_k)
    if facts:
        facts_block = "\n".join(f"- {fact}" for fact in facts)
        messages.append(
            {
                "role": "system",
                "content": f"Relevant facts from the knowledge base:\n{facts_block}",
            }
        )

    # Replay the earlier conversation so the model has context.
    messages.extend(history)

    # The new question is always the last message.
    messages.append({"role": "user", "content": question})
    return messages
