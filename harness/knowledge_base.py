"""
knowledge_base.py — a tiny "retrieval" store for helpdesk facts.

Real assistants often use RAG (Retrieval-Augmented Generation): before
answering, they look up relevant documents and include them in the prompt,
so the model answers from real facts instead of guessing.

Here the knowledge base is just a list of (keywords, fact) pairs and the
retrieval is simple keyword overlap scoring. That keeps the idea without the
complexity of vector databases and embeddings — perfect for learning.
"""

from __future__ import annotations

# Each entry: a tuple of keywords that identify the topic, plus the fact
# text we want to show the model when the topic comes up.
FACTS: list[tuple[list[str], str]] = [
    (
        ["password", "reset", "login", "credentials"],
        "Password resets are done from the login page via 'Forgot password'; "
        "the reset link expires after 30 minutes.",
    ),
    (
        ["vpn", "remote", "network"],
        "Remote access uses the corporate VPN client with the 'Corporate' "
        "profile; MFA is required on first connect each day.",
    ),
    (
        ["refund", "money back", "return"],
        "Refunds are issued to the original payment method within 5-10 "
        "business days and can be tracked under Orders > Refunds.",
    ),
    (
        ["hours", "open", "available", "support"],
        "Human support is available Monday-Friday 9:00-18:00; outside those "
        "hours the assistant handles common requests automatically.",
    ),
    (
        ["invoice", "billing", "charged", "receipt"],
        "Invoices and receipts are available in the account portal under "
        "Billing > Invoices; VAT details are shown on each invoice.",
    ),
]


def retrieve(question: str, top_k: int = 2) -> list[str]:
    """Return up to `top_k` facts most relevant to the question.

    Scoring: for each fact, count how many of its keywords appear in the
    (lowercased) question. Facts with a score above zero are returned,
    best first. This is deliberately simple — no embeddings, no vectors.
    """
    lowered = question.lower()
    scored: list[tuple[int, str]] = []
    for keywords, fact in FACTS:
        score = sum(1 for k in keywords if k in lowered)
        if score > 0:
            scored.append((score, fact))

    # Sort by score descending; ties keep their original order (stable sort).
    scored.sort(key=lambda item: item[0], reverse=True)
    return [fact for _, fact in scored[:top_k]]
