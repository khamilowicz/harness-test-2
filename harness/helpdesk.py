"""
helpdesk.py — the harness itself.

HelpdeskHarness ties the pieces together:
  1. take a user question,
  2. retrieve relevant facts from the knowledge base,
  3. build the prompt (system prompt + facts + history + question),
  4. call the LLM client,
  5. store the exchange in history and return the answer.

Keeping all of this in one small class makes it easy to see the whole flow.
"""

from __future__ import annotations

from .llm_client import LLMClient, Message, default_client
from .prompts import build_messages


class HelpdeskHarness:
    """A minimal conversational helpdesk assistant."""

    def __init__(self, client: LLMClient | None = None, max_history: int = 20):
        # If no client is passed in, pick one automatically (real model if
        # an API key exists, otherwise the offline mock).
        self.client = client or default_client()

        # Conversation history as a list of messages. We cap its length so
        # long chats don't grow the prompt forever (which costs tokens and
        # can push important context out of the model's window).
        self.history: list[Message] = []
        self.max_history = max_history

    def ask(self, question: str) -> str:
        """Ask one question and get an answer, remembering the exchange."""
        # 1-2. Build the full prompt: system instructions, retrieved facts,
        #      past conversation, and the new question.
        messages = build_messages(question, self.history)

        # 3. Call the model. temperature=0.2 keeps answers fairly focused
        #    and consistent, which is what you want for a helpdesk bot.
        answer = self.client.complete(messages, temperature=0.2)

        # 4. Record both sides of the exchange so the next question has
        #    context (e.g. follow-ups like "and how long does it take?").
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        # Trim history to the most recent `max_history` messages.
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return answer

    def reset(self) -> None:
        """Forget the conversation (start a fresh ticket, basically)."""
        self.history.clear()
