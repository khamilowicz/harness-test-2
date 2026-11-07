"""
llm_client.py — the "talking to the model" layer.

A *harness* is the code around an LLM: it builds prompts, keeps history,
and decides what to do with the model's answer. This module is the single
place where we actually call a model, so everything else stays model-agnostic.

We define a small interface (LLMClient) with one method, `complete`, and two
implementations:

  1. MockLLMClient  — a fake model that answers from canned rules. Lets the
                      whole harness run offline, with no API key, which is
                      perfect for learning and testing.
  2. OpenAIClient  — a real client for OpenAI-compatible APIs. It is included
                      so you can see how a real call looks, but it is only
                      used if you explicitly pass an API key.
"""

from __future__ import annotations

import os
from typing import Protocol

# A "message" is one turn in a conversation. Roles:
#   "system"    — instructions that steer the model's behavior
#   "user"      — something the end user said
#   "assistant" — something the model said previously
# Messages are plain dicts, which is the exact shape most LLM APIs expect.
Message = dict[str, str]


class LLMClient(Protocol):
    """The minimal contract every model client must satisfy.

    Using a Protocol (instead of a base class) means any object with a
    `complete` method qualifies — no inheritance required. That makes it
    trivial to swap in a mock during tests.
    """

    def complete(self, messages: list[Message], temperature: float = 0.2) -> str:
        """Given the conversation so far, return the model's next reply."""
        ...


class MockLLMClient:
    """A fake LLM that answers helpdesk questions from simple rules.

    It never touches the network. Instead it looks at the last user message
    and picks a canned answer using keyword matching. This is enough to
    exercise the whole harness (prompts, history, retrieval) end to end.
    """

    # A few canned answers keyed by keywords found in the user's question.
    # In a real system the model would generate these; here we hard-code them.
    CANNED_ANSWERS = [
        (
            ["password", "reset", "forgot"],
            "To reset your password, go to the login page and click "
            "'Forgot password'. You will receive an email with a reset link.",
        ),
        (
            ["vpn", "connect"],
            "To connect to the VPN, open the VPN client, choose the "
            "'Corporate' profile, and sign in with your work credentials.",
        ),
        (
            ["refund"],
            "Refunds are processed within 5-10 business days. You can check "
            "the status in your account under 'Orders' > 'Refunds'.",
        ),
        (
            ["hours", "open", "closing", "opening"],
            "Our support team is available Monday to Friday, 9:00-18:00.",
        ),
    ]

    # Fallback answer when no keyword matches.
    FALLBACK = (
        "I'm a mock assistant, so I don't have a real answer for that yet. "
        "Try asking about passwords, VPN, refunds, or opening hours."
    )

    def complete(self, messages: list[Message], temperature: float = 0.2) -> str:
        # Find the most recent thing the user said; that's what we answer.
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        lowered = last_user_msg.lower()

        # Return the first canned answer where ANY keyword appears. Using
        # "any" (not "all") matters: a real question like "how do I reset
        # my password?" won't contain every keyword, so requiring all of
        # them would miss it and fall through to the fallback.
        for keywords, answer in self.CANNED_ANSWERS:
            if any(k in lowered for k in keywords):
                return answer
        return self.FALLBACK


class OpenAIClient:
    """A real client for OpenAI-compatible chat APIs.

    This shows the shape of a real call: you POST the message list to the
    /chat/completions endpoint and read back the assistant message. We use
    only the standard library (urllib) to avoid extra dependencies; in a
    real project you would use the official `openai` SDK.

    The API key is read from the OPENAI_API_KEY environment variable, never
    hard-coded, so secrets don't end up in source control.
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Either export it, or use "
                "MockLLMClient for offline runs."
            )

    def complete(self, messages: list[Message], temperature: float = 0.2) -> str:
        import json
        import urllib.request

        # Build the request body expected by chat-completion APIs.
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": temperature}
        ).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))

        # The reply lives at choices[0].message.content in the API response.
        return body["choices"][0]["message"]["content"]


def default_client() -> LLMClient:
    """Pick which client to use based on the environment.

    If OPENAI_API_KEY is set we use the real model; otherwise we fall back
    to the mock so the harness always works, even offline.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient()
    return MockLLMClient()
