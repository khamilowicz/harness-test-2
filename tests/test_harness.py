"""
tests/test_harness.py — a few sanity tests for the harness.

Run with:
    python3 -m unittest discover -s tests -v
or simply:
    python3 tests/test_harness.py

These use the mock client, so they run offline and deterministically.
"""

import unittest

from harness import HelpdeskHarness
from harness.knowledge_base import retrieve
from harness.llm_client import MockLLMClient
from harness.prompts import build_messages


class KnowledgeBaseTests(unittest.TestCase):
    def test_retrieve_finds_relevant_fact(self):
        facts = retrieve("how do I reset my password?")
        self.assertTrue(any("password" in f.lower() for f in facts))

    def test_retrieve_returns_nothing_for_unknown_topic(self):
        self.assertEqual(retrieve("what is the airspeed of a swallow?"), [])


class HarnessTests(unittest.TestCase):
    def setUp(self):
        # Every test gets a fresh harness with the offline mock model.
        self.harness = HelpdeskHarness(client=MockLLMClient())

    def test_known_question_gets_canned_answer(self):
        answer = self.harness.ask("I forgot my password, how do I reset it?")
        self.assertIn("Forgot password", answer)

    def test_unknown_question_gets_fallback(self):
        answer = self.harness.ask("do you like pineapple on pizza?")
        self.assertIn("mock assistant", answer)

    def test_history_is_recorded(self):
        self.harness.ask("how do I connect to the vpn?")
        # One user message + one assistant reply were stored.
        self.assertEqual(len(self.harness.history), 2)
        self.assertEqual(self.harness.history[0]["role"], "user")
        self.assertEqual(self.harness.history[1]["role"], "assistant")

    def test_reset_clears_history(self):
        self.harness.ask("hello there")
        self.harness.reset()
        self.assertEqual(self.harness.history, [])

    def test_prompt_contains_system_prompt_and_question(self):
        messages = build_messages("vpn help please", history=[])
        roles = [m["role"] for m in messages]
        self.assertEqual(roles[0], "system")          # system prompt first
        self.assertEqual(roles[-1], "user")           # question last
        self.assertEqual(messages[-1]["content"], "vpn help please")


if __name__ == "__main__":
    unittest.main()
