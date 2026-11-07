# test-harness

A tiny learning project: a **harness around an LLM** for a helpdesk assistant.
It answers common support questions (passwords, VPN, refunds, ...) in a chat
loop, with conversation memory and a small built-in knowledge base.

## Run it

```bash
cd test-harness
python3 main.py
```

No API key or install needed — by default it uses an offline **mock model**
(`MockLLMClient`) that answers from canned rules, so you can play with the
whole flow immediately.

To use a real OpenAI-compatible model instead:

```bash
export OPENAI_API_KEY=sk-...   # never commit this value!
python3 main.py
```

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

## How it fits together

```
main.py                  CLI chat loop (entry point)
harness/
  helpdesk.py            HelpdeskHarness: ask() -> build prompt -> call LLM -> store history
  prompts.py             system prompt + retrieved facts + history -> message list
  knowledge_base.py      tiny keyword-based "retrieval" of helpdesk facts (mini-RAG)
  llm_client.py          LLMClient interface, MockLLMClient (offline), OpenAIClient (real)
tests/
  test_harness.py        offline unit tests using the mock client
```

Flow of one question: `main.py` → `HelpdeskHarness.ask()` → `build_messages()`
(system prompt + knowledge-base facts + history + question) → `LLMClient.complete()`
→ answer is appended to history and returned.

## Ideas to extend it

- Swap the keyword retrieval for real embeddings + a vector store.
- Stream the model's answer token by token.
- Add tools (e.g. "create a ticket") the model can call.
- Log every prompt/answer pair to a file to study what the harness sends.
