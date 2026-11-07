"""
main.py — a simple command-line chat with the helpdesk assistant.

Run it with:
    python3 main.py

Then type questions like "How do I reset my password?" and the assistant
answers. Type "quit" (or press Ctrl+C / Ctrl+D) to exit, and "reset" to
forget the conversation so far.

By default this uses the offline MockLLMClient, so it works with no setup.
To talk to a real model instead, set the OPENAI_API_KEY environment variable
first; the harness will pick the real client automatically.
"""

from harness import HelpdeskHarness


def main() -> None:
    harness = HelpdeskHarness()
    print("Helpdesk assistant ready. Type 'quit' to exit, 'reset' to start over.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D or Ctrl+C ends the session cleanly.
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if question.lower() == "reset":
            harness.reset()
            print("Conversation reset.")
            continue
        answer = harness.ask(question)
        print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()
