"""Check the LLM key in .env. Prints the detected provider and whether one test call
works; never prints any part of the key."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simsearch import config  # noqa: E402
from simsearch.llm import LLMError, get_llm  # noqa: E402

if not config.LLM_API_KEY:
    sys.exit("No key found. Open .env in the project folder and paste your key after LLM_API_KEY=")
print(f"Key found. Detected provider: {config.LLM_PROVIDER}")
llm = get_llm()
if llm is None:
    sys.exit("The provider could not be detected from the key. Add LLM_PROVIDER=groq (or gemini) to .env")
t = time.perf_counter()
try:
    llm.complete_json("Reply with a JSON object only.", 'Return {"ok": true}.')
except LLMError as e:
    sys.exit(f"The key did not work with {llm.name}: {e}")
print(f"OK: {llm.name} answered in {(time.perf_counter() - t) * 1000:.0f} ms. The search parser will use it.")
