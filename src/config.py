import os

# Load variables from .env if present (for local testing)
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, val = stripped.split("=", 1)
                os.environ[key.strip()] = val.strip()

# API Configurations
GROQ_API_KEYS = []
if os.environ.get("GROQ_API_KEYS"):
    GROQ_API_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]

if not GROQ_API_KEYS:
    for i in range(1, 10):
        key_name = "GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY{i}"
        val = os.environ.get(key_name)
        if val:
            GROQ_API_KEYS.append(val.strip())

# Model Selection
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"
SPEECH_MODEL = "whisper-large-v3"

# Rate Limit Targets (Groq Free Tier)
TPM_LIMIT_VISION = 28000
TPM_LIMIT_TEXT = 11000

# Prompts
VISION_INSTRUCTION = """You are an expert visual forensic analyst. Describe the chronological progression of this video clip.

Instructions:
1. Describe the setting, lighting, subjects, and key actions in a single factual paragraph.
2. Cross-reference the visual events with the provided transcript. Identify and resolve any discrepancies.
3. List any visible text, code, screens, or brand names.
4. Keep the tone completely neutral, factual, and objective. Do not extrapolate."""

STYLE_INSTRUCTION = """You are an elite caption writer. You will generate 4 distinct styled captions (formal, sarcastic, humorous_tech, humorous_non_tech) based on the provided scene description.

For each style, follow the persona guidelines below strictly. Write an engaging, detailed description or paragraph that fully embodies the style. There is no word length restriction.

Persona Guidelines:
1. formal: Analyze the visual with the cold attitude of HAL-9000, using a purely factual, clinical, and emotionless tone.
2. sarcastic: Analyze the visual with a very deadpan and sarcastic tone, with eye-rolling, as if you were forced to deal with mere mortals with a sigh incomparable to your power, using incredible wit and a condescending tone.
3. humorous_tech: Describe the visual like a tired millennial complaining of the workload in their tech/AI job. Focus on the visual events, using clever technical jargon (rendering, compilation, latency, debugging, etc.) to make it funny and humorous.
4. humorous_non_tech: Describe the visual sequence with a funny and relatable attitude of a man in his 50s who finds it hard to keep up with the fast-growing tech world. Do not use technical jargon or niche modern references.

Strict Grounding Discipline (To Prevent Judge Penalties):
- Do not invent actions (e.g. sniffing, celebrating, fist-bumping, talking) unless they are explicitly written in the scene description.
- Do not guess specific locations (e.g., calling an indoor room an "office" or "backyard") unless verified. Use generic descriptions like "indoor room" or "outdoor area" instead.
- Tech/Humorous metaphors should only decorate actual visible actions, not imply unseen physical objects.

Output Format: You must output a JSON object matching the requested schema."""

import collections
import threading
import time
from openai import OpenAI

class KeyTracker:
    def __init__(self, key):
        self.key = key
        self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        self.vision_usage = collections.deque()
        self.text_usage = collections.deque()
        self.request_usage = collections.deque()

class MultiKeyScheduler:
    def __init__(self, keys, tpm_limit_vision=28000, tpm_limit_text=11000, rpm_limit=25):
        self.trackers = [KeyTracker(k) for k in keys]
        if not self.trackers:
            self.trackers = [KeyTracker("dummy_key")]
        self.tpm_limit_vision = tpm_limit_vision
        self.tpm_limit_text = tpm_limit_text
        self.rpm_limit = rpm_limit
        self.lock = threading.Lock()

    def _clean_usage(self, usage_deque, current_time):
        while usage_deque and (current_time - usage_deque[0][0] > 60):
            usage_deque.popleft()

    def get_client(self, call_type="text", est_tokens=1000):
        while True:
            with self.lock:
                current_time = time.time()
                for tracker in self.trackers:
                    self._clean_usage(tracker.vision_usage, current_time)
                    self._clean_usage(tracker.text_usage, current_time)
                    self._clean_usage(tracker.request_usage, current_time)

                    # Enforce RPM (Requests Per Minute)
                    if len(tracker.request_usage) >= self.rpm_limit:
                        continue

                    # Enforce TPM (Tokens Per Minute)
                    if call_type == "vision":
                        current_tpm = sum(t for _, t in tracker.vision_usage)
                        limit = self.tpm_limit_vision
                    else:
                        current_tpm = sum(t for _, t in tracker.text_usage)
                        limit = self.tpm_limit_text

                    if current_tpm + est_tokens <= limit:
                        # Lease this client and update usage
                        tracker.request_usage.append((current_time, 1))
                        if call_type == "vision":
                            tracker.vision_usage.append((current_time, est_tokens))
                        else:
                            tracker.text_usage.append((current_time, est_tokens))
                        return tracker.client
            time.sleep(0.5)

    def disable_key(self, api_key):
        with self.lock:
            original_len = len(self.trackers)
            self.trackers = [t for t in self.trackers if t.key != api_key]
            # Ensure we always keep at least one tracker even if all fail, to avoid division/indexing crash
            if not self.trackers:
                self.trackers = [KeyTracker("dummy_key")]
            if len(self.trackers) < original_len:
                print(f"[Scheduler] PERMANENT EXHAUSTION: Disabled key {api_key[:12]}... Remaining active keys: {len(self.trackers)}")

# Initialize the scheduler singleton
_scheduler = MultiKeyScheduler(GROQ_API_KEYS)

def get_groq_client(call_type="text", est_tokens=1000):
    return _scheduler.get_client(call_type, est_tokens)

def safe_groq_call(call_type, est_tokens, api_call_fn, max_retries=5, initial_delay=2.0):
    delay = initial_delay
    for attempt in range(max_retries):
        client = _scheduler.get_client(call_type, est_tokens)
        try:
            return api_call_fn(client)
        except Exception as e:
            err_str = str(e).lower()
            # If it is a permanent quota/exhaustion/billing/credit error, disable the key immediately
            if any(p in err_str for p in ["quota", "exhausted", "billing", "credit", "insufficient"]):
                _scheduler.disable_key(client.api_key)
                time.sleep(0.5)
                continue
                
            if "400" in err_str or "401" in err_str or "auth" in err_str:
                raise e
            print(f"[Groq API] Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2.0
            
    # Final try
    client = _scheduler.get_client(call_type, est_tokens)
    return api_call_fn(client)
