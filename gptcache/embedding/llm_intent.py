"""
LLM Intent Embedding for GPTCache (CSE 584 Project)

Two-step pipeline:
  Step 1 (LLM via Vocareum API): Extract a structured "intent signature" from the query
          e.g. task_type, domain, required_facts, constraints, intent_keywords
  Step 2 (Local SBERT, free): Embed that intent signature into a vector

Why this is better than raw-text embeddings:
  - "What's the capital of France?" and "Tell me the capital city of France"
    produce identical intent signatures → guaranteed cache hit
  - Captures semantic reasoning intent, not surface-level word similarity
  - Inspired by chain-of-thought and interpretability research
"""

import json
import numpy as np
from gptcache.utils.log import gptcache_log
from gptcache.embedding.base import BaseEmbedding

INTENT_EXTRACTION_PROMPT = """\
Analyze the following query and produce a compact structured intent signature as a JSON object.
Fields:
  - task_type: one of [factual_lookup, creative_writing, code_generation, summarization,
                       translation, math_reasoning, comparison, explanation, instruction, other]
  - domain: subject domain (e.g. science, history, programming, medicine, general, etc.)
  - required_facts: list of 1-3 key entities or facts needed to answer correctly
  - constraints: list of specific answer constraints (e.g. brevity, formal_tone, specific_format).
                 Use ["none"] if there are none.
  - intent_keywords: 3-5 core keywords capturing the semantic meaning of the request

Return ONLY the JSON object — no markdown, no explanation.

Query: {query}"""


class LLMIntentEmbedding(BaseEmbedding):
    """
    Generates cache embeddings from LLM-produced intent signatures.

    Step 1: Calls the Vocareum LLM API to extract a structured JSON intent
            signature from the raw query (task type, domain, facts, keywords).
    Step 2: Embeds that intent signature using a local SBERT model (free, fast).

    Args:
        openai_api_key (str): Vocareum API key (starts with "voc-...")
        openai_base_url (str): Vocareum base URL, e.g. "https://genai.vocareum.com/v1"
        llm_model (str): Chat model name on Vocareum, e.g. "@azure-1/gpt-4o"
        sbert_model (str): Local sentence-transformers model for embedding.
        timeout (float): LLM call timeout in seconds.
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: str = "https://genai.vocareum.com/v1",
        llm_model: str = "@azure-1/gpt-4o",
        sbert_model: str = "all-MiniLM-L6-v2",
        timeout: float = 15.0,
    ):
        self._llm_model = llm_model
        self._timeout = timeout

        # ── LLM client (Vocareum) ──────────────────────────────────────────
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install openai: pip3 install openai>=1.0.0") from e

        # Ensure base_url ends with /v1
        base = openai_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"

        self._client = OpenAI(api_key=openai_api_key, base_url=base)

        # ── Local SBERT for embedding (free, no API cost) ─────────────────
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Please install sentence-transformers: pip3 install sentence-transformers"
            ) from e

        self._sbert = SentenceTransformer(sbert_model)
        self._dimension = self._sbert.get_sentence_embedding_dimension()

    def _extract_intent(self, query: str) -> str:
        """
        Calls the LLM to extract a structured intent signature from the query.
        Falls back to the raw query string if the LLM call fails.
        """
        response = None
        try:
            response = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{
                    "role": "user",
                    "content": INTENT_EXTRACTION_PROMPT.format(query=query)
                }],
                temperature=0,
                max_completion_tokens=250,
                timeout=self._timeout,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            # Flatten to descriptive text — this is what gets embedded
            intent_text = (
                f"task:{parsed.get('task_type', '')} "
                f"domain:{parsed.get('domain', '')} "
                f"facts:{' '.join(parsed.get('required_facts', []))} "
                f"constraints:{' '.join(parsed.get('constraints', []))} "
                f"keywords:{' '.join(parsed.get('intent_keywords', []))}"
            )
            return intent_text

        except json.JSONDecodeError:
            gptcache_log.warning(
                "LLMIntentEmbedding: LLM returned invalid JSON, using raw output."
            )
            if response:
                return response.choices[0].message.content.strip()
            return query

        except Exception as e:
            gptcache_log.warning(
                "LLMIntentEmbedding: intent extraction failed (%s), using raw query.", str(e)
            )
            return query

    def to_embeddings(self, data, **kwargs):
        """
        Step 1: Extract intent signature via LLM.
        Step 2: Embed the intent signature with local SBERT.
        Returns np.ndarray of shape (dimension,) float32.
        """
        if not isinstance(data, str):
            data = str(data)

        intent_text = self._extract_intent(data)
        vector = self._sbert.encode(intent_text, convert_to_numpy=True)
        return vector.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension
