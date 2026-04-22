"""
LLM Cache-Hit Validator for GPTCache (CSE 584 Project)

After vector similarity search finds a candidate cached answer, this evaluator
asks an LLM: "Is this cached answer truly appropriate for the new question?"

Why this matters:
  - Vector similarity can produce false positives (questions that sound similar
    but require different answers, e.g. "How do I sort a list in Python?" vs
    "How do I sort a list in Java?" — high embedding similarity, wrong answer)
  - The LLM acts as a reasoning judge, not just a distance calculator
  - It reads the actual cached answer, not just the question, before scoring

Design choices:
  - temperature=0 for deterministic scoring
  - max_tokens=10 to keep validation cheap (just needs a number)
  - Timeout + fallback ensure cache latency stays bounded
"""

from gptcache.utils.log import gptcache_log
from gptcache.similarity_evaluation.similarity_evaluation import SimilarityEvaluation

VALIDATION_PROMPT = """\
You are a semantic cache validator. Your job is to decide if a cached LLM response \
can be reused for a new question.

New Question:    {new_question}
Cached Question: {cached_question}
Cached Answer:   {cached_answer}

Score how appropriate the cached answer is for the new question on a scale from 0.0 to 1.0:
  1.0  — identical intent, cached answer is fully correct and complete
  0.8  — very similar intent, cached answer needs no or trivial modifications
  0.6  — related intent, cached answer is partially useful
  0.3  — loosely related, cached answer would mislead or be insufficient
  0.0  — different intent entirely, cached answer is wrong or irrelevant

Rules:
- Focus on whether the ANSWER would satisfy the NEW QUESTION, not just question similarity.
- Pay close attention to specific entities, languages, versions, or numeric values.
- Return ONLY a single decimal number between 0.0 and 1.0. No explanation."""


class LLMValidatorEvaluation(SimilarityEvaluation):
    """
    Uses an LLM as a lightweight semantic validator for cache hits.

    Sits after the vector search step and re-scores each candidate by asking
    the LLM whether the cached answer is truly appropriate for the new query.
    This catches false positives that embedding similarity alone misses.

    Args:
        openai_api_key (str): API key for OpenAI-compatible endpoint.
        openai_base_url (str, optional): Custom base URL (e.g. Vocareum endpoint).
        llm_model (str): Model to use for validation. Default: "gpt-4o".
        max_answer_len (int): Truncate long cached answers to this length.
        timeout (float): LLM call timeout in seconds.
        fallback_score (float): Score to return if LLM call fails (0.5 = neutral).

    Example::

        from gptcache.similarity_evaluation import LLMValidatorEvaluation
        evaluator = LLMValidatorEvaluation(
            openai_api_key="sk-...",
            openai_base_url="https://vocareum-proxy.example.com/v1",
        )
        cache.init(similarity_evaluation=evaluator, ...)
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: str = None,
        llm_model: str = "gpt-4o",
        max_answer_len: int = 600,
        timeout: float = 8.0,
        fallback_score: float = 0.5,
    ):
        self._llm_model = llm_model
        self._max_answer_len = max_answer_len
        self._timeout = timeout
        self._fallback_score = fallback_score

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install openai: pip install openai>=1.0.0") from e

        client_kwargs = {"api_key": openai_api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url

        self._client = OpenAI(**client_kwargs)

    def evaluation(self, src_dict: dict, cache_dict: dict, **kwargs) -> float:
        """
        Score whether the cached answer is appropriate for the new query.

        Args:
            src_dict:   Contains 'question' (the new incoming query).
            cache_dict: Contains 'question' (cached question) and
                        'answers' (list of Answer objects with .answer attribute).
        Returns:
            float in [0.0, 1.0]
        """
        try:
            new_question = src_dict.get("question", "").strip()
            cached_question = cache_dict.get("question", "").strip()

            answers = cache_dict.get("answers", [])
            if answers:
                raw_answer = answers[0].answer if hasattr(answers[0], "answer") else str(answers[0])
            else:
                raw_answer = ""

            cached_answer = str(raw_answer)[: self._max_answer_len]

            if not new_question or not cached_question:
                return self._fallback_score

            prompt = VALIDATION_PROMPT.format(
                new_question=new_question,
                cached_question=cached_question,
                cached_answer=cached_answer,
            )

            response = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=10,
                timeout=self._timeout,
            )

            score_str = response.choices[0].message.content.strip().split()[0]
            score = float(score_str)
            return max(0.0, min(1.0, score))

        except Exception as e:
            gptcache_log.warning(
                "LLMValidatorEvaluation: validation failed (%s), "
                "returning fallback score %.2f", str(e), self._fallback_score
            )
            return self._fallback_score

    def range(self):
        """Returns (min, max) score range."""
        return 0.0, 1.0
