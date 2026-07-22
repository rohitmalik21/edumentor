"""
EduMentor AI - LLM Client
Unified interface for Google Gemini, OpenAI, and Local Hugging Face models.
"""

import time

from config import Config
from utils.metrics_logger import metrics

MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds to wait on rate limit

# Cache for local model (loaded once, reused)
_local_model = None
_local_tokenizer = None


def get_llm_response(
    prompt: str,
    service_name: str = "general",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    Get a response from the configured LLM provider with automatic retry.

    Args:
        prompt: The prompt to send to the LLM.
        service_name: Name of the calling service (for metrics).
        temperature: Creativity parameter (0-1).
        max_tokens: Maximum response length.

    Returns:
        The LLM's text response.
    """
    start_time = time.time()
    tokens_used = 0
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            if Config.LLM_PROVIDER == "gemini":
                response, tokens_used = _call_gemini(prompt, temperature, max_tokens)
            elif Config.LLM_PROVIDER == "openai":
                response, tokens_used = _call_openai(prompt, temperature, max_tokens)
            elif Config.LLM_PROVIDER == "local":
                response, tokens_used = _call_local(prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown LLM provider: {Config.LLM_PROVIDER}")

            latency = time.time() - start_time
            metrics.log_request(
                service=service_name,
                latency=latency,
                tokens_used=tokens_used,
                success=True,
            )
            return response

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Retry on rate limit (429) errors
            if "429" in str(e) or "quota" in error_msg or "rate" in error_msg:
                if attempt < MAX_RETRIES - 1:
                    print(f"  Rate limited. Waiting {RETRY_DELAY}s before retry {attempt + 2}/{MAX_RETRIES}...")
                    time.sleep(RETRY_DELAY)
                    continue

            # Don't retry on other errors
            break

    latency = time.time() - start_time
    metrics.log_request(
        service=service_name,
        latency=latency,
        tokens_used=0,
        success=False,
    )
    raise RuntimeError(f"LLM request failed: {str(last_error)}") from last_error


def _load_local_model():
    """Load the local Hugging Face model (cached after first load)."""
    global _local_model, _local_tokenizer

    if _local_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        model_name = Config.LOCAL_MODEL
        print(f"  Loading local model: {model_name} (first time may download ~3GB)...")

        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        _local_model.eval()
        print(f"  Local model loaded successfully!")

    return _local_model, _local_tokenizer


def _call_local(prompt: str, temperature: float, max_tokens: int) -> tuple[str, int]:
    """Call a local Hugging Face model (FLAN-T5)."""
    import torch

    model, tokenizer = _load_local_model()

    # Tokenize input (FLAN-T5 has a 512 token input limit)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    input_token_count = inputs["input_ids"].shape[1]

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 512),  # FLAN-T5 works best with shorter outputs
            temperature=max(temperature, 0.1),  # Avoid temperature=0
            do_sample=temperature > 0.1,
            top_p=0.9,
            num_beams=2 if temperature <= 0.3 else 1,  # Use beam search for factual tasks
            early_stopping=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_token_count = outputs.shape[1]
    total_tokens = input_token_count + output_token_count

    return response, total_tokens


def _call_gemini(prompt: str, temperature: float, max_tokens: int) -> tuple[str, int]:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=Config.GEMINI_API_KEY)
    model = genai.GenerativeModel(Config.GEMINI_MODEL)

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    response = model.generate_content(prompt, generation_config=generation_config)

    # Estimate token usage
    tokens_used = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = (
            getattr(response.usage_metadata, "total_token_count", 0)
            or getattr(response.usage_metadata, "candidates_token_count", 0) +
            getattr(response.usage_metadata, "prompt_token_count", 0)
        )

    return response.text, tokens_used


def _call_openai(prompt: str, temperature: float, max_tokens: int) -> tuple[str, int]:
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are EduMentor AI, an educational assistant that helps students learn effectively."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    tokens_used = response.usage.total_tokens if response.usage else 0
    return response.choices[0].message.content, tokens_used
