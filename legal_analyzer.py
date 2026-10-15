from pathlib import Path
from functools import lru_cache
import os

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = BASE_DIR / "models" / "your_legal_ai_v3"

@lru_cache(maxsize=1)
def _load_model():
    """Load the user's fine-tuned Legal AI V3 once, only when AI is requested."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        # Some Transformers installations do not re-export the Auto classes
        # from transformers.__init__.py. Import them from their real modules.
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from peft import PeftModel
    import torch

    # Small CPU setting; does not affect app startup because this function is lazy.
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    except Exception:
        pass

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Your Legal AI V3 adapter was not found: {ADAPTER_DIR}"
        )

    # Prefer local cache so a temporary internet/DNS problem does not block AI startup.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, local_files_only=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, local_files_only=True, low_cpu_mem_usage=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, low_cpu_mem_usage=True
        )

    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), local_files_only=True)
    try:
        model = model.merge_and_unload()
    except Exception:
        pass
    model.eval()
    return tokenizer, model


def _generate(prompt, max_new_tokens=24):
    import torch
    tokenizer, model = _load_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document-grounded legal information assistant. "
                "Use only the supplied document and law context. Never invent "
                "names, dates, amounts, rights, duties or legal conclusions. "
                "If information is missing, say: Not stated in the document. "
                "Use simple English. Keep the answer short."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = "System: grounded legal assistant.\nUser: " + prompt + "\nAssistant:"

    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=768
    )
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.10,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _get_law_context(text, top_k=1):
    # Fast path: do not initialize the embedding model during the AI button click
    # unless RAG has already been loaded by the Indian Law Search page.
    try:
        from models import rag_search
        if rag_search._load_rag.cache_info().currsize == 0:
            return "RAG context not preloaded. Use Indian Law Search for semantic legal retrieval."
        results = rag_search.search_law(text[:900], top_k=top_k)
    except Exception:
        return "No relevant Indian-law reference was retrieved from the local knowledge base."
    blocks = []
    for item in results:
        section = str(item.get("section", "")).strip()
        if section:
            blocks.append(f"Similarity: {float(item.get('score', 0.0)):.3f}\n{section}")
    return "\n\n---\n\n".join(blocks) if blocks else "No relevant Indian-law reference was retrieved from the local knowledge base."


def run_own_legal_ai(text, max_new_tokens=24):
    document = text[:650]
    law_context = _get_law_context(document, top_k=1)
    prompt = f"""Analyze this legal document using the supplied law context.\n\nDOCUMENT:\n{document}\n\nINDIAN-LAW CONTEXT:\n{law_context}\n\nReturn only short sections:\nSUMMARY:\nIMPORTANT CLAUSES:\nRISKS:\nWHAT TO CHECK:"""
    return _generate(prompt, max_new_tokens=max_new_tokens)


def ask_legal_question(text, question, max_new_tokens=28):
    document = text[:650]
    prompt = f"""Answer the question using only the document below. If the answer is not present, say 'Not stated in the document.' Keep it short.\n\nDOCUMENT:\n{document}\n\nQUESTION:\n{question}"""
    return _generate(prompt, max_new_tokens=max_new_tokens)
