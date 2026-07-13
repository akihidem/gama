

def test_ollama_default_model_is_a_readable_constant():
    """`DEFAULT_MODEL` is the single source of truth for what a missing tier loads.

    Tools that must *predict* what this backend will pull into VRAM (a budgeter, a trace
    verifier) have to know this value. Keeping it as an inline literal inside complete()
    forces them to duplicate the string, and the copy then drifts silently when this
    changes. Read it; don't copy it.
    """
    from gama.backends import OllamaBackend
    from gama.models import ModelTier

    be = OllamaBackend(model_by_tier={ModelTier.SMALL: "llama3.2:3b"})
    assert OllamaBackend.DEFAULT_MODEL == "gemma4:latest"
    # a tier with no entry resolves to DEFAULT_MODEL (what complete() would send to ollama)
    assert be.model_by_tier.get(ModelTier.LARGE, OllamaBackend.DEFAULT_MODEL) == "gemma4:latest"
    # and the built-in defaults are expressed in terms of it, not a second copy
    assert OllamaBackend().model_by_tier[ModelTier.LARGE] == OllamaBackend.DEFAULT_MODEL
