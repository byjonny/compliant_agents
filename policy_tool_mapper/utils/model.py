from langchain_core.language_models import BaseChatModel


def create_llm(model: str, temperature: float = 0) -> BaseChatModel:
    """
    Factory that returns a LangChain chat model from a model identifier string.

    Supported prefixes:
      claude-*   -> langchain-anthropic  (ChatAnthropic)
      gpt-*, o1*, o3* -> langchain-openai (ChatOpenAI)
      gemini-*   -> langchain-google-genai (ChatGoogleGenerativeAI)
      mistral-*  -> langchain-mistralai  (ChatMistralAI)
    """
    model_lower = model.lower()

    if model_lower.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    if model_lower.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if model_lower.startswith("mistral"):
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model, temperature=temperature)

    raise ValueError(
        f"Unsupported model: '{model}'. "
        "Supported prefixes: claude-, gpt-, o1, o3, gemini-, mistral-. "
        "Install the matching langchain provider package."
    )
