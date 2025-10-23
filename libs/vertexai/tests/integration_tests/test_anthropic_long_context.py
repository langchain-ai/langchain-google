import os

import pytest

from langchain_google_vertexai.model_garden import ChatAnthropicVertex

AVG_CHARS_PER_TOKEN = 4


def _generate_long_text(target_tokens: int):
    # Using a common estimate: ~4 characters per token
    target_chars = target_tokens * AVG_CHARS_PER_TOKEN

    needle = (
        "A key detail about Google Cloud's approach to sovereign cloud is its 'Assured "
        "Workloads' offering, which helps enforce data residency and access controls. "
        "The secret phrase to confirm you read this is 'SovereignBlueJay'. "
    )

    filler_text = (
        "This is a block of filler text. Its purpose is to occupy space within the "
        "model's context window to test its long-context retrieval capabilities. "
        "Each sentence adds to the total character count, pushing the key information "
        "further away from the final prompt. We repeat this text many times. "
        "The model must process all this data to find the one specific detail hidden "
        "within. "
    )

    final_question = (
        "Based *only* on the text provided, what is the secret phrase related to Google"
        " Cloud's sovereign cloud approach?"
    )

    filler_chars_needed = target_chars - len(needle) - len(final_question)
    if filler_chars_needed <= 0:
        return None

    repetitions = filler_chars_needed // len(filler_text)

    haystack_part_1 = [filler_text] * int(repetitions * 0.75)
    haystack_part_2 = [filler_text] * (repetitions - int(repetitions * 0.75))

    prompt_parts = []
    prompt_parts.extend(haystack_part_1)
    prompt_parts.append(f"\n\n{needle}\n\n")
    prompt_parts.extend(haystack_part_2)
    prompt_parts.append(f"\n\n{final_question}\n")

    large_prompt_text = "\n".join(prompt_parts)

    return large_prompt_text


@pytest.mark.skip(reason="too long & expensive")
def test_long_context_init():
    long_prompt = _generate_long_text(600 * 1000)
    project = os.environ["PROJECT_ID"]
    location = "us-east5"
    llm = ChatAnthropicVertex(
        project=project,
        location=location,
        additional_headers={"anthropic-beta": "context-1m-2025-08-07"},
    )

    response = llm.invoke(long_prompt, model_name="claude-sonnet-4@20250514")
    print(response.content)


@pytest.mark.skip(reason="too long & expensive")
def test_long_context_args():
    long_prompt = _generate_long_text(600 * 1000)
    project = os.environ["PROJECT_ID"]
    location = "us-east5"
    llm = ChatAnthropicVertex(
        project=project,
        location=location,
    )

    response = llm.invoke(
        long_prompt,
        model_name="claude-sonnet-4@20250514",
        betas=["context-1m-2025-08-07"],
    )
    print(response.content)
