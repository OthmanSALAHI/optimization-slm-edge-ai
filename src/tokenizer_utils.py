from transformers import AutoTokenizer


def mask_padding_tokens(input_ids, pad_token_id):
    """Replace padding token IDs with -100 so they are ignored by the loss."""
    if not input_ids:
        return input_ids

    if isinstance(input_ids[0], list):
        return [mask_padding_tokens(row, pad_token_id) for row in input_ids]

    return [token_id if token_id != pad_token_id else -100 for token_id in input_ids]


def load_tokenizer(model_name):
    """Load the tokenizer for the selected model."""
    return AutoTokenizer.from_pretrained(model_name)


def preprocess_function(
    examples,
    tokenizer,
    max_input_length=256,
    max_target_length=128,
):
    """Tokenize source_text and target_text for sequence-to-sequence training."""
    model_inputs = tokenizer(
        examples["source_text"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )

    pad_token_id = tokenizer.pad_token_id
    model_inputs["labels"] = mask_padding_tokens(labels["input_ids"], pad_token_id)

    return model_inputs


def decode_predictions(predictions, tokenizer):
    """Decode generated token IDs into readable text."""
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)
