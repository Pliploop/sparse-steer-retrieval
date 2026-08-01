from transformers.models.auto.tokenization_auto import PreTrainedTokenizerFast


def make_tokenizer(tokenizer_file, max_sequence_length=512):
    UNKNOWN_TOK = "<unk>"  # unknown token
    START_TOK = "<start>"
    END_TOK = "<end>"
    PAD_TOK = "<pad>"
    MASK_TOK = "<mask>"
    CLS_TOK = "<cls>"
    SEP_TOK = "<sep>"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file,
                                        lowercase=True,
                                        padding='longest',
                                        pad_to_max_length=True,
                                        model_max_length=max_sequence_length,
                                        bos_token=START_TOK,
                                        cls_token=CLS_TOK,
                                        unk_token=UNKNOWN_TOK,
                                        pad_toen=PAD_TOK,
                                        mask_token=MASK_TOK,
                                        sep_token=SEP_TOK,
                                        eos_token=END_TOK)
    tokenizer.add_special_tokens({'pad_token': PAD_TOK})

    from tokenizers.processors import TemplateProcessing

    # defines how our processor should add special tokens for different situations
    # we are only interested in encoding single sentences so we just define that cases
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{START_TOK} $A {END_TOK}",
        special_tokens=[(f"{START_TOK}", tokenizer.bos_token_id),
                        (f"{END_TOK}", tokenizer.eos_token_id)])

    return tokenizer