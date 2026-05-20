import argparse
import os
import json
import torch
from transformers import MarianTokenizer

from transformer import Config, Transformer
from preprocess import TOKENIZER_DIR


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Config(**data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MarianTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 65000
    config = load_config(args.config)

    model = Transformer(config=config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    src = tokenizer(args.text,
                    max_length=config.max_src_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt")
    src_input_ids = src["input_ids"].to(device)
    src_attention_mask = src["attention_mask"].to(device)

    bos_token_id = tokenizer.pad_token_id

    output_ids = model.generate(src_input_ids,
                                src_attention_mask,
                                max_new_tokens=args.max_new_tokens,
                                bos_token_id=bos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(decoded[0])


if __name__ == "__main__":
    main()
