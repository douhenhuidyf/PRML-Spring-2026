import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import MarianTokenizer

TOKENIZER_DIR = "./tokenizer"


def _shift_right(input_ids: torch.Tensor, bos_token_id: int,
                 pad_token_id: int) -> torch.Tensor:
    shifted = input_ids.new_full(input_ids.shape, pad_token_id)
    shifted[:, 1:] = input_ids[:, :-1]
    shifted[:, 0] = bos_token_id
    return shifted


def _iter_json_lines(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _save_shard(shard: Dict[str, List[List[int]]], out_path: str) -> int:
    tensors = {
        "src_input_ids":
        torch.tensor(shard["src_input_ids"], dtype=torch.long),
        "src_attention_mask":
        torch.tensor(shard["src_attention_mask"], dtype=torch.long),
        "tgt_input_ids":
        torch.tensor(shard["tgt_input_ids"], dtype=torch.long),
        "tgt_attention_mask":
        torch.tensor(shard["tgt_attention_mask"], dtype=torch.long),
        "labels":
        torch.tensor(shard["labels"], dtype=torch.long),
    }
    count = tensors["src_input_ids"].size(0)
    torch.save(tensors, out_path)
    return count


_WORKER_TOKENIZER: MarianTokenizer | None = None
_WORKER_MAX_SRC_LEN = 0
_WORKER_MAX_TGT_LEN = 0
_WORKER_PAD_TOKEN_ID = 0


def _init_worker(tokenizer_dir: str, max_src_len: int, max_tgt_len: int,
                 pad_token_id: int) -> None:
    global _WORKER_TOKENIZER
    global _WORKER_MAX_SRC_LEN
    global _WORKER_MAX_TGT_LEN
    global _WORKER_PAD_TOKEN_ID
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = pad_token_id
    _WORKER_TOKENIZER = tokenizer
    _WORKER_MAX_SRC_LEN = max_src_len
    _WORKER_MAX_TGT_LEN = max_tgt_len
    _WORKER_PAD_TOKEN_ID = pad_token_id


def _tokenize_batch(
    src_batch: List[str], tgt_batch: List[str]
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]],
           List[List[int]]]:
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("Tokenizer worker not initialized")
    src_enc = tokenizer(src_batch,
                        max_length=_WORKER_MAX_SRC_LEN,
                        padding="max_length",
                        truncation=True)
    tgt_enc = tokenizer(tgt_batch,
                        max_length=_WORKER_MAX_TGT_LEN,
                        padding="max_length",
                        truncation=True)
    src_ids = torch.tensor(src_enc["input_ids"], dtype=torch.long)
    src_mask = torch.tensor(src_enc["attention_mask"], dtype=torch.long)
    tgt_ids = torch.tensor(tgt_enc["input_ids"], dtype=torch.long)
    tgt_mask = torch.tensor(tgt_enc["attention_mask"], dtype=torch.long)
    decoder_input_ids = _shift_right(tgt_ids, _WORKER_PAD_TOKEN_ID,
                                     _WORKER_PAD_TOKEN_ID)
    labels = tgt_ids.clone()
    labels[labels == _WORKER_PAD_TOKEN_ID] = -100
    return (src_ids.tolist(), src_mask.tolist(), decoder_input_ids.tolist(),
            tgt_mask.tolist(), labels.tolist())


def preprocess_split(tokenizer: MarianTokenizer,
                     src_path: str,
                     out_dir: str,
                     max_src_len: int,
                     max_tgt_len: int,
                     shard_size: int,
                     batch_size: int,
                     num_workers: int = 2,
                     max_pending_batches: int = 4) -> None:
    os.makedirs(out_dir, exist_ok=True)
    shard_index: List[Dict[str, int | str]] = []
    shard = {
        "src_input_ids": [],
        "src_attention_mask": [],
        "tgt_input_ids": [],
        "tgt_attention_mask": [],
        "labels": [],
    }
    buffer_src: List[str] = []
    buffer_tgt: List[str] = []
    shard_id = 0
    total = 0

    bos_token_id = tokenizer.pad_token_id

    def flush_shard_if_needed() -> None:
        nonlocal shard_id, total, shard
        if len(shard["src_input_ids"]) < shard_size:
            return
        out_path = os.path.join(out_dir, f"shard_{shard_id:05d}.pt")
        count = _save_shard(shard, out_path)
        shard_index.append({
            "file": os.path.basename(out_path),
            "count": count
        })
        total += count
        shard = {
            "src_input_ids": [],
            "src_attention_mask": [],
            "tgt_input_ids": [],
            "tgt_attention_mask": [],
            "labels": [],
        }
        shard_id += 1

    def consume_result(
        result: Tuple[List[List[int]], List[List[int]], List[List[int]],
                      List[List[int]], List[List[int]]]
    ) -> None:
        src_ids, src_mask, tgt_ids, tgt_mask, labels = result
        shard["src_input_ids"].extend(src_ids)
        shard["src_attention_mask"].extend(src_mask)
        shard["tgt_input_ids"].extend(tgt_ids)
        shard["tgt_attention_mask"].extend(tgt_mask)
        shard["labels"].extend(labels)
        flush_shard_if_needed()

    pending = []
    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(TOKENIZER_DIR, max_src_len, max_tgt_len,
                        bos_token_id)) as pool:
        for item in _iter_json_lines(src_path):
            buffer_src.append(item["chinese"])
            buffer_tgt.append(item["english"])
            if len(buffer_src) < batch_size:
                continue

            pending.append(
                pool.apply_async(_tokenize_batch,
                                 args=(buffer_src, buffer_tgt)))
            buffer_src = []
            buffer_tgt = []

            if len(pending) >= max_pending_batches:
                result = pending.pop(0).get()
                consume_result(result)

        if buffer_src:
            pending.append(
                pool.apply_async(_tokenize_batch,
                                 args=(buffer_src, buffer_tgt)))

        for task in pending:
            consume_result(task.get())

    if shard["src_input_ids"]:
        out_path = os.path.join(out_dir, f"shard_{shard_id:05d}.pt")
        count = _save_shard(shard, out_path)
        shard_index.append({
            "file": os.path.basename(out_path),
            "count": count
        })
        total += count

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({"total": total, "shards": shard_index}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json",
                        default=os.path.join("data",
                                             "translation2019zh_train.json"))
    parser.add_argument("--valid-json",
                        default=os.path.join("data",
                                             "translation2019zh_valid.json"))
    parser.add_argument("--out-dir", default=os.path.join("data", "processed"))
    parser.add_argument("--max-src-len", default=192, type=int)
    parser.add_argument("--max-tgt-len", default=128, type=int)
    parser.add_argument("--shard-size", default=10000, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    args = parser.parse_args()

    tokenizer = MarianTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 65000
    train_out = os.path.join(args.out_dir, "train")
    valid_out = os.path.join(args.out_dir, "valid")

    preprocess_split(tokenizer,
                     args.valid_json,
                     valid_out,
                     args.max_src_len,
                     args.max_tgt_len,
                     args.shard_size,
                     args.batch_size,
                     num_workers=args.num_workers,
                     max_pending_batches=args.num_workers * 2)
    preprocess_split(tokenizer,
                     args.train_json,
                     train_out,
                     args.max_src_len,
                     args.max_tgt_len,
                     args.shard_size,
                     args.batch_size,
                     num_workers=args.num_workers,
                     max_pending_batches=args.num_workers * 2)


if __name__ == "__main__":
    main()
