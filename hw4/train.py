import json
import os
from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from transformers import MarianTokenizer
import sacrebleu

from transformer import Config, Transformer
from preprocess import TOKENIZER_DIR

BASE_CONFIG = {
    "num_blocks": 8,
    "d_model": 1536,
    "d_key": 256,
    "num_heads": 6,
    "d_ff": 4096,
    "max_src_len": 256,
    "max_tgt_len": 128,
    "use_attn_value": True,
    "use_residual": True,
    "use_moe": False,
    "num_experts": 4,
    "use_pre_norm": False,
}

VARIANTS = {
    "baseline": {},
    "no_attn_value": {
        "use_attn_value": False
    },
    "no_residual": {
        "use_residual": False
    },
    "pre_norm": {
        "use_pre_norm": True
    },
    "moe": {
        "use_moe": True
    },
}

DATA_DIR = os.path.join("data", "processed")
OUTPUT_ROOT = os.path.join("checkpoints")
TRAIN_VARIANTS = [
    "baseline",
    "no_attn_value",
    "no_residual",
    "pre_norm",
    "moe",
]

BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 3e-4
NUM_WORKERS = 2
GRAD_ACCUM = 1
EVAL_MAX_BATCHES = 10
EVAL_MAX_NEW_TOKENS = 64
EVAL_EVERY_STEPS = 50
MAX_UPDATES = 1200
EVAL_LOSS_MAX_BATCHES = 10


def _setup_distributed() -> Dict[str, int]:
    if dist.is_available() and dist.is_initialized():
        return {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        }

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="gloo")
        return {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        }

    return {"rank": 0, "world_size": 1, "local_rank": 0}


class ShardedTranslationDataset(IterableDataset):

    def __init__(self, shard_dir: str, shuffle: bool) -> None:
        self.shard_dir = shard_dir
        self.shuffle = shuffle
        index_path = os.path.join(shard_dir, "index.json")
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        self.total = int(index.get("total", 0))
        self.shards = [
            os.path.join(shard_dir, item["file"]) for item in index["shards"]
        ]

    def _select_shards(self) -> List[str]:
        shards = self.shards[:]
        if self.shuffle:
            generator = torch.Generator().manual_seed(0)
            shards = [
                shards[i] for i in torch.randperm(
                    len(shards), generator=generator).tolist()
            ]

        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        shards = [
            shard for idx, shard in enumerate(shards)
            if idx % world_size == rank
        ]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return shards

        return [
            shard for idx, shard in enumerate(shards)
            if idx % worker_info.num_workers == worker_info.id
        ]

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        for shard_path in self._select_shards():
            data = torch.load(shard_path, map_location="cpu")
            size = data["src_input_ids"].size(0)
            for i in range(size):
                yield {
                    "src_input_ids": data["src_input_ids"][i],
                    "src_attention_mask": data["src_attention_mask"][i],
                    "tgt_input_ids": data["tgt_input_ids"][i],
                    "tgt_attention_mask": data["tgt_attention_mask"][i],
                    "labels": data["labels"][i],
                }

    def __len__(self) -> int:
        return self.total


def _to_device(batch: Dict[str, torch.Tensor],
               device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model: Transformer,
                    loader: DataLoader,
                    optimizer,
                    scheduler,
                    device: torch.device,
                    grad_accum: int,
                    eval_every_steps: int,
                    global_step: int,
                    eval_fn: Optional[Callable[[int], None]] = None,
                    max_updates: Optional[int] = None,
                    is_main_process: bool = True) -> Tuple[float, int, bool]:
    model.train()
    total_loss = 0.0
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        if max_updates is not None and global_step >= max_updates:
            return total_loss / max(1, step), global_step, True
        batch = _to_device(batch, device)
        outputs = model(src_input_ids=batch["src_input_ids"],
                        src_attention_mask=batch["src_attention_mask"],
                        tgt_input_ids=batch["tgt_input_ids"],
                        tgt_attention_mask=batch["tgt_attention_mask"],
                        labels=batch["labels"])
        loss = outputs["loss"] / grad_accum
        loss.backward()
        step += 1
        if step % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if is_main_process:
                print(
                    f"Step {global_step}: loss={loss.item() * grad_accum:.4f}")
            if eval_fn is not None:
                if global_step % eval_every_steps == 0:
                    eval_fn(global_step, loss.item() * grad_accum)

            global_step += 1
        total_loss += loss.item() * grad_accum

    return total_loss / max(1, step), global_step, False


def evaluate(model: Transformer,
             loader: DataLoader,
             device: torch.device,
             max_batches: Optional[int] = None) -> float:
    model.eval()
    total_loss = 0.0
    step = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            batch = _to_device(batch, device)
            outputs = model(src_input_ids=batch["src_input_ids"],
                            src_attention_mask=batch["src_attention_mask"],
                            tgt_input_ids=batch["tgt_input_ids"],
                            tgt_attention_mask=batch["tgt_attention_mask"],
                            labels=batch["labels"])
            total_loss += outputs["loss"].item()
            step += 1
    return total_loss / max(1, step)


def evaluate_metrics(model: Transformer, loader: DataLoader,
                     device: torch.device, tokenizer: MarianTokenizer,
                     max_new_tokens: int,
                     max_batches: Optional[int]) -> Dict[str, float]:
    model.eval()
    source: List[str] = []
    translation: List[str] = []
    bos_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            batch = _to_device(batch, device)
            output_ids = model.generate(batch["src_input_ids"],
                                        batch["src_attention_mask"],
                                        max_new_tokens=max_new_tokens,
                                        bos_token_id=bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            decoded_hyp = tokenizer.batch_decode(output_ids,
                                                 skip_special_tokens=True)

            labels = batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            decoded_ref = tokenizer.batch_decode(labels,
                                                 skip_special_tokens=True)
            source.extend(decoded_ref)
            translation.extend(decoded_hyp)

    bleu = sacrebleu.corpus_bleu(translation, [source]).score
    chrf = sacrebleu.corpus_chrf(translation, [source]).score
    return {"bleu": bleu, "chrf": chrf, "samples": float(len(source))}


def save_checkpoint(path: str, model: Transformer, optimizer, scheduler,
                    config: Config, step: int) -> None:
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": asdict(config),
    }
    torch.save(state, path)


def main() -> None:
    agparser = argparse.ArgumentParser(
        description="Train a Transformer model.")
    agparser.add_argument("--variants",
                          required=True,
                          help="comma-separated variant names to train")
    args = agparser.parse_args()
    selected_variants = [
        v.strip() for v in args.variants.split(",") if v.strip()
    ]

    dist_state = _setup_distributed()
    device = torch.device("cuda",
                          dist_state["local_rank"]) if torch.cuda.is_available(
                          ) else torch.device("cpu")

    tokenizer = MarianTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 65000

    train_dataset = ShardedTranslationDataset(os.path.join(DATA_DIR, "train"),
                                              shuffle=True)
    valid_dataset = ShardedTranslationDataset(os.path.join(DATA_DIR, "valid"),
                                              shuffle=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())

    variants_to_run = selected_variants or TRAIN_VARIANTS
    for variant in variants_to_run:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        config_kwargs = dict(BASE_CONFIG)
        config_kwargs.update(VARIANTS[variant])
        config = Config(num_tokens=len(tokenizer),
                        pad_token_id=tokenizer.pad_token_id,
                        **config_kwargs)

        model = Transformer(config=config).to(device)
        if dist_state["world_size"] > 1:
            model = DDP(model,
                        device_ids=None if device.type == "cpu" else
                        [dist_state["local_rank"]])

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        per_rank = len(train_dataset)
        if dist_state["world_size"] > 1:
            per_rank = max(1, per_rank // dist_state["world_size"])
        steps_per_epoch = max(1,
                              (per_rank // BATCH_SIZE) // max(1, GRAD_ACCUM))
        total_steps = max(MAX_UPDATES, NUM_EPOCHS * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, total_steps=total_steps)

        output_dir = os.path.join(OUTPUT_ROOT, variant)
        os.makedirs(output_dir, exist_ok=True)
        if dist_state["rank"] == 0:
            with open(os.path.join(output_dir, "config.json"),
                      "w",
                      encoding="utf-8") as handle:
                json.dump(asdict(config), handle, indent=2)

        def run_eval(step: int, loss: float) -> None:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if dist_state["rank"] == 0:
                val_loss_step = evaluate(model, valid_loader, device,
                                         EVAL_LOSS_MAX_BATCHES)
                metrics_step = evaluate_metrics(
                    model.module if isinstance(model, DDP) else model,
                    valid_loader, device, tokenizer, EVAL_MAX_NEW_TOKENS,
                    EVAL_MAX_BATCHES)
                metrics_path = os.path.join(output_dir, "metrics.jsonl")
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    record = {
                        "step": step,
                        "train_loss": loss,
                        "val_loss": val_loss_step,
                        "bleu": metrics_step["bleu"],
                        "chrf": metrics_step["chrf"],
                        "eval_samples": metrics_step["samples"],
                    }
                    handle.write(json.dumps(record) + "\n")
                ckpt_path = os.path.join(output_dir,
                                         f"ckpt_step_{step:03d}.pt")
                save_checkpoint(
                    ckpt_path,
                    model.module if isinstance(model, DDP) else model,
                    optimizer, scheduler, config, step)

                print(
                    f"[{variant}] Step {step}: val_loss={val_loss_step:.4f} "
                    f"bleu={metrics_step['bleu']:.2f} chrf={metrics_step['chrf']:.2f}"
                )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            model.train()

        global_step = 0
        reached_max = False
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, global_step, reached_max = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, GRAD_ACCUM,
                EVAL_EVERY_STEPS, global_step, run_eval, MAX_UPDATES,
                dist_state["rank"] == 0)
            if reached_max:
                if dist_state["rank"] == 0:
                    run_eval(global_step, train_loss)
                break
            val_loss = evaluate(model, valid_loader, device,
                                EVAL_LOSS_MAX_BATCHES)
            metrics = {"bleu": 0.0, "chrf": 0.0, "samples": 0.0}
            if dist_state["rank"] == 0:
                metrics = evaluate_metrics(
                    model.module if isinstance(model, DDP) else model,
                    valid_loader, device, tokenizer, EVAL_MAX_NEW_TOKENS,
                    EVAL_MAX_BATCHES)
            if dist_state["rank"] == 0:

                metrics_path = os.path.join(output_dir, "metrics.jsonl")
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "bleu": metrics["bleu"],
                        "chrf": metrics["chrf"],
                        "eval_samples": metrics["samples"],
                    }
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"[{variant}] Epoch {epoch}: train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} bleu={metrics['bleu']:.2f} "
                    f"chrf={metrics['chrf']:.2f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
