import math
import os
import gin
import torch
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_decoder_model(
    tokenizer,
    vae_codebook_size,
    t5_d_model,
    t5_num_heads,
    t5_d_ff,
    t5_num_layers,
    top_k_for_generation,
    should_add_sep_token,
    num_user_bins,
):
    codebooks = tokenizer.cached_ids.cpu()
    if codebooks.ndim != 2 or codebooks.shape[1] != tokenizer.sem_ids_dim:
        raise ValueError("cached semantic IDs must match tokenizer.sem_ids_dim")
    if tokenizer.sem_ids_dim != 4:
        raise ValueError("TIGER decoder requires exactly four semantic ID tokens")
    if torch.unique(codebooks, dim=0).shape[0] != codebooks.shape[0]:
        raise ValueError("four-token semantic IDs must map one-to-one to items")
    token_cardinalities = (
        *([vae_codebook_size] * (tokenizer.sem_ids_dim - 1)),
        int(codebooks[:, -1].max().item()) + 1,
    )
    model = EncoderDecoderRetrievalModel(
        codebooks=codebooks,
        num_hierarchies=tokenizer.sem_ids_dim,
        num_embeddings_per_hierarchy=vae_codebook_size,
        token_cardinalities=token_cardinalities,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
    )
    return model, codebooks, token_cardinalities


def build_decoder_scheduler(optimizer, lr_warmup_steps):
    return InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=lr_warmup_steps,
    )


def get_full_eval_targets(tokenized_data):
    return tokenized_data.sem_ids_fut


METRIC_STATE_KEYS = (
    "item_hr@5",
    "item_hr@10",
    "item_ndcg@5",
    "item_ndcg@10",
    "item_mrr@5",
    "item_mrr@10",
    "item_total",
    "invalid_id_count",
    "prediction_count",
    "semantic_h@5",
    "semantic_h@10",
    "semantic_ndcg@10",
    "semantic_total",
)


class ItemTopKAccumulator:
    def __init__(self, codebooks, ks=(5, 10)):
        if tuple(ks) != (5, 10):
            raise ValueError("item metrics are fixed to cutoffs (5, 10)")
        self.code_to_item = {
            tuple(code): item_id for item_id, code in enumerate(codebooks.tolist())
        }
        self.reset()

    def reset(self):
        self.state_values = {key: 0.0 for key in METRIC_STATE_KEYS[:9]}

    def accumulate(self, actual_item_ids, generated_sem_ids):
        actual_items = actual_item_ids.reshape(-1).detach().cpu().tolist()
        generated_codes = generated_sem_ids.detach().cpu().tolist()
        for actual_item, ranked_codes in zip(actual_items, generated_codes, strict=True):
            ranked_items = []
            self.state_values["prediction_count"] += len(ranked_codes)
            for code in ranked_codes:
                item_id = self.code_to_item.get(tuple(code))
                self.state_values["invalid_id_count"] += item_id is None
                ranked_items.append(item_id)
            rank = next(
                (
                    index
                    for index, item_id in enumerate(ranked_items, start=1)
                    if item_id == actual_item
                ),
                None,
            )
            for cutoff in (5, 10):
                if rank is not None and rank <= cutoff:
                    self.state_values[f"item_hr@{cutoff}"] += 1.0
                    self.state_values[f"item_ndcg@{cutoff}"] += 1.0 / math.log2(
                        rank + 1.0
                    )
                    self.state_values[f"item_mrr@{cutoff}"] += 1.0 / rank
            self.state_values["item_total"] += 1.0

    def state(self):
        return dict(self.state_values)


class SemanticTopKAccumulator:
    def __init__(self, ks=(5, 10)):
        if tuple(ks) != (5, 10):
            raise ValueError("semantic diagnostics are fixed to cutoffs (5, 10)")
        self.reset()

    def reset(self):
        self.state_values = {key: 0.0 for key in METRIC_STATE_KEYS[9:]}

    def accumulate(self, actual, top_k):
        matches = (actual.unsqueeze(1) == top_k).all(dim=-1)
        for row in matches.detach().cpu().tolist():
            rank = next((index for index, matched in enumerate(row, start=1) if matched), None)
            if rank is not None and rank <= 5:
                self.state_values["semantic_h@5"] += 1.0
            if rank is not None and rank <= 10:
                self.state_values["semantic_h@10"] += 1.0
                self.state_values["semantic_ndcg@10"] += 1.0 / math.log2(rank + 1.0)
            self.state_values["semantic_total"] += 1.0

    def state(self):
        return dict(self.state_values)


def merge_metric_states(metric_states):
    return {
        key: sum(float(state.get(key, 0.0)) for state in metric_states)
        for key in METRIC_STATE_KEYS
    }


def reduce_metric_state(metric_state, accelerator):
    values = torch.tensor(
        [metric_state[key] for key in METRIC_STATE_KEYS],
        dtype=torch.float64,
        device=accelerator.device,
    )
    reduced = accelerator.reduce(values, reduction="sum")
    return {key: reduced[index].item() for index, key in enumerate(METRIC_STATE_KEYS)}


def finalize_eval_metrics(metric_state):
    item_total = metric_state["item_total"]
    semantic_total = metric_state["semantic_total"]
    prediction_count = metric_state["prediction_count"]
    if item_total <= 0 or semantic_total <= 0 or prediction_count <= 0:
        raise ValueError("cannot finalize evaluation metrics without predictions")
    return {
        "hr@5": metric_state["item_hr@5"] / item_total,
        "hr@10": metric_state["item_hr@10"] / item_total,
        "ndcg@5": metric_state["item_ndcg@5"] / item_total,
        "ndcg@10": metric_state["item_ndcg@10"] / item_total,
        "mrr@5": metric_state["item_mrr@5"] / item_total,
        "mrr@10": metric_state["item_mrr@10"] / item_total,
        "invalid_id_count": int(metric_state["invalid_id_count"]),
        "invalid_id_rate": metric_state["invalid_id_count"] / prediction_count,
        "diagnostic_semantic_h@5": metric_state["semantic_h@5"] / semantic_total,
        "diagnostic_semantic_h@10": metric_state["semantic_h@10"] / semantic_total,
        "diagnostic_semantic_ndcg@10": (
            metric_state["semantic_ndcg@10"] / semantic_total
        ),
    }


def run_full_evaluation(
    model,
    tokenizer,
    eval_dataloader,
    device,
    accelerator,
    codebooks,
    description=None,
):
    item_accumulator = ItemTopKAccumulator(codebooks, ks=(5, 10))
    semantic_accumulator = SemanticTopKAccumulator(ks=(5, 10))
    with tqdm(
        eval_dataloader,
        desc=description,
        disable=not accelerator.is_main_process,
    ) as progress:
        for batch in progress:
            data = batch_to(batch, device)
            tokenized_data = tokenizer(data)
            with torch.no_grad():
                generated = model.generate_next_sem_id(
                    tokenized_data, top_k=True, temperature=1
                )
            actual_sem_ids = get_full_eval_targets(tokenized_data)
            if actual_sem_ids.shape[-1] != 4:
                raise ValueError(
                    f"full target width must be 4; got {actual_sem_ids.shape[-1]}"
                )
            (
                gathered_item_ids,
                gathered_generated_sem_ids,
                gathered_actual_sem_ids,
            ) = accelerator.gather_for_metrics(
                (data.ids_fut, generated.sem_ids, actual_sem_ids)
            )
            if accelerator.is_main_process:
                semantic_accumulator.accumulate(
                    actual=gathered_actual_sem_ids,
                    top_k=gathered_generated_sem_ids,
                )
                item_accumulator.accumulate(
                    actual_item_ids=gathered_item_ids,
                    generated_sem_ids=gathered_generated_sem_ids,
                )

    local_state = merge_metric_states(
        [item_accumulator.state(), semantic_accumulator.state()]
    )
    return finalize_eval_metrics(reduce_metric_state(local_state, accelerator))


def publish_eval_metrics(
    eval_metrics,
    accelerator,
    wandb_logging,
    log_fn=None,
    print_fn=print,
):
    if not accelerator.is_main_process:
        return False
    print_fn(eval_metrics)
    if wandb_logging:
        (log_fn or wandb.log)(eval_metrics)
    return True


def should_run_full_eval(step, iterations, full_eval_every, full_eval_iterations):
    explicit_steps = set(full_eval_iterations or ())
    return (
        step == iterations
        or step in explicit_steps
        or (full_eval_every is not None and step % full_eval_every == 0)
    )


@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    lr_warmup_steps=10000,
    weight_decay=0.01,
    optimizer_name="adamw",
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    full_eval_iterations=None,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",
    max_grad_norm=None,
    t5_d_model=128,
    t5_num_heads=6,
    t5_d_ff=1024,
    t5_num_layers=4,
    top_k_for_generation=10,
    should_add_sep_token=True,
    num_user_bins=None,
):
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(project="gen-retrieval-decoder-training", config=params)

    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split,
    )
    train_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=True,
        subsample=train_data_subsample,
        split=dataset_split,
    )
    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=False,
        subsample=False,
        split=dataset_split,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    model, codebooks, token_cardinalities = build_decoder_model(
        tokenizer=tokenizer,
        vae_codebook_size=vae_codebook_size,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
    )
    model = torch.compile(model)

    if optimizer_name == "adafactor":
        from transformers import Adafactor

        optimizer = Adafactor(
            params=model.parameters(),
            lr=None,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True,
            weight_decay=0.0,
            clip_threshold=1.0,
        )
        lr_scheduler = None
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        lr_scheduler = build_decoder_scheduler(
            optimizer=optimizer,
            lr_warmup_steps=lr_warmup_steps,
        )
    else:
        raise ValueError(f"unsupported optimizer_name: {optimizer_name}")

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    if lr_scheduler is not None:
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )
    else:
        model, optimizer = accelerator.prepare(model, optimizer)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            train_debug_metrics = {}

            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every

                total_loss += loss.detach().item()

                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data)

                accelerator.backward(loss)

            assert model.item_sid_embedding_table.weight.grad is not None

            pbar.set_description(f"loss: {total_loss:.4f}")

            accelerator.wait_for_everyone()

            if max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            accelerator.wait_for_everyone()

            if (iter + 1) % partial_eval_every == 0:
                model.eval()
                eval_loss = 0.0
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)
                    with torch.no_grad():
                        eval_loss = model(tokenized_data).loss.item()

                if wandb_logging and accelerator.is_main_process:
                    wandb.log({"eval_loss": eval_loss})

            if should_run_full_eval(
                step=iter + 1,
                iterations=iterations,
                full_eval_every=full_eval_every,
                full_eval_iterations=full_eval_iterations,
            ):
                model.eval()
                eval_metrics = run_full_evaluation(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    accelerator=accelerator,
                    codebooks=codebooks,
                    description=f"Eval {iter + 1}",
                )
                publish_eval_metrics(
                    eval_metrics=eval_metrics,
                    accelerator=accelerator,
                    wandb_logging=wandb_logging,
                )

            if accelerator.is_main_process:
                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": (
                            lr_scheduler.state_dict()
                            if lr_scheduler is not None
                            else None
                        ),
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")

                if wandb_logging:
                    wandb.log(
                        {
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "total_loss": total_loss,
                            **train_debug_metrics,
                        }
                    )

            pbar.update(1)

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
