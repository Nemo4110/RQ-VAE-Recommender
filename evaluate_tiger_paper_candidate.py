#!/usr/bin/env python
"""Post-hoc TIGER full-ID candidate evaluation without prefix masking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gin
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.processed import ItemData, RecDataset, SeqData
from data.utils import batch_to
from evaluate.tiger_native import PaperCandidateTopKAccumulator
from evaluate.tiger_native import unconstrained_autoregressive_beam_search
from modules.model import EncoderDecoderRetrievalModel
from modules.tiger_policy import PAPER_CANDIDATE_UNCONSTRAINED
from modules.tiger_policy import PAPER_FULL_ID
from modules.tiger_policy import TIGERPolicyConfig
from modules.tiger_policy import select_semantic_id_tokens
from modules.tiger_policy import token_cardinalities
from modules.tiger_policy import token_type_offsets
from modules.tiger_policy import validate_checkpoint_policy
from modules.tiger_policy import validate_full_semantic_ids
from modules.tiger_policy import validate_semantic_id_source
from modules.tokenizer.semids import SemanticIdTokenizer
from train_decoder import train as _register_train_config


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


def _query(name: str, default=None):
    try:
        return gin.query_parameter(f"train.{name}")
    except ValueError:
        return default


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated positive integer list")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a full-ID TIGER decoder checkpoint with deterministic, "
            "unconstrained four-token beam candidates and invalid-ID filtering."
        )
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", type=_parse_int_list, default=(1, 5, 10))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.top_k <= 0 or max(args.ks) > args.top_k:
        raise ValueError("top_k must cover every requested metric cutoff")
    if args.beam_size < args.top_k:
        raise ValueError("beam_size must be at least top_k before valid-ID filtering")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    gin.parse_config_file(args.config_path)
    if _query("tiger_token_policy") != PAPER_FULL_ID:
        raise ValueError("paper-candidate evaluation requires tiger_token_policy=paper/full-id")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    dataset_folder = _query("dataset_folder")
    dataset_split = _query("dataset_split")
    vae_n_layers = _query("vae_n_layers")
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        force_process=False,
        split=dataset_split,
    )
    train_dataset = SeqData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        is_train=True,
        subsample=_query("train_data_subsample"),
        include_future_item_in_subsample=_query(
            "train_subsample_include_validation_target", True
        ),
        split=dataset_split,
    )
    policy = TIGERPolicyConfig(
        token_policy=_query("tiger_token_policy"),
        paper_aligned=_query("paper_aligned"),
        num_user_bins=_query("num_user_bins"),
        user_token_policy=_query("user_token_policy"),
        user_bin_mode=_query("user_bin_mode"),
        user_bin_cap=_query("user_bin_cap"),
        evaluator_policy=PAPER_CANDIDATE_UNCONSTRAINED,
        rqvae_n_layers=vae_n_layers,
    ).for_dataset(int(torch.unique(train_dataset.sequence_data["userId"]).numel()))
    policy.validate()
    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        is_train=False,
        subsample=False,
        split=dataset_split,
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=_query("vae_input_dim"),
        hidden_dims=_query("vae_hidden_dims"),
        output_dim=_query("vae_embed_dim"),
        codebook_size=_query("vae_codebook_size"),
        n_layers=vae_n_layers,
        n_cat_feats=_query("vae_n_cat_feats"),
        rqvae_weights_path=_query("pretrained_rqvae_path"),
        rqvae_hf_model_path=_query("pretrained_rqvae_hf_path", None),
        rqvae_codebook_normalize=_query("vae_codebook_normalize"),
        rqvae_sim_vq=_query("vae_sim_vq"),
        semantic_id_source=_query("semantic_id_source", "rqvae"),
        random_id_seed=_query("random_id_seed", 0),
        random_id_cardinality=_query("random_id_cardinality", None),
        lsh_seed=_query("lsh_seed", 0),
        lsh_num_hyperplanes=_query("lsh_num_hyperplanes", 8),
    ).to(device)
    tokenizer.precompute_corpus_ids(item_dataset)
    codebooks = tokenizer.cached_ids.cpu()
    full_id_summary = validate_full_semantic_ids(codebooks, semantic_id_source=_query("semantic_id_source", "rqvae"))
    if not full_id_summary["full_id_unique"]:
        raise ValueError("paper-candidate evaluation requires unique full semantic IDs")
    cardinalities = token_cardinalities(
        codebooks,
        n_layers=vae_n_layers,
        rqvae_codebook_size=_query("vae_codebook_size"),
        token_policy=PAPER_FULL_ID,
        full_id_suffix_cardinality=_query("full_id_suffix_cardinality", None),
    )
    model = EncoderDecoderRetrievalModel(
        codebooks=codebooks,
        num_hierarchies=len(cardinalities),
        num_embeddings_per_hierarchy=_query("vae_codebook_size"),
        token_cardinalities=cardinalities,
        token_policy=PAPER_FULL_ID,
        source_sem_ids_dim=tokenizer.sem_ids_dim,
        t5_d_model=_query("t5_d_model"),
        t5_num_heads=_query("t5_num_heads"),
        t5_d_ff=_query("t5_d_ff"),
        t5_num_layers=_query("t5_num_layers"),
        top_k_for_generation=_query("top_k_for_generation"),
        should_add_sep_token=_query("should_add_sep_token"),
        num_user_bins=policy.effective_num_user_bins,
        user_token_policy=policy.user_token_policy,
        decoder_head_mode=_query("decoder_head_mode", "per_position_heads"),
        output_embedding_mode=_query("output_embedding_mode", "untied"),
        decoder_loss_reduction=_query("decoder_loss_reduction", "sum"),
        decoder_z_loss=_query("decoder_z_loss", 0.0),
    ).to(device)
    checkpoint = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_policy(
        checkpoint.get("tiger_policy"), policy, require_evaluator_policy=False
    )
    semantic_id_source = _query("semantic_id_source", "rqvae")
    random_id_seed = _query("random_id_seed", 0)
    lsh_seed = _query("lsh_seed", 0)
    lsh_num_hyperplanes = _query("lsh_num_hyperplanes", 8)
    validate_semantic_id_source(
        checkpoint.get("semantic_id_source"),
        semantic_id_source,
        checkpoint_random_seed=checkpoint.get("random_id_seed"),
        requested_random_seed=random_id_seed if semantic_id_source == "random" else None,
        checkpoint_random_cardinality=checkpoint.get("random_id_cardinality"),
        requested_random_cardinality=(
            tokenizer.random_id_cardinality if semantic_id_source == "random" else None
        ),
        checkpoint_lsh_seed=checkpoint.get("lsh_seed"),
        requested_lsh_seed=tokenizer.lsh_seed if semantic_id_source == "lsh" else None,
        checkpoint_lsh_num_hyperplanes=checkpoint.get("lsh_num_hyperplanes"),
        requested_lsh_num_hyperplanes=(
            tokenizer.lsh_num_hyperplanes if semantic_id_source == "lsh" else None
        ),
    )
    if checkpoint.get("train_subsample_include_validation_target", True) != _query(
        "train_subsample_include_validation_target", True
    ):
        raise ValueError(
            "train_subsample_include_validation_target mismatch between evaluator config and checkpoint"
        )
    checkpoint_head_mode = checkpoint.get("decoder_head_mode", "per_position_heads")
    if checkpoint_head_mode != model.decoder_head_mode:
        raise ValueError(
            "decoder_head_mode mismatch between evaluator config and checkpoint"
        )
    checkpoint_output_embedding_mode = checkpoint.get("output_embedding_mode", "untied")
    if checkpoint_output_embedding_mode != model.output_embedding_mode:
        raise ValueError(
            "output_embedding_mode mismatch between evaluator config and checkpoint"
        )
    checkpoint_loss_reduction = checkpoint.get("decoder_loss_reduction", "sum")
    if checkpoint_loss_reduction != model.decoder_loss_reduction:
        raise ValueError(
            "decoder_loss_reduction mismatch between evaluator config and checkpoint"
        )
    checkpoint_z_loss = checkpoint.get("decoder_z_loss", 0.0)
    if checkpoint_z_loss != model.decoder_z_loss:
        raise ValueError(
            "decoder_z_loss mismatch between evaluator config and checkpoint"
        )
    model.load_state_dict(_strip_compile_prefix(checkpoint["model"]), strict=True)
    model.eval()

    candidate_codebooks = codebooks
    if model.decoder_head_mode == "shared_vocab":
        offsets = torch.tensor(token_type_offsets(cardinalities), dtype=codebooks.dtype)
        candidate_codebooks = codebooks + offsets
    accumulator = PaperCandidateTopKAccumulator(
        candidate_codebooks, top_k=args.top_k, ks=args.ks, raw_ks=args.ks
    )
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    for batch in tqdm(dataloader, desc=f"Paper candidate eval {dataset_split}"):
        data = batch_to(batch, device)
        tokenized = tokenizer(data)
        input_ids = select_semantic_id_tokens(
            tokenized.sem_ids,
            n_layers=vae_n_layers,
            token_policy=PAPER_FULL_ID,
            source_item_width=tokenizer.sem_ids_dim,
        )
        attention_mask = select_semantic_id_tokens(
            tokenized.seq_mask.long(),
            n_layers=vae_n_layers,
            token_policy=PAPER_FULL_ID,
            source_item_width=tokenizer.sem_ids_dim,
        )
        generated, scores = unconstrained_autoregressive_beam_search(
            model,
            attention_mask,
            input_ids,
            beam_size=args.beam_size,
            user_id=tokenized.user_ids,
        )
        actual = select_semantic_id_tokens(
            tokenized.sem_ids_fut,
            n_layers=vae_n_layers,
            token_policy=PAPER_FULL_ID,
            source_item_width=tokenizer.sem_ids_dim,
        )
        if model.decoder_head_mode == "shared_vocab":
            actual = actual + torch.tensor(
                token_type_offsets(cardinalities), device=actual.device, dtype=actual.dtype
            )
        accumulator.accumulate(actual, generated, scores)

    result = {
        "evaluator_policy": PAPER_CANDIDATE_UNCONSTRAINED,
        "candidate_generation": "unconstrained_autoregressive_full_head_beam",
        "valid_prefix_masking": False,
        "invalid_id_metric_scope": f"raw_top_{args.top_k}_beams_before_valid_id_filtering",
        "ranking_metric_scope": "distinct_valid_catalog_items_after_filtering",
        "config": str(args.config_path),
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "decoder_iter": int(checkpoint.get("iter", -1)),
        "dataset_folder": dataset_folder,
        "dataset_split": dataset_split,
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "metric_cutoffs": list(args.ks),
        "batch_size": args.batch_size,
        "device": str(device),
        "tiger_training_policy": checkpoint.get("tiger_policy"),
        "evaluation_policy": policy.metadata(),
        "token_cardinalities": list(cardinalities),
        "decoder_head_mode": model.decoder_head_mode,
        "output_embedding_mode": model.output_embedding_mode,
        "decoder_loss_reduction": model.decoder_loss_reduction,
        "decoder_z_loss": model.decoder_z_loss,
        "semantic_id_source": semantic_id_source,
        "random_id_seed": random_id_seed if semantic_id_source == "random" else None,
        "random_id_cardinality": tokenizer.random_id_cardinality
        if semantic_id_source == "random"
        else None,
        "lsh_seed": tokenizer.lsh_seed if semantic_id_source == "lsh" else None,
        "lsh_num_hyperplanes": tokenizer.lsh_num_hyperplanes
        if semantic_id_source == "lsh"
        else None,
        "train_subsample_include_validation_target": _query(
            "train_subsample_include_validation_target", True
        ),
        "full_id_summary": full_id_summary,
        "metrics": accumulator.reduce(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "protocol_limitations": [
            "The TIGER TeX describes beam search and invalid-ID filtering but does not specify an exact beam width or tie-break.",
            "This evaluator fixes beam_size=50 by default and uses stable score/index tie-breaks for auditability.",
            "This is separate from constrained native generation and from the Temporal-v1 teacher-forced full-catalog scorer.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
