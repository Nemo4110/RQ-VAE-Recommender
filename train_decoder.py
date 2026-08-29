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
from evaluate.metrics import TopKAccumulator
from evaluate.tiger_native import NativeItemTopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.tiger_policy import HISTORICAL_NATIVE
from modules.tiger_policy import DATASET_CAPPED_USER_BINS
from modules.tiger_policy import NATIVE_SEMANTIC_ID
from modules.tiger_policy import TIGERPolicyConfig
from modules.tiger_policy import select_semantic_id_tokens
from modules.tiger_policy import token_cardinalities
from modules.tiger_policy import validate_checkpoint_policy
from modules.tiger_policy import validate_full_semantic_ids
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_rqvae_hf_path=None,
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
    tiger_token_policy=HISTORICAL_NATIVE,
    paper_aligned=False,
    user_token_policy="modulo_hashed_bucket",
    user_bin_mode="explicit",
    user_bin_cap=2000,
    evaluator_policy="native_semantic_id",
    top_k_eval_list=[1, 5, 10],
):
    policy = TIGERPolicyConfig(
        token_policy=tiger_token_policy,
        paper_aligned=paper_aligned,
        num_user_bins=num_user_bins,
        user_token_policy=user_token_policy,
        user_bin_mode=user_bin_mode,
        user_bin_cap=user_bin_cap,
        evaluator_policy=evaluator_policy,
        rqvae_n_layers=vae_n_layers,
    )
    policy.validate()
    if evaluator_policy != NATIVE_SEMANTIC_ID:
        raise ValueError(
            "train_decoder.py implements the native semantic-ID evaluator; "
            "teacher-forced full-catalog scoring belongs to a separate adapter"
        )

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
    dataset_user_count = int(torch.unique(train_dataset.sequence_data["userId"]).numel())
    policy = policy.for_dataset(dataset_user_count)
    policy.validate()
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
        rqvae_hf_model_path=pretrained_rqvae_hf_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    all_codebooks = tokenizer.cached_ids.cpu()
    if policy.uses_collision_suffix:
        full_id_summary = validate_full_semantic_ids(all_codebooks)
        if not full_id_summary["full_id_unique"]:
            raise ValueError(
                "paper/full-id policy requires unique post-training full semantic IDs"
            )
        codebooks = all_codebooks
    else:
        codebooks = all_codebooks[:, :vae_n_layers]
    decoder_cardinalities = token_cardinalities(
        codebooks,
        n_layers=vae_n_layers,
        rqvae_codebook_size=vae_codebook_size,
        token_policy=policy.token_policy,
    )
    print({"tiger_policy": policy.metadata(), "token_cardinalities": decoder_cardinalities})

    model = EncoderDecoderRetrievalModel(
        codebooks=codebooks,
        num_hierarchies=len(decoder_cardinalities),
        num_embeddings_per_hierarchy=vae_codebook_size,
        token_cardinalities=decoder_cardinalities,
        token_policy=policy.token_policy,
        source_sem_ids_dim=tokenizer.sem_ids_dim,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=policy.effective_num_user_bins,
    )
    model = torch.compile(model)

    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    lr_scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=10000)

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        validate_checkpoint_policy(checkpoint.get("tiger_policy"), policy)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    metrics_accumulator = (
        NativeItemTopKAccumulator(codebooks, ks=tuple(top_k_eval_list))
        if policy.uses_collision_suffix
        else TopKAccumulator(ks=top_k_eval_list)
    )
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

            if (iter + 1) % full_eval_every == 0:
                model.eval()
                with tqdm(
                    eval_dataloader,
                    desc=f"Eval {iter + 1}",
                    disable=not accelerator.is_main_process,
                ) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        with torch.no_grad():
                            generated = model.generate_next_sem_id(
                                tokenized_data, top_k=True, temperature=1
                            )

                        actual = select_semantic_id_tokens(
                            tokenized_data.sem_ids_fut,
                            n_layers=vae_n_layers,
                            token_policy=policy.token_policy,
                            source_item_width=tokenizer.sem_ids_dim,
                        )
                        if policy.uses_collision_suffix:
                            metrics_accumulator.accumulate(
                                actual_semantic_ids=actual,
                                generated_semantic_ids=generated.sem_ids,
                            )
                        else:
                            metrics_accumulator.accumulate(
                                actual=actual, top_k=generated.sem_ids
                            )

                eval_metrics = metrics_accumulator.reduce()
                print(eval_metrics)
                if accelerator.is_main_process and wandb_logging:
                    wandb.log(eval_metrics)
                metrics_accumulator.reset()

            if accelerator.is_main_process:
                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "tiger_policy": policy.metadata(),
                        "token_cardinalities": decoder_cardinalities,
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
