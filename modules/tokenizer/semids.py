import random

import torch

from data.processed import ItemData
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange
from einops import pack
from modules.utils import eval_mode
from modules.rqvae import RqVae
from typing import List
from typing import Optional
from torch import nn
from torch import Tensor
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

BATCH_SIZE = 16


def generate_unique_random_semantic_ids(
    *,
    item_count: int,
    width: int,
    cardinality: int,
    seed: int,
) -> Tensor:
    """Generate deterministic, collision-free random IDs in a fixed token space."""

    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    if width <= 0 or cardinality <= 1:
        raise ValueError("width must be positive and cardinality must exceed one")
    capacity = cardinality**width
    if item_count > capacity:
        raise ValueError(
            "random semantic-ID space is too small for collision-free item assignment: "
            f"{item_count} > {capacity}"
        )

    rng = random.Random(seed)
    sampled: set[int] = set()
    flat_ids: list[int] = []
    while len(flat_ids) < item_count:
        candidate = rng.randrange(capacity)
        if candidate not in sampled:
            sampled.add(candidate)
            flat_ids.append(candidate)

    ids = torch.empty((item_count, width), dtype=torch.long)
    for row, value in enumerate(flat_ids):
        for column in range(width - 1, -1, -1):
            ids[row, column] = value % cardinality
            value //= cardinality
    return ids


def generate_lsh_semantic_ids(
    item_vectors: Tensor,
    *,
    width: int,
    num_hyperplanes: int,
    seed: int,
) -> Tensor:
    """Generate deterministic multi-codeword LSH IDs from frozen item vectors."""

    if item_vectors.ndim != 2:
        raise ValueError("item_vectors must have shape [item_count, embedding_dim]")
    if width <= 0 or num_hyperplanes <= 0:
        raise ValueError("width and num_hyperplanes must be positive")
    if num_hyperplanes > 62:
        raise ValueError("num_hyperplanes must be at most 62")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    vectors = item_vectors.detach().cpu().to(dtype=torch.float32)
    hyperplanes = torch.randn(
        (width, num_hyperplanes, vectors.shape[1]), generator=generator
    )
    powers = (1 << torch.arange(num_hyperplanes, dtype=torch.long)).unsqueeze(0)
    codes = []
    for hyperplanes_for_codeword in hyperplanes:
        bits = (vectors @ hyperplanes_for_codeword.T > 0).to(dtype=torch.long)
        codes.append((bits * powers).sum(dim=1))
    return torch.stack(codes, dim=1)


class SemanticIdTokenizer(nn.Module):
    """
    Tokenizes a batch of sequences of item features into a batch of sequences of semantic ids.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_layers: int = 3,
        n_cat_feats: int = 18,
        commitment_weight: float = 0.25,
        rqvae_weights_path: Optional[str] = None,
        rqvae_hf_model_path: Optional[str] = None,
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False,
        semantic_id_source: str = "rqvae",
        random_id_seed: int = 0,
        random_id_cardinality: int | None = None,
        lsh_seed: int = 0,
        lsh_num_hyperplanes: int = 8,
    ) -> None:
        super().__init__()

        if rqvae_weights_path is not None and rqvae_hf_model_path is not None:
            raise ValueError("at most one RQ-VAE source may be supplied")

        if rqvae_hf_model_path is not None:
            self.rq_vae = RqVae.from_pretrained(
                rqvae_hf_model_path,
                local_files_only=True,
            )
            expected = {
                "input_dim": input_dim,
                "embed_dim": output_dim,
                "hidden_dims": hidden_dims,
                "codebook_size": codebook_size,
                "n_layers": n_layers,
                "n_cat_feats": n_cat_feats,
            }
            actual = {
                "input_dim": self.rq_vae.input_dim,
                "embed_dim": self.rq_vae.embed_dim,
                "hidden_dims": self.rq_vae.hidden_dims,
                "codebook_size": self.rq_vae.codebook_size,
                "n_layers": self.rq_vae.n_layers,
                "n_cat_feats": self.rq_vae.n_cat_feats,
            }
            if actual != expected:
                raise ValueError(
                    f"local Hub RQ-VAE architecture mismatch: {actual} != {expected}"
                )
        else:
            self.rq_vae = RqVae(
                input_dim=input_dim,
                embed_dim=output_dim,
                hidden_dims=hidden_dims,
                codebook_size=codebook_size,
                codebook_kmeans_init=False,
                codebook_normalize=rqvae_codebook_normalize,
                codebook_sim_vq=rqvae_sim_vq,
                n_layers=n_layers,
                n_cat_features=n_cat_feats,
                commitment_weight=commitment_weight,
            )

            if rqvae_weights_path is not None:
                self.rq_vae.load_pretrained(rqvae_weights_path)

        self.rq_vae.eval()

        if semantic_id_source not in ("rqvae", "random", "lsh"):
            raise ValueError(
                "semantic_id_source must be 'rqvae', 'random', or 'lsh'; "
                f"got {semantic_id_source!r}"
            )
        self.codebook_size = codebook_size
        self.n_layers = n_layers
        resolved_random_id_cardinality = (
            codebook_size if random_id_cardinality is None else int(random_id_cardinality)
        )
        if not 1 < resolved_random_id_cardinality <= codebook_size:
            raise ValueError(
                "random_id_cardinality must be in [2, codebook_size]; "
                f"got {resolved_random_id_cardinality} for codebook_size={codebook_size}"
            )
        if semantic_id_source == "lsh":
            if not 0 < lsh_num_hyperplanes <= 62:
                raise ValueError("lsh_num_hyperplanes must be in [1, 62]")
            if 2**int(lsh_num_hyperplanes) > codebook_size:
                raise ValueError(
                    "lsh_num_hyperplanes produces codes outside codebook_size: "
                    f"2**{lsh_num_hyperplanes} > {codebook_size}"
                )
        self.semantic_id_source = semantic_id_source
        self.random_id_seed = int(random_id_seed)
        self.random_id_cardinality = resolved_random_id_cardinality
        self.lsh_seed = int(lsh_seed)
        self.lsh_num_hyperplanes = int(lsh_num_hyperplanes)
        self.reset()

    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(
            axis=-1
        )

    def reset(self):
        self.cached_ids = None

    @property
    def rqvae_n_layers(self) -> int:
        """Number of trainable RQ-VAE codebook levels."""

        return self.n_layers

    @property
    def sem_ids_dim(self) -> int:
        """Number of full item-ID tokens, including the collision suffix."""

        return self.n_layers + 1

    @property
    def collision_suffix_column(self) -> int:
        """Column containing the post-training collision-resolution suffix."""

        return self.n_layers

    def summarize_collision_resolution(self) -> dict[str, int | bool]:
        """Summarize the cached post-training full semantic IDs."""

        if self.cached_ids is None:
            raise RuntimeError("corpus IDs must be precomputed before summarizing collisions")
        full_ids = self.cached_ids
        unique_count = int(torch.unique(full_ids, dim=0).shape[0])
        suffix = full_ids[:, self.collision_suffix_column]
        return {
            "item_count": int(full_ids.shape[0]),
            "rqvae_n_layers": self.rqvae_n_layers,
            "full_id_width": int(full_ids.shape[1]),
            "unique_full_id_count": unique_count,
            "full_id_unique": unique_count == int(full_ids.shape[0]),
            "suffix_cardinality": int(suffix.max().item()) + 1
            if suffix.numel()
            else 0,
            "max_collision_bucket": int(suffix.max().item()) + 1
            if suffix.numel()
            else 0,
        }

    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        if self.semantic_id_source == "random":
            self.cached_ids = generate_unique_random_semantic_ids(
                item_count=len(movie_dataset),
                width=self.sem_ids_dim,
                cardinality=self.random_id_cardinality,
                seed=self.random_id_seed,
            ).to(self.rq_vae.device)
            return self.cached_ids
        if self.semantic_id_source == "lsh":
            self.cached_ids = generate_lsh_semantic_ids(
                movie_dataset.item_data[:, : self.rq_vae.input_dim],
                width=self.sem_ids_dim,
                num_hyperplanes=self.lsh_num_hyperplanes,
                seed=self.lsh_seed,
            ).to(self.rq_vae.device)
            return self.cached_ids

        cached_ids = None
        dedup_dim = []
        sampler = BatchSampler(
            SequentialSampler(range(len(movie_dataset))),
            batch_size=512,
            drop_last=False,
        )
        dataloader = DataLoader(
            movie_dataset,
            sampler=sampler,
            shuffle=False,
            collate_fn=lambda batch: batch[0],
        )
        for batch in dataloader:
            batch_ids = self.forward(batch_to(batch, self.rq_vae.device)).sem_ids
            # Detect in-batch duplicates
            is_hit = self._get_hits(batch_ids, batch_ids)
            hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
            assert hits.min() >= 0
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                # Detect batch-cache duplicates
                is_hit = self._get_hits(batch_ids, cached_ids)
                hits += is_hit.sum(axis=-1)
                cached_ids = pack([cached_ids, batch_ids], "* d")[0]
            dedup_dim.append(hits)
        # Append a post-training collision-resolution suffix; this is not an RQ-VAE layer.
        dedup_dim_tensor = pack(dedup_dim, "*")[0]
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]

        return self.cached_ids

    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        return rearrange(
            self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1]
        )

    @torch.no_grad
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        # TODO: Handle output inconstency in If-else.
        # If block has to return 3-sized ids for use in precompute_corpus_ids
        # Else block has to return deduped 4-sized ids for use in decoder training.
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            if self.semantic_id_source in ("random", "lsh"):
                raise RuntimeError(
                    f"{self.semantic_id_source} semantic IDs require precompute_corpus_ids "
                    "before tokenization"
                )
            B, N = batch.ids.shape
            sem_ids = self.rq_vae.get_semantic_ids(batch.x).sem_ids
            D = sem_ids.shape[-1]
            seq_mask, sem_ids_fut = None, None
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1

            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)

        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut,
        )
