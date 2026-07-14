import torch

from data.processed import ItemData
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange
from einops import pack
from modules.tokenizer.fixed_collision import build_fixed_four_token_ids
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
        collision_token_cardinality: Optional[int] = None,
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False,
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

        self.codebook_size = codebook_size
        self.collision_token_cardinality = collision_token_cardinality
        self.n_layers = n_layers
        self.reset()

    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(
            axis=-1
        )

    def reset(self):
        self.cached_ids = None

    @property
    def sem_ids_dim(self):
        return self.n_layers + 1

    @torch.no_grad
    @eval_mode
    def _precompute_three_token_ids(self, movie_dataset: ItemData) -> Tensor:
        batches = []
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
            batches.append(
                self.forward(batch_to(batch, self.rq_vae.device)).sem_ids
            )
        return pack(batches, "* d")[0]

    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        three_token_ids = self._precompute_three_token_ids(movie_dataset)
        if self.collision_token_cardinality is not None:
            result = build_fixed_four_token_ids(
                three_token_ids,
                collision_cardinality=self.collision_token_cardinality,
            )
            self.cached_ids = result.four_token_ids.to(three_token_ids.device)
            return self.cached_ids

        seen: dict[tuple[int, int, int], int] = {}
        fourth = torch.empty(
            (three_token_ids.shape[0], 1),
            dtype=torch.long,
            device=three_token_ids.device,
        )
        for item_index, code in enumerate(three_token_ids.detach().cpu().tolist()):
            key = tuple(int(token) for token in code)
            collision_index = seen.get(key, 0)
            fourth[item_index, 0] = collision_index
            seen[key] = collision_index + 1
        self.cached_ids = pack([three_token_ids, fourth], "b *")[0]
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
