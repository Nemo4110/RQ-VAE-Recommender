import torch
import torch.nn as nn
import torch.nn.functional as F

from data.schemas import TokenizedSeqBatch
from typing import NamedTuple, Optional, Sequence
from torch import Tensor
from transformers import T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Config, T5Stack
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from modules.tiger_policy import HISTORICAL_NATIVE
from modules.tiger_policy import PAPER_FULL_ID
from modules.tiger_policy import select_semantic_id_tokens
from modules.tiger_policy import user_bucket_indices
from modules.tiger_policy import MODULO_HASHED_BUCKET
from modules.tiger_policy import token_type_offsets

torch.set_float32_matmul_precision("high")

PER_POSITION_HEADS = "per_position_heads"
SHARED_VOCAB_HEAD = "shared_vocab"


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


def t5x_z_loss(logits: Tensor, coefficient: float) -> Tensor:
    """Return the public T5X auxiliary ``coefficient * log(Z)^2`` mean."""

    if coefficient < 0:
        raise ValueError("z-loss coefficient must be non-negative")
    if coefficient == 0:
        return logits.new_zeros(())
    return coefficient * torch.logsumexp(logits, dim=-1).square().mean()


def _select_policy_tokens(
    tensor: torch.Tensor,
    *,
    source_item_width: int,
    target_item_width: int,
) -> torch.Tensor:
    """Select a named token-policy width from flattened item token groups."""

    if tensor.ndim != 2:
        raise ValueError("token tensor must have shape [batch, flattened_items]")
    if source_item_width < target_item_width or tensor.shape[1] % source_item_width:
        raise ValueError(
            f"token tensor width {tensor.shape[1]} is incompatible with "
            f"source item width {source_item_width}"
        )
    batch_size, total_width = tensor.shape
    return tensor.view(batch_size, total_width // source_item_width, source_item_width)[
        ..., :target_item_width
    ].reshape(batch_size, -1)


class EncoderDecoderRetrievalModel(nn.Module):
    """HuggingFace T5 encoder-decoder for sequential recommendation.

    Uses T5EncoderModel for encoding and T5Stack for decoding. Per-hierarchy
    linear output heads project decoder hidden states to codebook logits.
    Beam search uses multinomial sampling with log-probability accumulation
    and a float("-inf") mask for invalid SID prefixes.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        token_cardinalities: Optional[Sequence[int]] = None,
        token_policy: str = HISTORICAL_NATIVE,
        source_sem_ids_dim: Optional[int] = None,
        t5_d_model: int = 128,
        t5_num_heads: int = 6,
        t5_d_ff: int = 1024,
        t5_num_layers: int = 4,
        top_k_for_generation: int = 10,
        should_add_sep_token: bool = True,
        num_user_bins: Optional[int] = None,
        user_token_policy: str = MODULO_HASHED_BUCKET,
        decoder_head_mode: str = PER_POSITION_HEADS,
        output_embedding_mode: str = "untied",
        decoder_loss_reduction: str = "sum",
        decoder_z_loss: float = 0.0,
    ):
        super().__init__()

        self.num_hierarchies = num_hierarchies
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        if token_policy not in (HISTORICAL_NATIVE, PAPER_FULL_ID):
            raise ValueError(f"unsupported token policy: {token_policy!r}")
        self.token_policy = token_policy
        self.rqvae_n_layers = (
            num_hierarchies
            if token_policy == HISTORICAL_NATIVE
            else num_hierarchies - 1
        )
        self.source_sem_ids_dim = source_sem_ids_dim or (
            self.rqvae_n_layers
            if token_policy == PAPER_FULL_ID
            else self.rqvae_n_layers + 1
        )
        self.token_cardinalities = tuple(
            int(cardinality)
            for cardinality in (
                token_cardinalities
                if token_cardinalities is not None
                else [num_embeddings_per_hierarchy] * num_hierarchies
            )
        )
        if len(self.token_cardinalities) != num_hierarchies:
            raise ValueError("token_cardinalities must match num_hierarchies")
        if any(cardinality <= 0 for cardinality in self.token_cardinalities):
            raise ValueError("token cardinalities must be positive")
        if decoder_head_mode not in (PER_POSITION_HEADS, SHARED_VOCAB_HEAD):
            raise ValueError(
                f"unsupported decoder_head_mode={decoder_head_mode!r}"
            )
        self.decoder_head_mode = decoder_head_mode
        if output_embedding_mode not in ("untied", "tied"):
            raise ValueError(
                f"unsupported output_embedding_mode={output_embedding_mode!r}"
            )
        self.output_embedding_mode = output_embedding_mode
        if decoder_loss_reduction not in ("sum", "mean"):
            raise ValueError(
                f"unsupported decoder_loss_reduction={decoder_loss_reduction!r}"
            )
        self.decoder_loss_reduction = decoder_loss_reduction
        if decoder_z_loss < 0:
            raise ValueError("decoder_z_loss must be non-negative")
        self.decoder_z_loss = decoder_z_loss
        if self.source_sem_ids_dim < num_hierarchies:
            raise ValueError("source_sem_ids_dim cannot be smaller than decoder width")
        self.register_buffer(
            "token_type_offsets",
            torch.tensor(token_type_offsets(self.token_cardinalities), dtype=torch.long),
            persistent=False,
        )
        self.top_k_for_generation = top_k_for_generation
        self.user_token_policy = user_token_policy
        self.register_buffer("codebooks", codebooks)
        if codebooks.ndim != 2 or codebooks.shape[1] != num_hierarchies:
            raise ValueError("codebooks width must match num_hierarchies")

        encoder_config = T5Config(
            vocab_size=sum(self.token_cardinalities),
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            is_decoder=False,
        )
        self.encoder = T5EncoderModel(encoder_config)

        decoder_config = T5Config(
            vocab_size=sum(self.token_cardinalities),
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            is_decoder=True,
            is_encoder_decoder=False,
        )
        self.t5_decoder = T5Stack(decoder_config)
        self.bos_token = nn.Parameter(torch.randn(1, t5_d_model), requires_grad=True)
        if self.output_embedding_mode == "tied":
            self.decoder_mlp = nn.ModuleList()
            self.shared_decoder_mlp = None
        elif self.decoder_head_mode == PER_POSITION_HEADS:
            self.decoder_mlp = nn.ModuleList(
                [
                    nn.Linear(t5_d_model, cardinality, bias=False)
                    for cardinality in self.token_cardinalities
                ]
            )
            self.shared_decoder_mlp = None
        else:
            self.decoder_mlp = nn.ModuleList()
            self.shared_decoder_mlp = nn.Linear(
                t5_d_model, sum(self.token_cardinalities), bias=False
            )

        # Shared embedding table uses cumulative offsets for variable token cardinalities.
        self.item_sid_embedding_table = nn.Embedding(
            num_embeddings=sum(self.token_cardinalities),
            embedding_dim=t5_d_model,
        )

        self.user_embedding = (
            nn.Embedding(num_user_bins, t5_d_model) if num_user_bins else None
        )
        self.sep_token = (
            nn.Parameter(torch.randn(1, t5_d_model), requires_grad=True)
            if should_add_sep_token
            else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _is_cache_valid(self, kv) -> bool:
        if isinstance(kv, (EncoderDecoderCache, DynamicCache)):
            return len(kv) > 0
        return isinstance(kv, tuple)

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add per-hierarchy offsets so a single embedding table covers all hierarchies."""
        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")
        _, num_cols = input_sids.shape
        if num_hierarchies != self.num_hierarchies:
            raise ValueError("num_hierarchies must match the model token policy")
        offsets = self.token_type_offsets.to(input_sids.device)
        num_repeats = (num_cols + num_hierarchies - 1) // num_hierarchies
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]
        result = input_sids + repeated_offsets
        if attention_mask is not None:
            result = result * attention_mask
        return result

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ):
        """Inject a separator embedding after each item's token group."""
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count = seq_len // num_hierarchies
        reshaped_emb = id_embeddings.view(batch_size, item_count, num_hierarchies, -1)
        reshaped_mask = attention_mask.view(batch_size, item_count, num_hierarchies)
        sep = sep_token.unsqueeze(0).expand(batch_size, item_count, -1).unsqueeze(-2)
        id_embeddings = torch.cat([reshaped_emb, sep], dim=-2)
        attention_mask = torch.cat([reshaped_mask, reshaped_mask[:, :, [-1]]], dim=-1)
        return id_embeddings.reshape(batch_size, -1, emb_dim), attention_mask.reshape(
            batch_size, -1
        )

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        """Return a boolean mask indicating which prefixes exist in the corpus codebook."""
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)
        trimmed = self.codebooks[:, : prefix.shape[1]]
        results = []
        for i in range(0, prefix.shape[0], batch_size):
            batch = prefix[i : i + batch_size]
            results.append(
                (trimmed.unsqueeze(1) == batch.unsqueeze(0)).all(dim=2).any(dim=0)
            )
        return torch.cat(results)

    def encoder_forward_pass(self, attention_mask, input_ids, user_id=None):
        shifted = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds = self.item_sid_embedding_table(shifted)

        if self.sep_token is not None:
            inputs_embeds, attention_mask = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        if user_id is not None and self.user_embedding is not None:
            user_embeds = self.user_embedding(
                user_bucket_indices(
                    user_id[:, 0],
                    self.user_embedding.num_embeddings,
                    self.user_token_policy,
                )
            )
            inputs_embeds = torch.cat([user_embeds.unsqueeze(1), inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones(attention_mask.size(0), 1, device=attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        encoder_output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        return encoder_output, attention_mask

    def decoder_logits(
        self,
        hidden_state: torch.Tensor,
        hierarchy: int,
        *,
        full_vocab: bool = False,
    ) -> torch.Tensor:
        """Return decoder logits for one semantic-ID position.

        ``shared_vocab`` retains a single 1024-token-style output head.  Native
        generation slices that head to the position's legal token range, while
        paper-candidate generation can explicitly request the whole vocabulary.
        """

        if hierarchy < 0 or hierarchy >= self.num_hierarchies:
            raise ValueError("hierarchy is out of range")
        if self.output_embedding_mode == "tied":
            # Match T5X T5 1.0's logits-via-embedding convention: scale decoder
            # states before projecting through the input token embedding table.
            scale = hidden_state.shape[-1] ** -0.5
            if self.decoder_head_mode == SHARED_VOCAB_HEAD:
                logits = F.linear(
                    hidden_state * scale, self.item_sid_embedding_table.weight
                )
                if full_vocab:
                    return logits
                start = sum(self.token_cardinalities[:hierarchy])
                end = start + self.token_cardinalities[hierarchy]
                return logits[:, start:end]
            if full_vocab:
                raise ValueError("tied per-position head has no shared full vocabulary")
            start = sum(self.token_cardinalities[:hierarchy])
            end = start + self.token_cardinalities[hierarchy]
            return F.linear(
                hidden_state * scale, self.item_sid_embedding_table.weight[start:end]
            )
        if self.decoder_head_mode == PER_POSITION_HEADS:
            return self.decoder_mlp[hierarchy](hidden_state)
        assert self.shared_decoder_mlp is not None
        logits = self.shared_decoder_mlp(hidden_state)
        if full_vocab:
            return logits
        start = sum(self.token_cardinalities[:hierarchy])
        end = start + self.token_cardinalities[hierarchy]
        return logits[:, start:end]

    def decoder_target_ids(
        self, local_ids: torch.Tensor, hierarchy: int
    ) -> torch.Tensor:
        """Map local hierarchy IDs to the configured decoder target vocabulary."""

        if self.decoder_head_mode == PER_POSITION_HEADS:
            return local_ids
        return local_ids + sum(self.token_cardinalities[:hierarchy])

    def decoder_forward_pass(
        self,
        attention_mask=None,
        future_ids=None,
        encoder_output=None,
        attention_mask_for_encoder=None,
        use_cache=False,
        past_key_values=None,
        future_ids_are_global: bool = False,
    ):
        if future_ids is not None:
            shifted = (
                future_ids
                if future_ids_are_global
                else self._add_repeating_offset_to_rows(
                    input_sids=future_ids,
                    codebook_size=self.num_embeddings_per_hierarchy,
                    num_hierarchies=self.num_hierarchies,
                    attention_mask=torch.ones_like(future_ids)
                    if attention_mask is None
                    else attention_mask,
                )
            )
            inputs_embeds = self.item_sid_embedding_table(shifted)

            if not self._is_cache_valid(past_key_values):
                bos = self.bos_token.unsqueeze(0).expand(future_ids.size(0), 1, -1)
                inputs_embeds = torch.cat([bos, inputs_embeds], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(future_ids.size(0), 1, device=future_ids.device),
                            attention_mask,
                        ],
                        dim=1,
                    )
            else:
                inputs_embeds = inputs_embeds[:, -1:, :]
        else:
            inputs_embeds = self.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

        out = self.t5_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=attention_mask_for_encoder,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        if use_cache:
            return out.last_hidden_state, out.past_key_values
        return out.last_hidden_state

    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        input_ids = select_semantic_id_tokens(
            batch.sem_ids,
            n_layers=self.rqvae_n_layers,
            token_policy=self.token_policy,
            source_item_width=self.source_sem_ids_dim,
        )
        attention_mask = select_semantic_id_tokens(
            batch.seq_mask.long(),
            n_layers=self.rqvae_n_layers,
            token_policy=self.token_policy,
            source_item_width=self.source_sem_ids_dim,
        )
        fut_ids = select_semantic_id_tokens(
            batch.sem_ids_fut,
            n_layers=self.rqvae_n_layers,
            token_policy=self.token_policy,
            source_item_width=self.source_sem_ids_dim,
        )

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=batch.user_ids,
        )
        decoder_output = self.decoder_forward_pass(
            future_ids=fut_ids,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,
        )[:, :-1]  # [B, num_hierarchies, d_model]

        total_loss = torch.tensor(0.0, device=decoder_output.device)
        loss_d = []
        for h in range(self.num_hierarchies):
            logits = self.decoder_logits(
                decoder_output[:, h],
                h,
                full_vocab=self.decoder_head_mode == SHARED_VOCAB_HEAD,
            )
            h_loss = F.cross_entropy(
                logits, self.decoder_target_ids(fut_ids[:, h].long(), h)
            ) + t5x_z_loss(logits, self.decoder_z_loss)
            total_loss = total_loss + h_loss
            loss_d.append(h_loss.detach())
        if self.decoder_loss_reduction == "mean":
            total_loss = total_loss / self.num_hierarchies

        return ModelOutput(loss=total_loss, logits=None, loss_d=torch.stack(loss_d))

    @torch.no_grad()
    def generate(self, attention_mask, input_ids, user_id=None):
        """Generate top-k semantic IDs using sampling-based beam search.

        For each hierarchy level, samples n_candidates tokens via multinomial,
        scores them using cumulative log-probabilities with a float("-inf") mask for
        invalid SID prefixes, and keeps the top-k highest-scoring candidates.

        Returns:
            generated_ids: [B, top_k, num_hierarchies]
            log_probas:    [B, top_k]
        """
        B = input_ids.size(0)
        k = self.top_k_for_generation
        n_cands = min(64, max(self.token_cardinalities))

        enc_out, enc_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )
        rep_enc = enc_out.repeat_interleave(k, dim=0)
        rep_mask = enc_mask.repeat_interleave(k, dim=0)

        generated = None  # [B, k, h] grows with each hierarchy step
        log_probas = 0
        past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())

        for h in range(self.num_hierarchies):
            if generated is not None:
                cur_enc, cur_mask = rep_enc, rep_mask
                squeezed = generated.reshape(-1, h)
            else:
                cur_enc, cur_mask = enc_out, enc_mask
                squeezed = None

            dec_out, past_kv = self.decoder_forward_pass(
                future_ids=squeezed,
                encoder_output=cur_enc,
                attention_mask_for_encoder=cur_mask,
                use_cache=True,
                past_key_values=past_kv,
            )

            probas = F.softmax(self.decoder_logits(dec_out[:, -1, :], h), dim=-1)
            n_cands = min(64, self.token_cardinalities[h])
            samples = torch.multinomial(probas, num_samples=n_cands)
            samp_log_p = torch.log(torch.gather(probas, 1, samples))

            if generated is None:
                is_valid = self._check_valid_prefix(samples.reshape(-1, 1)).reshape(
                    B, n_cands
                )
                scores, idx = samp_log_p.masked_fill(~is_valid, float("-inf")).sort(
                    -1, descending=True
                )
                top_k_idx = idx[:, :k]
                generated = torch.gather(samples, 1, top_k_idx).unsqueeze(
                    -1
                )  # [B, k, 1]
                log_probas = scores[:, :k]
                past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
            else:
                prev = generated.reshape(-1, h).repeat_interleave(n_cands, dim=0)
                prefix = torch.cat([prev, samples.reshape(-1, 1)], dim=1)
                is_valid = self._check_valid_prefix(prefix).reshape(B, k * n_cands)
                scores, idx = (
                    (
                        samp_log_p.reshape(B, k * n_cands)
                        + log_probas.repeat_interleave(n_cands, dim=1)
                    )
                    .masked_fill(~is_valid, float("-inf"))
                    .sort(-1, descending=True)
                )

                top_k_idx = idx[:, :k]
                parent_beam_idx = top_k_idx // n_cands
                parent_global = (
                    parent_beam_idx
                    + torch.arange(B, device=parent_beam_idx.device).unsqueeze(1) * k
                ).flatten()
                past_kv.reorder_cache(parent_global)

                parent_ids = torch.gather(
                    generated, 1, parent_beam_idx.unsqueeze(-1).expand(-1, -1, h)
                )
                new_ids = torch.gather(
                    samples.reshape(B, k * n_cands), 1, top_k_idx
                ).unsqueeze(-1)
                generated = torch.cat([parent_ids, new_ids], dim=-1)  # [B, k, h+1]
                log_probas = scores[:, :k]

        return generated, log_probas

    @torch.no_grad()
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        top_k: bool = True,
        temperature: int = 1,
    ) -> GenerationOutput:
        input_ids = select_semantic_id_tokens(
            batch.sem_ids,
            n_layers=self.rqvae_n_layers,
            token_policy=self.token_policy,
            source_item_width=self.source_sem_ids_dim,
        )
        attention_mask = select_semantic_id_tokens(
            batch.seq_mask.long(),
            n_layers=self.rqvae_n_layers,
            token_policy=self.token_policy,
            source_item_width=self.source_sem_ids_dim,
        )
        generated_ids, log_probas = self.generate(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=batch.user_ids,
        )
        return GenerationOutput(sem_ids=generated_ids, log_probas=log_probas)
