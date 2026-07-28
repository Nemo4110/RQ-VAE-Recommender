import torch
import torch.nn as nn
import torch.nn.functional as F

from data.schemas import TokenizedSeqBatch
from typing import NamedTuple, Optional, Sequence
from torch import Tensor
from transformers import T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Config, T5Stack
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

torch.set_float32_matmul_precision("high")


def build_valid_prefix_index(codebooks: torch.Tensor):
    prefix_index = []
    code_tuples = [tuple(int(token) for token in code) for code in codebooks.tolist()]
    for hierarchy in range(codebooks.shape[1]):
        allowed_next = {}
        for code in code_tuples:
            prefix = code[:hierarchy]
            allowed_next.setdefault(prefix, set()).add(code[hierarchy])
        prefix_index.append(
            {
                prefix: torch.tensor(sorted(tokens), dtype=torch.long)
                for prefix, tokens in allowed_next.items()
            }
        )
    return prefix_index


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class EncoderDecoderRetrievalModel(nn.Module):
    """HuggingFace T5 encoder-decoder for sequential recommendation.

    Uses T5EncoderModel for encoding and T5Stack for decoding. Per-hierarchy
    linear output heads project decoder hidden states to codebook logits.
    Beam search uses deterministic top-k expansion with log-probability accumulation
    and a float("-inf") mask for invalid SID prefixes.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        token_cardinalities: Optional[Sequence[int]] = None,
        t5_d_model: int = 128,
        t5_num_heads: int = 6,
        t5_d_ff: int = 1024,
        t5_num_layers: int = 4,
        top_k_for_generation: int = 10,
        should_add_sep_token: bool = True,
        num_user_bins: Optional[int] = None,
    ):
        super().__init__()

        self.num_hierarchies = num_hierarchies
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
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
        cardinalities = torch.tensor(self.token_cardinalities, dtype=torch.long)
        token_type_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.cumsum(cardinalities[:-1], dim=0)]
        )
        self.register_buffer("token_type_offsets", token_type_offsets)
        valid_tuple_count = torch.unique(codebooks, dim=0).shape[0]
        if not 1 <= top_k_for_generation <= valid_tuple_count:
            raise ValueError(
                "top_k_for_generation must not exceed valid corpus tuples"
            )
        self.top_k_for_generation = top_k_for_generation
        self.register_buffer("codebooks", codebooks)
        self.valid_prefix_index = build_valid_prefix_index(codebooks)
        self.last_generation_max_expansion_score_elements = 0

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
        self.decoder_mlp = nn.ModuleList(
            [
                nn.Linear(t5_d_model, cardinality, bias=False)
                for cardinality in self.token_cardinalities
            ]
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
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add per-hierarchy offsets so a single embedding table covers all hierarchies."""
        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")
        _, num_cols = input_sids.shape
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
                torch.remainder(user_id[:, 0], self.user_embedding.num_embeddings)
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

    def decoder_forward_pass(
        self,
        attention_mask=None,
        future_ids=None,
        encoder_output=None,
        attention_mask_for_encoder=None,
        use_cache=False,
        past_key_values=None,
    ):
        if future_ids is not None:
            shifted = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids)
                if attention_mask is None
                else attention_mask,
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
        input_ids = batch.sem_ids
        attention_mask = batch.seq_mask.long()
        fut_ids = batch.sem_ids_fut

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
            logits = self.decoder_mlp[h](decoder_output[:, h])
            h_loss = F.cross_entropy(logits, fut_ids[:, h].long())
            total_loss = total_loss + h_loss
            loss_d.append(h_loss.detach())

        return ModelOutput(loss=total_loss, logits=None, loss_d=torch.stack(loss_d))

    def _select_valid_expansions(
        self,
        probabilities,
        beam_prefixes,
        log_probas,
        hierarchy,
    ):
        batch_size = len(beam_prefixes)
        beam_count = len(beam_prefixes[0])
        probabilities = probabilities.reshape(
            batch_size, beam_count, probabilities.shape[-1]
        )
        batch_candidates = []
        min_candidate_count = None

        for batch_index, prefixes in enumerate(beam_prefixes):
            score_parts = []
            token_parts = []
            parent_parts = []
            next_prefixes = []
            for parent_index, prefix in enumerate(prefixes):
                allowed_token_values = self.valid_prefix_index[hierarchy][prefix].tolist()
                allowed_tokens = torch.tensor(
                    allowed_token_values,
                    dtype=torch.long,
                    device=probabilities.device,
                )
                scores = torch.log(
                    probabilities[batch_index, parent_index, allowed_tokens].clamp_min(
                        1e-30
                    )
                )
                if log_probas is not None:
                    scores = scores + log_probas[batch_index, parent_index]
                score_parts.append(scores)
                token_parts.append(allowed_tokens)
                parent_parts.append(
                    torch.full_like(allowed_tokens, parent_index, dtype=torch.long)
                )
                next_prefixes.extend(
                    prefix + (token,) for token in allowed_token_values
                )

            candidate_scores = torch.cat(score_parts)
            candidate_tokens = torch.cat(token_parts)
            parent_indices = torch.cat(parent_parts)
            candidate_count = int(candidate_scores.numel())
            self.last_generation_max_expansion_score_elements = max(
                self.last_generation_max_expansion_score_elements,
                candidate_count,
            )
            min_candidate_count = (
                candidate_count
                if min_candidate_count is None
                else min(min_candidate_count, candidate_count)
            )
            batch_candidates.append(
                (candidate_scores, candidate_tokens, parent_indices, next_prefixes)
            )

        keep_count = min(self.top_k_for_generation, min_candidate_count)
        if keep_count == 0:
            raise RuntimeError("no finite valid semantic ID continuations remain")

        selected_scores = []
        selected_tokens = []
        selected_parents = []
        selected_prefixes = []
        for candidate_scores, candidate_tokens, parent_indices, next_prefixes in batch_candidates:
            scores, indices = torch.topk(candidate_scores, k=keep_count, dim=0)
            selected_scores.append(scores)
            selected_tokens.append(candidate_tokens[indices])
            selected_parents.append(parent_indices[indices])
            selected_prefixes.append([next_prefixes[index] for index in indices.tolist()])

        return (
            torch.stack(selected_tokens),
            torch.stack(selected_parents),
            torch.stack(selected_scores),
            selected_prefixes,
        )

    @torch.no_grad()
    def generate(self, attention_mask, input_ids, user_id=None):
        """Generate deterministic exact top-k valid semantic IDs."""
        batch_size = input_ids.size(0)
        self.last_generation_max_expansion_score_elements = 0

        encoder_output, encoder_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )
        generated = None
        log_probas = None
        beam_prefixes = [[()] for _ in range(batch_size)]
        past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())

        for hierarchy in range(self.num_hierarchies):
            beam_count = len(beam_prefixes[0])
            if generated is None:
                current_encoder = encoder_output
                current_mask = encoder_mask
                decoder_ids = None
            else:
                current_encoder = encoder_output.repeat_interleave(beam_count, dim=0)
                current_mask = encoder_mask.repeat_interleave(beam_count, dim=0)
                decoder_ids = generated.reshape(-1, hierarchy)

            decoder_output, past_kv = self.decoder_forward_pass(
                future_ids=decoder_ids,
                encoder_output=current_encoder,
                attention_mask_for_encoder=current_mask,
                use_cache=True,
                past_key_values=past_kv,
            )
            probabilities = F.softmax(
                self.decoder_mlp[hierarchy](decoder_output[:, -1, :]),
                dim=-1,
            )
            (
                new_tokens,
                parent_beam_indices,
                log_probas,
                beam_prefixes,
            ) = self._select_valid_expansions(
                probabilities=probabilities,
                beam_prefixes=beam_prefixes,
                log_probas=log_probas,
                hierarchy=hierarchy,
            )

            if generated is None:
                generated = new_tokens.unsqueeze(-1)
                past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
                continue

            parent_global_indices = (
                parent_beam_indices
                + torch.arange(
                    batch_size, device=parent_beam_indices.device
                ).unsqueeze(1)
                * beam_count
            ).flatten()
            past_kv.reorder_cache(parent_global_indices)
            parent_ids = torch.gather(
                generated,
                1,
                parent_beam_indices.unsqueeze(-1).expand(-1, -1, hierarchy),
            )
            generated = torch.cat([parent_ids, new_tokens.unsqueeze(-1)], dim=-1)

        return generated, log_probas

    @torch.no_grad()
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        top_k: bool = True,
        temperature: int = 1,
    ) -> GenerationOutput:
        input_ids = batch.sem_ids
        attention_mask = batch.seq_mask.long()
        generated_ids, log_probas = self.generate(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=batch.user_ids,
        )
        return GenerationOutput(sem_ids=generated_ids, log_probas=log_probas)
