import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field, asdict
from typing import Optional

from .ap_embed_methods import get_ap_method, LearnableShuffledEmbeds
from .normalization_methods import get_normalization_method


@dataclass
class EmbedderConfig:
    tie_embeddings: bool = True
    pad_vocab_size_multiple: int = 8

    # Number of dimensions for the Atomic Propositions
    d_ap: int = 0
    ap_embed: str = "randn"
    base_normalization: str = "l2"
    ap_normalization: str = "l2"
    final_normalization: str = "l2"

    feature_normalization: str = "disabled"
    
    embed_scaling: Optional[str] = None

    shuffle_aps: Optional[int] = None

    def build(self, d_model, vocab, **factory_kwargs):
        """
        Build and return the embedder.
        """
        return DynamicEmbedder(
            d_model=d_model,
            d_ap=self.d_ap,
            vocab=vocab,
            ap_method=self.ap_embed,
            base_normalization=self.base_normalization,
            ap_normalization=self.ap_normalization,
            final_normalization=self.final_normalization,
            feature_normalization=self.feature_normalization,
            embed_scaling=self.embed_scaling,
            shuffle_aps=self.shuffle_aps,
            **factory_kwargs,
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            d_ap = args.d_ap,
            ap_embed = args.ap_embed,
            base_normalization = args.embed_base_normalization,
            ap_normalization = args.embed_ap_normalization,
            final_normalization = args.embed_final_normalization,
            feature_normalization = args.feature_normalization,
            embed_scaling = args.embed_scaling,
            shuffle_aps = args.shuffle_aps,
        )


class EmbedScaler(nn.Module):
    def __init__(self, method, d_model, **factory_kwargs):
        super().__init__()
        if method is None:
            self.forward = lambda x: x
        elif method == "learnable":
            self.multiplier = nn.parameter.Parameter(torch.empty((1), **factory_kwargs))
            self.forward = lambda x: x * self.multiplier
        elif method == "sqrtd":
            # sqrt of embeddings dim
            multiplier = math.sqrt(d_model)
            self.forward = lambda x: x * multiplier
        else:
            raise ValueError(f"Unknown EmbedScaler method: {method}")

    def reset_parameters(self) -> None:
        try:
            self.multiplier.copy_(1.0)
            print("Reset embed:", self.multiplier)
        except AttributeError:
            pass


class DynamicEmbedder(nn.Module):
    def __init__(
            self,
            d_model,
            d_ap,
            vocab,
            ap_method,
            base_normalization="l2",
            ap_normalization="l2",
            final_normalization="l2",
            feature_normalization="disabled",
            embed_scaling: Optional[str] = None,
            shuffle_aps: Optional[int] = None,
            **factory_kwargs,
        ):
        self.vocab = vocab
        self.d_model = d_model
        self.d_ap = d_ap
        self.base_normalization = get_normalization_method(base_normalization)
        self.ap_normalization = get_normalization_method(ap_normalization)
        self.final_normalization = get_normalization_method(final_normalization)
        self.feature_normalization = get_normalization_method(feature_normalization)
        self.shuffle_aps = shuffle_aps

        super().__init__()

        if vocab.dynamic_aps:
            if shuffle_aps is not None:
                assert d_ap == 0
                assert shuffle_aps > 0
                vocab_size = vocab.size() - 26  # make sure no tokens for AP
                self.base_weight = nn.parameter.Parameter(torch.empty((vocab_size, d_model), **factory_kwargs))
                self.ap_weight = nn.parameter.Parameter(torch.empty((shuffle_aps, d_model), **factory_kwargs))
            else:
                assert d_ap > 0
                vocab_size = vocab.size() - 25  # make sure only one token for AP
                d_base_embed = d_model - d_ap
                self.base_weight = nn.parameter.Parameter(torch.empty((vocab_size, d_base_embed), **factory_kwargs))
                self.base_vocab_size = vocab_size
                self.d_base_embed = d_base_embed
                if ap_method.startswith("learnable"):
                    ap_method_count = int(ap_method[len("learnable"):])
                    self.ap_method = LearnableShuffledEmbeds(ap_method_count, self.d_ap, **factory_kwargs)
                else:
                    self.ap_method = get_ap_method(ap_method)
        else:
            vocab_size = vocab.size()
            self.base_weight = nn.parameter.Parameter(torch.empty((vocab_size, d_model), **factory_kwargs))

        # Effective weights of the embedding/projection matrix
        self.w = None

        self.embed_scaler = EmbedScaler(embed_scaling, d_model, **factory_kwargs)

        self.reset_parameters()
        self.prepare()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.base_weight)
        if (ap_weight := getattr(self, "ap_weight", None)) is not None:
            torch.nn.init.normal_(ap_weight)
        self.embed_scaler.reset_parameters()
        if hasattr(self, "ap_method") and hasattr(self.ap_method, "reset_parameters"):
            self.ap_method.reset_parameters()
    
    def prepare(self):
        if not self.vocab.dynamic_aps:
            w = self.base_weight
        elif self.shuffle_aps is not None:
            ap_count = len(self.vocab.aps)
            assert ap_count <= self.shuffle_aps
            ap_perm = torch.randperm(self.shuffle_aps, device=self.base_weight.device)[:ap_count]
            w = torch.cat((
                self.base_weight,
                self.ap_weight[ap_perm],
            ), dim=0)
        else:
            ap_count = len(self.vocab.aps)
            # Base vocab has operators and one base embedding for APs
            vocab_size = self.base_vocab_size -1 + ap_count

            factory_kwargs = {"device": self.base_weight.device, "dtype": self.base_weight.dtype}
            w = torch.zeros(vocab_size, self.d_model, **factory_kwargs)
            w[:self.base_vocab_size, :self.d_base_embed] = self.base_normalization(self.base_weight)
            # Replicate the last base embedding (which is the common base embedding for APs) for all APs
            w[self.base_vocab_size:, :self.d_base_embed] = w[self.base_vocab_size - 1, :self.d_base_embed]

            # Generate and set the AP embeddings
            ap_embeds = self.ap_method(ap_count, self.d_ap, **factory_kwargs)
            ap_embeds = self.ap_normalization(ap_embeds)
            w[self.base_vocab_size-1:, self.d_base_embed:] = ap_embeds

        # self.w rows must be normalized (L2 or another)
        # Better: normalize the base embeddings, then normalize the AP embeddings (d_ap), then normalize all (self.w)
        # Because we have two sides to balance: constant base embeddings and random AP embeddings
        # They should not override each other
        self.w = self.final_normalization(w)

    def shrink_w(self):
        """
        Resize the w matrix to the current vocab size.
        Returns the number of tokens removed.
        """
        if not self.vocab.dynamic_aps:
            raise ValueError("Cannot shrink w matrix if vocab does not have dynamic APs")
        if self.shuffle_aps is not None:
            vocab_size = self.base_weight.size(0) + len(self.vocab.aps)
        else:
            # Base vocab has operators and one base embedding for APs
            vocab_size = self.base_vocab_size -1 + len(self.vocab.aps)
        old_vocab_size = self.w.size(0)
        self.w = self.w[:vocab_size, :]
        return old_vocab_size - vocab_size
    
    def embed(self, input_ids):
        return self.embed_scaler(F.embedding(input_ids, self.w))
    
    def project(self, hidden):
        hidden = self.feature_normalization(hidden)
        logits = F.linear(hidden, self.w)
        # Disallow start/pad tokens
        if self.vocab.use_start_token:
            logits[..., self.vocab.start_id] = -float("inf")
        if self.vocab.use_pad_token:
            logits[..., self.vocab.pad_id] = -float("inf")
        return logits
    
    def _get_output_vocab_size(self):
        return self.w.size(0)
    
    output_vocab_size = property(fget=_get_output_vocab_size)