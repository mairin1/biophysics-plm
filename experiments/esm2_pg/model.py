"""ESM-2 encoder, regression head, and the three fine-tuning strategies.

The strategies mirror experiments/checkin2_v3/train.py, re-expressed for a
transformer stack instead of EvoDiff's ByteNet blocks:

  GENTLE   encoder learning rate reduced tenfold (2e-6 instead of 2e-5)
  L2SP     standard learning rate plus a penalty on normalised squared
           displacement from the pretrained weights
  PARTIAL  token embeddings and the first 75% of transformer layers frozen;
           the remaining layers, the final layer norm, and the head train
"""

import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EMBED_DIM, ENCODER_LR_GENTLE, ENCODER_LR_STANDARD, ESM2_MODEL, HEAD_LR,
    N_LAYERS, PARTIAL_TRAINABLE_FRACTION, WEIGHT_DECAY, run_dir,
)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# Ambiguity codes that appear in some ProteinGym target sequences. They never
# occur in the METL training data, but omitting them would make us drop whole
# assays at embedding time.
AMBIGUOUS_RESIDUES = 'XBUZO'


class TokenizerShim:
    """Exposes ESM-2 token ids through the `a_to_i` interface that the METL
    loader in scripts/train_evodiff_stage1.py expects."""

    def __init__(self, hf_tokenizer):
        self.hf = hf_tokenizer
        self.a_to_i = {}
        for aa in AMINO_ACIDS:
            token_id = hf_tokenizer.convert_tokens_to_ids(aa)
            if token_id is None or token_id == hf_tokenizer.unk_token_id:
                raise RuntimeError(f'ESM-2 tokenizer has no id for residue {aa}')
            if token_id > 255:
                raise RuntimeError('Residue token ids must fit in uint8 storage')
            self.a_to_i[aa] = token_id
        for aa in AMBIGUOUS_RESIDUES:
            token_id = hf_tokenizer.convert_tokens_to_ids(aa)
            if token_id is not None and token_id != hf_tokenizer.unk_token_id:
                self.a_to_i[aa] = token_id
        self.cls_id = hf_tokenizer.cls_token_id
        self.eos_id = hf_tokenizer.eos_token_id
        self.pad_id = hf_tokenizer.pad_token_id
        if self.cls_id is None or self.eos_id is None:
            raise RuntimeError('ESM-2 tokenizer is missing <cls>/<eos>')


def load_encoder():
    """Load the ESM-2 encoder and a tokenizer shim.

    The published checkpoint is an EsmForMaskedLM, so loading EsmModel prints
    five `lm_head.*` keys as UNEXPECTED. That is expected: we only mean-pool
    hidden states and never score tokens, so the masked-LM head is discarded.
    A MISSING section would be the real problem, since it would mean encoder
    weights were randomly initialised rather than loaded.

    Restoring the head is what a ProteinGym zero-shot or masked-token
    retention check would need; load EsmForMaskedLM for that.
    """
    from transformers import AutoTokenizer, EsmModel
    hf_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    encoder = EsmModel.from_pretrained(ESM2_MODEL, add_pooling_layer=False)
    layers = encoder.encoder.layer
    if len(layers) != N_LAYERS:
        raise RuntimeError(f'Expected {N_LAYERS} ESM-2 layers, found {len(layers)}')
    if encoder.config.hidden_size != EMBED_DIM:
        raise RuntimeError(
            f'Expected hidden size {EMBED_DIM}, found {encoder.config.hidden_size}'
        )
    return encoder, TokenizerShim(hf_tokenizer)


class Esm2Predictor(nn.Module):
    def __init__(self, encoder, n_targets, cls_id, eos_id):
        super().__init__()
        self.encoder = encoder
        self.cls_id = cls_id
        self.eos_id = eos_id
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_targets),
        )

    def embed(self, residue_tokens, attention_mask=None):
        """Mean-pool the final hidden states over residues only.

        `residue_tokens` holds residue ids without <cls>/<eos>; both are added
        here so the input matches the pretraining format. Special tokens are
        excluded from the pooled average.
        """
        batch = residue_tokens.shape[0]
        cls = residue_tokens.new_full((batch, 1), self.cls_id)
        eos = residue_tokens.new_full((batch, 1), self.eos_id)
        tokens = torch.cat([cls, residue_tokens, eos], dim=1)
        if attention_mask is None:
            residue_mask = torch.ones_like(residue_tokens, dtype=torch.long)
        else:
            residue_mask = attention_mask.long()
        ones = residue_mask.new_ones((batch, 1))
        full_mask = torch.cat([ones, residue_mask, ones], dim=1)
        hidden = self.encoder(input_ids=tokens, attention_mask=full_mask).last_hidden_state
        residues = hidden[:, 1:-1, :]
        weights = residue_mask.unsqueeze(-1).to(residues.dtype)
        return (residues * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    def forward(self, residue_tokens, attention_mask=None):
        return self.head(self.embed(residue_tokens, attention_mask))


def set_strategy(model, method):
    """Configure which pretrained parameters train. Returns a description."""
    encoder = model.encoder
    layers = encoder.encoder.layer
    final_norm = encoder.encoder.emb_layer_norm_after

    if method == 'PARTIAL':
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        n_trainable = max(1, math.ceil(len(layers) * PARTIAL_TRAINABLE_FRACTION))
        for layer in list(layers)[-n_trainable:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for parameter in final_norm.parameters():
            parameter.requires_grad_(True)
        detail = {
            'total_layers': len(layers),
            'trainable_last_layers': n_trainable,
            'frozen_first_layers': len(layers) - n_trainable,
            'token_embedding_frozen': True,
            'final_layer_norm_trainable': True,
        }
    else:
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
        detail = {
            'total_layers': len(layers),
            'full_encoder_trainable': True,
            'final_layer_norm_trainable': True,
        }

    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    return detail


def optimizer_for(model, method):
    encoder_lr = ENCODER_LR_GENTLE if method == 'GENTLE' else ENCODER_LR_STANDARD
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if not encoder_params or not head_params:
        raise RuntimeError('A trainable parameter group is unexpectedly empty')
    optimizer = torch.optim.AdamW(
        [{'params': encoder_params, 'lr': encoder_lr},
         {'params': head_params, 'lr': HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer, encoder_lr


def l2sp_penalty(model, anchor, denom):
    diffs = [
        (p.float() - anchor[name]).square().sum()
        for name, p in model.named_parameters()
        if name in anchor and p.requires_grad
    ]
    if not diffs:
        raise RuntimeError('L2-SP has no trainable pretrained parameters')
    return torch.stack(diffs).sum() / denom


def load_condition(condition, seed, device='cuda'):
    """Encoder for evaluation: pretrained for BASELINE, else from a checkpoint."""
    encoder, tokenizer = load_encoder()
    model = Esm2Predictor(encoder, 1, tokenizer.cls_id, tokenizer.eos_id)
    if condition != 'BASELINE':
        path = run_dir(condition, seed) / 'best_model.pt'
        if not path.is_file():
            raise RuntimeError(f'Missing fine-tuned checkpoint: {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint['model_state_dict']
        encoder_state = {
            key[len('encoder.'):]: value
            for key, value in state.items() if key.startswith('encoder.')
        }
        if not encoder_state:
            raise RuntimeError(f'Checkpoint has no encoder weights: {path}')
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model.to(device).eval(), tokenizer
