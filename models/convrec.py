from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax as F_softmax

from torch import Tensor, LongTensor, cat as torch_cat

from tools.utils import fix_random_seed
from .layers import LayerNorm
from .encoders import ConvItemEncoder

__all__ = ("ConvRec",)


class ConvolutionStack(nn.Module):
    def __init__(self, embed_dim: int, num_conv_heads: int, conv_params: list, input_length: int, cxt_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pooling_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.layer_norms_out = nn.ModuleList()
        self.Res_weights1 = nn.ParameterList()
        self.Res_weights2 = nn.ParameterList()
        self.dropout = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()
        self.padding_params = []  # left padding per layer

        current_length = input_length
        for (kernel_size, stride) in conv_params:
            if current_length >= kernel_size:
                remainder = (current_length - kernel_size) % stride
                pad = 0 if remainder == 0 else stride - remainder
            else:
                pad = kernel_size - current_length
            self.padding_params.append(pad)

            self.pooling_layers.append(nn.AvgPool1d(kernel_size=kernel_size, stride=stride))

            self.Res_weights1.append(nn.Parameter(torch.tensor(0.5)))
            self.Res_weights2.append(nn.Parameter(torch.tensor(0.5)))

            self.conv_layers.append(
                MultiHeadConvolution(embed_dim, num_conv_heads=num_conv_heads, kernel_size=kernel_size, stride=stride, dropout=dropout)
            )
            self.layer_norms.append(LayerNorm(embed_dim))
            self.layer_norms_out.append(LayerNorm(embed_dim))

            current_length = (current_length + pad - kernel_size) // stride + 1

        self.collapse = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor, x_cxt: torch.Tensor) -> torch.Tensor:  # x_cxt kept for signature parity
        x_res = x
        for pool, conv, pad, layernorm, w1, w2 in zip(
            self.pooling_layers, self.conv_layers, self.padding_params, self.layer_norms, self.Res_weights1, self.Res_weights2
        ):
            if pad > 0:
                x = F.pad(x.transpose(1, 2), (pad, 0)).transpose(1, 2)
                x_res = F.pad(x_res.transpose(1, 2), (pad, 0)).transpose(1, 2)

            x_pool = pool(x.transpose(1, 2)).transpose(1, 2)
            x = conv(x)
            x_res = pool(x_res.transpose(1, 2)).transpose(1, 2)

            x = self.gelu(x)
            x = self.dropout(x)
            x = layernorm(x + w1 * x_pool + w2 * x_res)

        return self.collapse(x.transpose(1, 2))  # (B, D, 1)


class MultiHeadConvolution(nn.Module):
    def __init__(self, embed_dim: int, num_conv_heads: int, kernel_size: int, stride: int, dropout: float) -> None:
        super().__init__()
        assert embed_dim % num_conv_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_conv_heads = num_conv_heads
        self.head_dim = embed_dim // num_conv_heads
        self.kernel_size = kernel_size

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                kernel_size=kernel_size,
                stride=stride,
            )
            for _ in range(num_conv_heads)
        ])
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_conv_heads, self.head_dim)
        return x.transpose(1, 2)  

    def Conv_layer(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = []
        for h, conv in enumerate(self.convs):
            head = x[:, h, :, :].transpose(1, 2)  
            out_h = conv(head).transpose(1, 2)    
            outputs.append(out_h)
        return torch_cat(outputs, dim=-1)           

    def forward(self, input: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = input.size(0)
        input_split = self.split_heads(input, B)
        conv_output = self.Conv_layer(input_split, mask)
        return self.out_linear(conv_output)


class ConvRec(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        num_items: int,
        num_users: int,
        ifeatures: np.ndarray,
        ifeature_dim: int,
        icontext_dim: int,
        hidden_dim: int = 256,
        num_known_item: Optional[int] = None,
        dropout_prob: float = 0.1,
        random_seed: Optional[int] = None,
        num_conv_heads: int = 1,
        conv_params: list = [[2, 2], [5, 5], [7, 7]],
    ) -> None:
        super().__init__()
        self.sequence_len = sequence_len
        self.num_items = num_items
        self.num_users = num_users
        self.ifeature_dim = ifeature_dim
        self.icontext_dim = icontext_dim
        self.hidden_dim = hidden_dim
        self.num_known_item = num_known_item
        self.dropout_prob = dropout_prob
        self.random_seed = random_seed
        self.num_conv_heads = num_conv_heads
        self.conv_params = conv_params

        if random_seed is not None:
            fix_random_seed(random_seed)

        self.item_encoder = ConvItemEncoder(
            sequence_len=sequence_len,
            num_items=num_items,
            num_users=num_users,
            ifeatures=ifeatures,
            ifeature_dim=ifeature_dim,
            icontext_dim=icontext_dim,
            hidden_dim=hidden_dim,
            num_known_item=num_known_item,
            random_seed=random_seed,
            dropout_prob=dropout_prob,
        )
        self.item_layernorm = LayerNorm(hidden_dim + (icontext_dim - 3))

        self.in_lin = nn.Linear(hidden_dim + (icontext_dim - 3), hidden_dim)
        self.cxt_layer = nn.Linear(20, hidden_dim)
        self.time_linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.stacked_multihead_conv = ConvolutionStack(
            embed_dim=hidden_dim,
            num_conv_heads=num_conv_heads,
            conv_params=conv_params,
            input_length=sequence_len,
            cxt_dim=icontext_dim,
            dropout=dropout_prob,
        )

    def forward(
        self,
        user_index: Tensor,
        profile_tokens: LongTensor,
        profile_icontexts: Tensor,
        extract_tokens: LongTensor,
        extract_icontexts: Tensor,
    ) -> torch.Tensor:

        profile_token_mask = (profile_tokens > 0).unsqueeze(-1).repeat(1, 1, self.hidden_dim)

        P, ac_vector_comb, icontexts, users_vector = self.item_encoder(
            user_index,
            profile_tokens,
            profile_icontexts,
            item_type="profile",
        )
        icontexts = icontexts[:, :, :3]
        interval = torch.diff(icontexts, dim=1)
        one = torch.ones(icontexts.size(0), 1, icontexts.size(2), device=icontexts.device, dtype=icontexts.dtype)
        T_diff = torch.cat([one, interval], dim=1)

        P_T_diff = torch.cat([P, T_diff], dim=-1)

        P_T_diff = F.gelu(P_T_diff)
        P_T_diff = self.dropout(P_T_diff)
        P_T_diff = self.item_layernorm(P_T_diff)
        P_T_diff = self.in_lin(P_T_diff)
        P_T_diff = F.gelu(P_T_diff)
        P_T_diff = P_T_diff * profile_token_mask

        cxt_enc = self.cxt_layer(torch.unsqueeze(users_vector, 1).repeat(1, P.size(1), 1))
        cxt_enc = cxt_enc * profile_token_mask

        P = self.stacked_multihead_conv(P_T_diff, cxt_enc)
        E, _, E_cxt, _ = self.item_encoder(
            user_index,
            extract_tokens,
            extract_icontexts,
            item_type="extract",
        )
        E_cxt = E_cxt[:, :, :3]
        target_diff = E_cxt - icontexts[:, -1:, :]
        E = torch.cat([E, target_diff], dim=-1)
        E = F.gelu(E)
        E = self.item_layernorm(E)
        E = self.in_lin(E)
        E = F.gelu(E)

        Y = torch.matmul(E, P)
        logits = Y.squeeze(-1)

        return logits
