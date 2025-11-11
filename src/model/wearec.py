import torch
import torch.nn as nn
import copy
from model._abstract_model import SequentialRecModel
from model._modules import  LayerNorm,FeedForward
import torch.nn.functional as F

class WEARecLayer(nn.Module):
    def __init__(self, args , combine_mode = 'gate' ,num_heads=4, dropout=0.1, adaptive=True):
        super(WEARecLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        
        if args.hidden_size % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = args.num_heads
        self.head_dim = args.hidden_size // self.num_heads
        self.seq_len = args.max_seq_length
        self.adaptive = adaptive
        self.combine_mode = 'gate'
        # Frequency bins for rFFT: (seq_len//2 + 1)
        self.freq_bins = args.max_seq_length // 2  + 1
        self.alpha = args.alpha
        self.complex_weight = nn.Parameter(torch.randn(1,self.num_heads, args.max_seq_length//2 , self.head_dim, dtype=torch.float32) * 0.02)
        # Base multiplicative filter: one per head and frequency bin.
        self.base_filter = nn.Parameter(torch.ones(self.num_heads, self.freq_bins, 1))
        # Base additive bias (think of it as a learned offset on the frequency magnitudes).
        self.base_bias = nn.Parameter(torch.full((self.num_heads, self.freq_bins, 1), -0.1))

        if adaptive:
            # Adaptive MLP: produces 2 values per head & frequency bin (scale and bias modulation).
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.GELU(),
                nn.Linear(args.hidden_size, self.num_heads * self.freq_bins * 2)
            )
        
        if self.combine_mode == 'gate':
            # We'll learn a single gate that is broadcast across batch, heads, and positions.
            self.gate_param = nn.Parameter(torch.tensor(0.0))
            self.proj_concat = None  # Not used in gating mode
        elif self.combine_mode == 'concat':
            # A projection to bring concatenated (local + global) features back to embed_dim
            self.proj_concat = nn.Linear(2 * args.hidden_size, args.hidden_size)
            self.gate_param = None
        else:
            raise ValueError("combine_mode must be either 'gate' or 'concat'")
            
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
    
    def wavelet_transform(self, x_heads):
        """
        Applies a single-level Haar wavelet transform (decomposition + reconstruction)
        to capture local dependencies along the sequence dimension.

        Args:
          x_heads: Tensor of shape (B, num_heads, seq_len, head_dim)

        Returns:
          Reconstructed wavelet-based features of the same shape (B, num_heads, seq_len, head_dim).
        """
        B, H, N, D = x_heads.shape

        # For simplicity, if N is odd, truncate by one
        N_even = N if (N % 2) == 0 else (N - 1)
        x_heads = x_heads[:, :, :N_even, :]  # shape -> (B, H, N_even, D)

        # Split even and odd positions along sequence dimension
        x_even = x_heads[:, :, 0::2, :]  # (B, H, N_even/2, D)
        x_odd  = x_heads[:, :, 1::2, :]  # (B, H, N_even/2, D)

        # Haar wavelet decomposition
        # approx = 0.5*(even + odd), detail = 0.5*(even - odd)
        approx = 0.5 * (x_even + x_odd) 
        detail = 0.5 * (x_even - x_odd)

        # A nonlinearity can optionally be applied to approx/detail
        
        detail = detail * self.complex_weight

        # Haar wavelet reconstruction
        # even' = approx + detail, odd' = approx - detail
        x_even_recon = approx + detail
        x_odd_recon  = approx - detail

        # Interleave even/odd back to original shape
        out = torch.zeros_like(x_heads)
        out[:, :, 0::2, :] = x_even_recon
        out[:, :, 1::2, :] = x_odd_recon

        # If we truncated one position, pad it back with zeros
        if N_even < N:
            pad = torch.zeros((B, H, 1, D), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=2)

        return out

    
    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        

        # Reshape to separate heads: (B, num_heads, seq_len, head_dim)
        x_heads = input_tensor.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---- (1) FFT-based global features ----
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')  # shape (B, num_heads, freq_bins, head_dim)

        # Compute adaptive modulation parameters if enabled.
        if self.adaptive:
            # Global context: average over tokens (B, embed_dim)
            context = input_tensor.mean(dim=1)
            # Adaptive MLP outputs 2 values per (head, frequency bin).
            adapt_params = self.adaptive_mlp(context)  # (B, num_heads*freq_bins*2)
            adapt_params = adapt_params.view(batch, self.num_heads, self.freq_bins, 2)
            # Split into multiplicative and additive modulations.
            adaptive_scale = adapt_params[..., 0:1]  # shape: (B, num_heads, freq_bins, 1)
            adaptive_bias  = adapt_params[..., 1:2]  # shape: (B, num_heads, freq_bins, 1)
        else:
            # If not adaptive, set modulation to neutral (scale=0, bias=0).
            adaptive_scale = torch.zeros(batch, self.num_heads, self.freq_bins, 1, device=x.device)
            adaptive_bias  = torch.zeros(batch, self.num_heads, self.freq_bins, 1, device=x.device)
            

        # Combine base parameters with adaptive modulations
        effective_filter = self.base_filter * (1 + adaptive_scale)  # (num_heads, freq_bins, 1) broadcast with (B, ...)
        effective_bias   = self.base_bias + adaptive_bias

        # Apply modulations in the frequency domain
        F_fft_mod = F_fft * effective_filter + effective_bias



        # Inverse FFT to bring data back to token space
        x_fft = torch.fft.irfft(F_fft_mod, dim=2, n=self.seq_len, norm='ortho')  # (B, num_heads, seq_len, head_dim)

        # ---- (2) Wavelet-based local features ----
        x_wavelet = self.wavelet_transform(x_heads)  # (B, num_heads, seq_len, head_dim)

        # ---- (3) Combine local/global ----
        if self.combine_mode == 'gate':
            # Gate in [0,1] after a sigmoid
            alpha = self.alpha
            # Blend wavelet and FFT features
            x_combined = (1.0 - alpha) * x_wavelet +  alpha * x_fft
        else:
            # Concatenate along the embedding dimension
            # First, reshape each to (B, seq_len, num_heads*head_dim) = (B, N, D)
            x_wavelet_reshaped = x_wavelet.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)
            x_fft_reshaped     = x_fft.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)
            x_cat = torch.cat([x_wavelet_reshaped, x_fft_reshaped], dim=-1)  # (B, N, 2*D)
            # Project back down to D
            x_combined = self.proj_concat(x_cat)
            # Reshape back to (B, num_heads, seq_len, head_dim) if we want to keep the same path
            x_combined = x_combined.view(batch, seq_len, -1).view(batch, seq_len, self.num_heads, self.head_dim)
            # Permute back to (B, num_heads, seq_len, head_dim)
            x_combined = x_combined.permute(0, 2, 1, 3)

            
        # Reshape: merge heads back into the embedding dimension.
        x_out = x_combined.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)
        
        hidden_states = self.out_dropout(x_out)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        
        
        return hidden_states


class WEARecBlock(nn.Module):
    def __init__(self, args):
        super(WEARecBlock, self).__init__()
        self.layer = WEARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states):
        layer_output = self.layer(hidden_states)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class WEARecEncoder(nn.Module):
    def __init__(self, args):
        super(WEARecEncoder, self).__init__()
        self.args = args
        block = WEARecBlock(args)

        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=False):

        all_encoder_layers = [ hidden_states ]

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states,)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])

        return all_encoder_layers

class WEARecModel(SequentialRecModel):
    def __init__(self, args):
        super(WEARecModel, self).__init__(args)

        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.item_encoder = WEARecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss
