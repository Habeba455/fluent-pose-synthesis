from typing import Optional, Dict
import torch
import torch.nn as nn

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess


class OutputProcessMLP(nn.Module):
    """
    Input:  x: [T, B, latent_dim]
    Output: [B, K, D, T]
    """

    def __init__(
        self,
        input_feats: int,
        latent_dim: int,
        njoints: int,
        nfeats: int,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, input_feats),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"OutputProcessMLP expected [T, B, D], got {tuple(x.shape)}")

        t, b, d = x.shape
        if d != self.latent_dim:
            raise ValueError(
                f"OutputProcessMLP latent dim mismatch: got {d}, expected {self.latent_dim}"
            )

        x = self.mlp(x)                                   # [T, B, input_feats]
        x = x.reshape(t, b, self.njoints, self.nfeats)    # [T, B, K, D]
        x = x.permute(1, 2, 3, 0).contiguous()            # [B, K, D, T]
        return x


class SignLanguagePoseDiffusion(nn.Module):
    """
    Diffusion denoiser for sign-language pose generation.

    Expected input shapes:
        fluent_clip   : [B, K, D, T_chunk]   noisy sample x_t
        disfluent_seq : [B, K, D, T_cond]    condition
        previous_output: [B, K, D, T_hist]   optional history
        t             : [B]

    Output: [B, K, D, T_chunk]

    Fixes vs original:
      - Separate PositionalEncoding per stream → each starts at position 0.
      - Time embedding no longer gets PE applied twice.
      - Per-stream classifier-free-guidance with learnable null embeddings.
      - forward_with_cfg() for guided sampling at inference.
      - fusion_proj wrapped in a residual + LayerNorm block.
      - GRU honours self.batch_first (and is bidirectional).
      - Device transfers removed from forward (do them in the data loader).
      - Default dropout lowered.
      - Unused `ablation`, `legacy` flags removed.
    """

    def __init__(
        self,
        input_feats: int,
        chunk_len: int,
        keypoints: int,
        dims: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,              # ★ was 0.2
        activation: str = "gelu",
        arch: str = "trans_enc",
        cond_mask_prob: float = 0.1,       # ★ sensible CFG default
        batch_first: bool = True,
    ):
        super().__init__()

        if arch not in {"trans_enc", "trans_dec", "gru"}:
            raise ValueError("arch must be one of ['trans_enc', 'trans_dec', 'gru']")

        self.input_feats = input_feats
        self.chunk_len = chunk_len
        self.keypoints = keypoints
        self.dims = dims
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.arch = arch
        self.cond_mask_prob = cond_mask_prob
        self.batch_first = batch_first

        # ★ FIX: separate positional encoders per stream. Each stream starts
        # at position 0 so "cond frame 0" isn't treated as "frame T_chunk+1".
        self.pe_noisy = PositionalEncoding(d_model=latent_dim, dropout=dropout)
        self.pe_cond = PositionalEncoding(d_model=latent_dim, dropout=dropout)
        self.pe_hist = PositionalEncoding(d_model=latent_dim, dropout=dropout)

        # TimestepEmbedder applies its own PE internally — we do NOT apply
        # another PE on top of it.
        self.embed_timestep = TimestepEmbedder(latent_dim, self.pe_noisy)

        # Separate encoders for separate semantic roles.
        self.noisy_encoder = MotionProcess(input_feats, latent_dim)
        self.disfluent_encoder = MotionProcess(input_feats, latent_dim)
        self.history_encoder = MotionProcess(input_feats, latent_dim)

        # ★ FIX: learnable null embeddings for classifier-free guidance.
        # Shape [1, 1, D] broadcasts over either [B, T, D] or [T, B, D].
        self.null_disfluent = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.null_history = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.null_disfluent, std=0.02)
        nn.init.normal_(self.null_history, std=0.02)

        if self.arch == "trans_enc":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
                norm_first=True,              # ★ pre-LN is more stable
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

        elif self.arch == "trans_dec":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers,
            )

        else:  # gru
            # ★ FIX: honour self.batch_first (was hard-coded True).
            # Also bidirectional — denoisers need full context.
            self.sequence_encoder = nn.GRU(
                input_size=latent_dim,
                hidden_size=latent_dim,
                num_layers=num_layers,
                batch_first=batch_first,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.gru_proj = nn.Linear(latent_dim * 2, latent_dim)

        # ★ Residual post-projection with LayerNorm.
        self.post_norm = nn.LayerNorm(latent_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.pose_projection = OutputProcessMLP(
            input_feats=input_feats,
            latent_dim=latent_dim,
            njoints=keypoints,
            nfeats=dims,
            hidden_dim=1024,
        )

    # -----------------------------------------------------------------
    # Encoding helpers
    # -----------------------------------------------------------------
    def _encode_sequence(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x: [B, K, D, T]
        Output: if batch_first=True  -> [B, T, latent_dim]
                else                 -> [T, B, latent_dim]
        Assumes MotionProcess returns [T, B, latent_dim].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected [B, K, D, T], got {tuple(x.shape)}")

        x_raw = encoder(x)
        if x_raw.dim() != 3:
            raise ValueError(f"Unexpected encoder output shape: {tuple(x_raw.shape)}")
        if x_raw.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Encoder latent dim mismatch: got {x_raw.shape[-1]}, expected {self.latent_dim}"
            )

        if self.batch_first:
            return x_raw.permute(1, 0, 2).contiguous()   # [B, T, D]
        return x_raw.contiguous()                        # [T, B, D]

    def _prepare_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns: if batch_first=True  -> [B, 1, D]
                 else                 -> [1, B, D]
        NOTE: TimestepEmbedder already applies PE internally. We do NOT
        re-apply it.
        """
        t_emb = self.embed_timestep(t)

        if t_emb.dim() == 2:
            t_emb = t_emb.unsqueeze(0)                    # [1, B, D]
        elif t_emb.dim() != 3:
            raise ValueError(f"Unexpected timestep embedding shape: {tuple(t_emb.shape)}")

        if t_emb.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Timestep embedding latent dim mismatch: got {t_emb.shape[-1]}, expected {self.latent_dim}"
            )

        if self.batch_first:
            return t_emb.permute(1, 0, 2).contiguous()   # [B, 1, D]
        return t_emb.contiguous()                        # [1, B, D]

    # -----------------------------------------------------------------
    # CFG masking helpers
    # -----------------------------------------------------------------
    def _apply_stream_mask(
        self,
        tokens: torch.Tensor,          # [B, T, D] or [T, B, D]
        null_embed: nn.Parameter,      # [1, 1, D]
        drop_prob: float,
    ) -> torch.Tensor:
        """
        Per-sample, per-stream mask: replaces a sample's tokens with the
        learnable null embedding with probability `drop_prob`. Independent
        across streams and across samples.
        """
        if drop_prob <= 0.0 or not self.training:
            return tokens

        if self.batch_first:
            B = tokens.shape[0]
            drop = (torch.rand(B, device=tokens.device) < drop_prob)    # [B]
            if not drop.any():
                return tokens
            drop = drop.view(B, 1, 1).to(tokens.dtype)                  # [B, 1, 1]
        else:
            B = tokens.shape[1]
            drop = (torch.rand(B, device=tokens.device) < drop_prob)    # [B]
            if not drop.any():
                return tokens
            drop = drop.view(1, B, 1).to(tokens.dtype)                  # [1, B, 1]

        null = null_embed.expand_as(tokens)
        return tokens * (1.0 - drop) + null * drop

    # -----------------------------------------------------------------
    # PE helper (handles batch_first internally since CAMDM PE wants [T,B,D])
    # -----------------------------------------------------------------
    def _apply_pe(self, tokens: torch.Tensor, pe_module: nn.Module) -> torch.Tensor:
        if self.batch_first:
            # [B, T, D] -> [T, B, D] -> PE -> [B, T, D]
            tokens_tb = tokens.permute(1, 0, 2).contiguous()
            tokens_tb = pe_module(tokens_tb)
            return tokens_tb.permute(1, 0, 2).contiguous()
        return pe_module(tokens)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        fluent_clip: torch.Tensor,
        disfluent_seq: torch.Tensor,
        t: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
        force_drop_disfluent: bool = False,
        force_drop_history: bool = False,
    ) -> torch.Tensor:
        """
        fluent_clip = x_t noisy target. Device transfers are the caller's
        responsibility (do it once in the DataLoader / training loop).
        """
        if fluent_clip.dim() != 4:
            raise ValueError(f"fluent_clip expected [B, K, D, T], got {tuple(fluent_clip.shape)}")
        if disfluent_seq.dim() != 4:
            raise ValueError(f"disfluent_seq expected [B, K, D, T], got {tuple(disfluent_seq.shape)}")
        if previous_output is not None and previous_output.dim() != 4:
            raise ValueError(
                f"previous_output expected [B, K, D, T], got {tuple(previous_output.shape)}"
            )

        t_chunk = fluent_clip.shape[-1]

        # --- 1. encode streams ---
        t_emb = self._prepare_timestep_embedding(t)
        noisy_emb = self._encode_sequence(self.noisy_encoder, fluent_clip)
        disfluent_emb = self._encode_sequence(self.disfluent_encoder, disfluent_seq)

        history_emb = None
        if previous_output is not None and previous_output.shape[-1] > 0:
            history_emb = self._encode_sequence(self.history_encoder, previous_output)

        # --- 2. per-stream positional encoding (each starts at 0) ---
        noisy_emb = self._apply_pe(noisy_emb, self.pe_noisy)
        disfluent_emb = self._apply_pe(disfluent_emb, self.pe_cond)
        if history_emb is not None:
            history_emb = self._apply_pe(history_emb, self.pe_hist)

        # --- 3. classifier-free-guidance masking (per-stream, independent) ---
        if force_drop_disfluent:
            disfluent_emb = self.null_disfluent.expand_as(disfluent_emb)
        else:
            disfluent_emb = self._apply_stream_mask(
                disfluent_emb, self.null_disfluent, self.cond_mask_prob
            )

        if history_emb is not None:
            if force_drop_history:
                history_emb = self.null_history.expand_as(history_emb)
            else:
                history_emb = self._apply_stream_mask(
                    history_emb, self.null_history, self.cond_mask_prob
                )

        # --- 4. assemble condition stream ---
        cond_tokens = [t_emb, disfluent_emb]
        if history_emb is not None:
            cond_tokens.append(history_emb)

        cat_dim = 1 if self.batch_first else 0

        # --- 5. sequence modeling ---
        if self.arch == "trans_dec":
            memory = torch.cat(cond_tokens, dim=cat_dim)
            tgt = noisy_emb
            x_encoded = self.sequence_encoder(tgt=tgt, memory=memory)
            x_target = x_encoded

        else:
            # encoder / gru: noisy first, then conditions
            cond_stream = torch.cat(cond_tokens, dim=cat_dim)
            xseq = torch.cat([noisy_emb, cond_stream], dim=cat_dim)

            if self.arch == "trans_enc":
                x_encoded = self.sequence_encoder(xseq)
            elif self.arch == "gru":
                x_encoded, _ = self.sequence_encoder(xseq)
                x_encoded = self.gru_proj(x_encoded)          # 2D -> D
            else:
                raise ValueError("Unsupported architecture")

            # slice first T_chunk tokens (they correspond to the noisy path)
            if self.batch_first:
                x_target = x_encoded[:, :t_chunk, :]          # [B, T_chunk, D]
            else:
                x_target = x_encoded[:t_chunk, :, :]          # [T_chunk, B, D]

        # --- 6. residual fusion_proj ---
        residual = x_target
        x_target = self.post_norm(x_target)
        x_target = self.fusion_proj(x_target) + residual

        # --- 7. prepare for OutputProcessMLP (expects [T, B, D]) ---
        if self.batch_first:
            x_out = x_target.permute(1, 0, 2).contiguous()    # [T, B, D]
        else:
            x_out = x_target

        output = self.pose_projection(x_out)                  # [B, K, D, T]
        return output

    # -----------------------------------------------------------------
    # Inference-time classifier-free guidance
    # -----------------------------------------------------------------
    @torch.no_grad()
    def forward_with_cfg(
        self,
        fluent_clip: torch.Tensor,
        disfluent_seq: torch.Tensor,
        t: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        """
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        guidance_scale=1.0 -> plain conditional.
        """
        pred_cond = self.forward(
            fluent_clip=fluent_clip,
            disfluent_seq=disfluent_seq,
            t=t,
            previous_output=previous_output,
        )
        if guidance_scale == 1.0:
            return pred_cond

        pred_uncond = self.forward(
            fluent_clip=fluent_clip,
            disfluent_seq=disfluent_seq,
            t=t,
            previous_output=previous_output,
            force_drop_disfluent=True,
            force_drop_history=True,
        )
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    # -----------------------------------------------------------------
    # Diffusion interface
    # -----------------------------------------------------------------
    def interface(
        self,
        fluent_clip: torch.Tensor,
        t: torch.Tensor,
        y: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Training-time hook used by the diffusion process."""
        if "input_sequence" not in y:
            raise KeyError("y must contain 'input_sequence'")

        return self.forward(
            fluent_clip=fluent_clip,
            disfluent_seq=y["input_sequence"],
            t=t,
            previous_output=y.get("previous_output", None),
        )
