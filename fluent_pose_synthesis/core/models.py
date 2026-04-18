from typing import Optional, Dict
import torch
import torch.nn as nn

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess


class OutputProcessMLP(nn.Module):
    """
    Projects transformer output from latent space back to pose space.

    Input:  x: [T, B, latent_dim]
    Output: y: [B, K, D_feat, T]
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
        self.hidden_dim = hidden_dim

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

        x = self.mlp(x)                              # [T, B, input_feats]
        x = x.reshape(t, b, self.njoints, self.nfeats)  # [T, B, K, D_feat]
        x = x.permute(1, 2, 3, 0).contiguous()       # [B, K, D_feat, T]
        return x


class SignLanguagePoseDiffusion(nn.Module):
    """
    Diffusion denoiser for sign-language pose synthesis.

    Canonical internal sequence format: [T, B, D]

    External input format:
        fluent_clip     : [B, K, D_feat, T_chunk]   noisy sample x_t
        disfluent_seq   : [B, K, D_feat, T_cond]    conditioning sequence
        previous_output : [B, K, D_feat, T_hist]    optional fluent history

    Output: predicted clean chunk: [B, K, D_feat, T_chunk]

    Fixes vs original:
      - Separate PositionalEncoding instance for each stream (no double-PE on
        time embedding; noisy / cond / history get their own position indices
        from 0).
      - Per-stream classifier-free-guidance masks with learnable null
        embeddings (not zero-masking).
      - `forward_with_cfg` for guided sampling at inference.
      - Residual post-projection with LayerNorm.
      - Device transfers removed from forward (done by DataLoader / training loop).
      - Lower default dropout.
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
        cond_mask_prob: float = 0.1,       # ★ reasonable default for CFG
    ):
        super().__init__()

        if arch not in {"trans_enc", "gru", "trans_dec"}:
            raise ValueError("arch must be one of: trans_enc, gru, trans_dec")

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

        # ★ SEPARATE positional encoders per stream so each starts from pos 0.
        # This avoids treating "cond frame 0" as if it were "cond_start + offset".
        self.pe_noisy = PositionalEncoding(d_model=latent_dim, dropout=dropout)
        self.pe_cond = PositionalEncoding(d_model=latent_dim, dropout=dropout)
        self.pe_hist = PositionalEncoding(d_model=latent_dim, dropout=dropout)

        # TimestepEmbedder uses its own PE internally (inherited from CAMDM);
        # we do NOT apply another PE on top of its output.
        self.embed_timestep = TimestepEmbedder(
            latent_dim,
            self.pe_noisy,  # only used internally by the embedder
        )

        # Separate encoders for the semantic streams.
        self.noisy_encoder = MotionProcess(input_feats, latent_dim)
        self.disfluent_encoder = MotionProcess(input_feats, latent_dim)
        self.history_encoder = MotionProcess(input_feats, latent_dim)

        # ★ Learnable null embeddings for classifier-free guidance.
        # Shape [1, 1, D] so they broadcast over [T, B, D].
        self.null_disfluent = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.null_history = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.null_disfluent, std=0.02)
        nn.init.normal_(self.null_history, std=0.02)

        if arch == "trans_enc":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=False,
                norm_first=True,  # ★ pre-LN: more stable training
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

        elif arch == "trans_dec":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=False,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers,
            )

        elif arch == "gru":
            # ★ bidirectional GRU + projection back to latent_dim
            self.sequence_encoder = nn.GRU(
                input_size=latent_dim,
                hidden_size=latent_dim,
                num_layers=num_layers,
                batch_first=False,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.gru_proj = nn.Linear(latent_dim * 2, latent_dim)

        # ★ Post-projection with residual + LayerNorm
        self.post_norm = nn.LayerNorm(latent_dim)
        self.post_proj = nn.Sequential(
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
    def _encode_motion(self, encoder: nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
        """
        Expects x as [B, K, D_feat, T]. Returns [T, B, latent_dim].
        """
        if x.dim() != 4:
            raise ValueError(f"{name} expected [B, K, D_feat, T], got {tuple(x.shape)}")

        x_enc = encoder(x)

        if x_enc.dim() != 3:
            raise ValueError(
                f"{name} encoder returned unexpected shape {tuple(x_enc.shape)}; "
                f"expected [T, B, D]"
            )

        t, b, d = x_enc.shape
        if d != self.latent_dim:
            raise ValueError(
                f"{name} latent dim mismatch: got {d}, expected {self.latent_dim}"
            )

        return x_enc.contiguous()  # [T, B, D]

    def _encode_timestep(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Returns timestep token(s) as [1, B, D]. TimestepEmbedder already
        applies its own position encoding internally; we do not add more.
        """
        t_emb = self.embed_timestep(t)

        if t_emb.dim() == 2:
            t_emb = t_emb.unsqueeze(0)        # [B, D] -> [1, B, D]
        elif t_emb.dim() != 3:
            raise ValueError(f"Unexpected timestep embedding shape: {tuple(t_emb.shape)}")

        if t_emb.shape[1] != batch_size:
            raise ValueError(
                f"Timestep embedding batch mismatch: got {t_emb.shape[1]}, expected {batch_size}"
            )
        if t_emb.shape[2] != self.latent_dim:
            raise ValueError(
                f"Timestep embedding latent mismatch: got {t_emb.shape[2]}, expected {self.latent_dim}"
            )

        return t_emb.contiguous()  # [1, B, D]

    # -----------------------------------------------------------------
    # CFG masking (per-stream, learnable null embeddings)
    # -----------------------------------------------------------------
    def _apply_stream_mask(
        self,
        tokens: torch.Tensor,          # [T, B, D]
        null_embed: nn.Parameter,      # [1, 1, D]
        drop_prob: float,
    ) -> torch.Tensor:
        if drop_prob <= 0.0 or not self.training:
            return tokens
        _, batch_size, _ = tokens.shape
        drop = (torch.rand(batch_size, device=tokens.device) < drop_prob)  # [B]
        if not drop.any():
            return tokens
        drop = drop.view(1, batch_size, 1).to(tokens.dtype)                # [1, B, 1]
        null = null_embed.expand_as(tokens)                                # [T, B, D]
        return tokens * (1.0 - drop) + null * drop

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        fluent_clip: torch.Tensor,
        disfluent_seq: torch.Tensor,
        t: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
        force_drop_disfluent: bool = False,   # ★ for CFG sampling
        force_drop_history: bool = False,
    ) -> torch.Tensor:
        if fluent_clip.dim() != 4:
            raise ValueError(f"fluent_clip expected [B, K, D_feat, T], got {tuple(fluent_clip.shape)}")
        if disfluent_seq.dim() != 4:
            raise ValueError(f"disfluent_seq expected [B, K, D_feat, T], got {tuple(disfluent_seq.shape)}")
        if previous_output is not None and previous_output.dim() != 4:
            raise ValueError(
                f"previous_output expected [B, K, D_feat, T], got {tuple(previous_output.shape)}"
            )

        batch_size = fluent_clip.shape[0]
        t_chunk = fluent_clip.shape[-1]

        # --- 1. Encode each stream in [T, B, D] ---
        noisy_tokens = self._encode_motion(self.noisy_encoder, fluent_clip, "noisy")
        cond_tokens = self._encode_motion(self.disfluent_encoder, disfluent_seq, "disfluent")
        time_tokens = self._encode_timestep(t, batch_size)  # [1, B, D]

        history_tokens = None
        if previous_output is not None and previous_output.shape[-1] > 0:
            history_tokens = self._encode_motion(self.history_encoder, previous_output, "history")

        # --- 2. Per-stream independent positional encoding ---
        # Each stream starts at position 0 — the model learns stream identity
        # implicitly from the separate encoders, not from arbitrary offsets.
        noisy_tokens = self.pe_noisy(noisy_tokens)
        cond_tokens = self.pe_cond(cond_tokens)
        if history_tokens is not None:
            history_tokens = self.pe_hist(history_tokens)

        # --- 3. Classifier-free-guidance masking (per stream, independent) ---
        if force_drop_disfluent:
            cond_tokens = self.null_disfluent.expand_as(cond_tokens)
        else:
            cond_tokens = self._apply_stream_mask(
                cond_tokens, self.null_disfluent, self.cond_mask_prob
            )

        if history_tokens is not None:
            if force_drop_history:
                history_tokens = self.null_history.expand_as(history_tokens)
            else:
                history_tokens = self._apply_stream_mask(
                    history_tokens, self.null_history, self.cond_mask_prob
                )

        # --- 4. Build condition stream (time is a single summary token) ---
        cond_parts = [time_tokens, cond_tokens]
        if history_tokens is not None:
            cond_parts.append(history_tokens)
        cond_stream = torch.cat(cond_parts, dim=0)  # [T_cond_total, B, D]

        # --- 5. Sequence modeling ---
        if self.arch == "trans_enc":
            xseq = torch.cat([noisy_tokens, cond_stream], dim=0)  # [T_total, B, D]
            x_encoded = self.sequence_encoder(xseq)               # [T_total, B, D]
            x_target = x_encoded[:t_chunk]                        # [T_chunk, B, D]

        elif self.arch == "gru":
            xseq = torch.cat([noisy_tokens, cond_stream], dim=0)
            x_encoded, _ = self.sequence_encoder(xseq)            # [T_total, B, 2*D]
            x_encoded = self.gru_proj(x_encoded)                  # [T_total, B, D]
            x_target = x_encoded[:t_chunk]

        elif self.arch == "trans_dec":
            # No extra PE here — tokens are already positionally encoded above.
            tgt = noisy_tokens
            memory = cond_stream
            x_target = self.sequence_encoder(tgt=tgt, memory=memory)  # [T_chunk, B, D]

        else:
            raise ValueError(f"Unsupported arch: {self.arch}")

        # --- 6. Residual post-projection ---
        residual = x_target
        x_target = self.post_norm(x_target)
        x_target = self.post_proj(x_target) + residual

        # --- 7. Decode to pose space ---
        output = self.pose_projection(x_target)  # [B, K, D_feat, T_chunk]
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
        Classifier-free guidance at inference time.

        Runs the model twice — once with real conditioning, once with null
        conditioning — and extrapolates:
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        guidance_scale=1.0 reproduces the conditional output.
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
        """
        Used by the diffusion process during training.

        y:
            {
                "input_sequence": [B, K, D_feat, T_cond],
                "previous_output": [B, K, D_feat, T_hist]  (optional)
            }
        """
        if "input_sequence" not in y:
            raise KeyError("y must contain 'input_sequence'")

        return self.forward(
            fluent_clip=fluent_clip,
            disfluent_seq=y["input_sequence"],
            t=t,
            previous_output=y.get("previous_output", None),
        )
