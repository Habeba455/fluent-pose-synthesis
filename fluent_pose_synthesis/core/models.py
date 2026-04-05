from typing import Optional, Dict
import torch
import torch.nn as nn

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess


class OutputProcessMLP(nn.Module):
    """
    Projects transformer output from latent space back to pose space.

    Input:
        x: [T, B, latent_dim]

    Output:
        y: [B, K, D_feat, T]
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

        x = self.mlp(x)  # [T, B, input_feats]
        x = x.reshape(t, b, self.njoints, self.nfeats)  # [T, B, K, D_feat]
        x = x.permute(1, 2, 3, 0).contiguous()  # [B, K, D_feat, T]
        return x


class SignLanguagePoseDiffusion(nn.Module):
    """
    Safer diffusion denoiser for sign-language pose synthesis.

    Canonical internal sequence format:
        [T, B, D]

    External input format:
        fluent_clip     : [B, K, D_feat, T_chunk]   -> noisy sample x_t
        disfluent_seq   : [B, K, D_feat, T_cond]    -> conditioning sequence
        previous_output : [B, K, D_feat, T_hist]    -> optional fluent history

    Output:
        predicted clean chunk: [B, K, D_feat, T_chunk]
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
        dropout: float = 0.2,
        activation: str = "gelu",
        arch: str = "trans_enc",
        cond_mask_prob: float = 0.0,
        device: Optional[torch.device] = None,
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
        self.device = device

        # Positional encoding is used in canonical [T, B, D] format.
        self.sequence_pos_encoder = PositionalEncoding(
            d_model=latent_dim,
            dropout=dropout,
        )
        self.embed_timestep = TimestepEmbedder(
            latent_dim,
            self.sequence_pos_encoder,
        )

        # Separate encoders for separate semantic streams.
        self.noisy_encoder = MotionProcess(input_feats, latent_dim)
        self.disfluent_encoder = MotionProcess(input_feats, latent_dim)
        self.history_encoder = MotionProcess(input_feats, latent_dim)

        if arch == "trans_enc":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=False,
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
            )
            self.sequence_encoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers,
            )

        elif arch == "gru":
            self.sequence_encoder = nn.GRU(
                input_size=latent_dim,
                hidden_size=latent_dim,
                num_layers=num_layers,
                batch_first=False,
            )

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

        if self.device is not None:
            self.to(self.device)

    def _move_to_device(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if self.device is None:
            return x
        return x.to(self.device)

    def _encode_motion(self, encoder: nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
        """
        Expects x as [B, K, D_feat, T].
        Returns encoded tokens as [T, B, latent_dim].
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
        Returns timestep token(s) as [1, B, D].
        """
        t_emb = self.embed_timestep(t)

        if t_emb.dim() == 2:
            # [B, D] -> [1, B, D]
            t_emb = t_emb.unsqueeze(0)
        elif t_emb.dim() == 3:
            # Usually already [1, B, D]
            pass
        else:
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

    def forward(
        self,
        fluent_clip: torch.Tensor,
        disfluent_seq: torch.Tensor,
        t: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            fluent_clip: noisy sample x_t, [B, K, D_feat, T_chunk]
            disfluent_seq: conditioning sequence, [B, K, D_feat, T_cond]
            t: diffusion timestep, [B]
            previous_output: optional history, [B, K, D_feat, T_hist]

        Returns:
            [B, K, D_feat, T_chunk]
        """
        fluent_clip = self._move_to_device(fluent_clip)
        disfluent_seq = self._move_to_device(disfluent_seq)
        t = self._move_to_device(t)
        previous_output = self._move_to_device(previous_output)

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

        # Encode each stream in canonical [T, B, D]
        noisy_tokens = self._encode_motion(self.noisy_encoder, fluent_clip, "noisy")
        cond_tokens = self._encode_motion(self.disfluent_encoder, disfluent_seq, "disfluent")
        time_tokens = self._encode_timestep(t, batch_size)

        history_tokens = None
        if previous_output is not None and previous_output.shape[-1] > 0:
            history_tokens = self._encode_motion(self.history_encoder, previous_output, "history")

        # Build condition stream
        cond_parts = [time_tokens, cond_tokens]
        if history_tokens is not None:
            cond_parts.append(history_tokens)
        cond_stream = torch.cat(cond_parts, dim=0)  # [T_cond_total, B, D]

        # Sequence modeling
        if self.arch == "trans_enc":
            # Put noisy tokens first so we can always read back the first T_chunk tokens.
            xseq = torch.cat([noisy_tokens, cond_stream], dim=0)  # [T_total, B, D]
            xseq = self.sequence_pos_encoder(xseq)
            x_encoded = self.sequence_encoder(xseq)  # [T_total, B, D]
            x_target = x_encoded[:t_chunk]  # [T_chunk, B, D]

        elif self.arch == "gru":
            xseq = torch.cat([noisy_tokens, cond_stream], dim=0)
            xseq = self.sequence_pos_encoder(xseq)
            x_encoded, _ = self.sequence_encoder(xseq)  # [T_total, B, D]
            x_target = x_encoded[:t_chunk]

        elif self.arch == "trans_dec":
            # Decoder target = noisy tokens, memory = conditions
            tgt = self.sequence_pos_encoder(noisy_tokens)
            memory = self.sequence_pos_encoder(cond_stream)
            x_target = self.sequence_encoder(tgt=tgt, memory=memory)  # [T_chunk, B, D]

        else:
            raise ValueError(f"Unsupported arch: {self.arch}")

        x_target = self.post_proj(x_target)  # [T_chunk, B, D]
        output = self.pose_projection(x_target)  # [B, K, D_feat, T_chunk]
        return output

    def interface(
        self,
        fluent_clip: torch.Tensor,
        t: torch.Tensor,
        y: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Interface used by the diffusion process.

        fluent_clip:
            noisy sample x_t, [B, K, D_feat, T_chunk]

        y:
            {
                "input_sequence": [B, K, D_feat, T_cond],
                "previous_output": [B, K, D_feat, T_hist]  # optional
            }
        """
        if "input_sequence" not in y:
            raise KeyError("y must contain 'input_sequence'")

        disfluent_seq = y["input_sequence"]
        previous_output = y.get("previous_output", None)

        # Classifier-free guidance masking on conditions only.
        if self.cond_mask_prob > 0:
            batch_size = fluent_clip.size(0)
            keep = (
                torch.rand(batch_size, device=disfluent_seq.device)
                < (1.0 - self.cond_mask_prob)
            ).float().view(batch_size, 1, 1, 1)

            disfluent_seq = disfluent_seq * keep
            if previous_output is not None:
                previous_output = previous_output * keep

        return self.forward(
            fluent_clip=fluent_clip,
            disfluent_seq=disfluent_seq,
            t=t,
            previous_output=previous_output,
        )
