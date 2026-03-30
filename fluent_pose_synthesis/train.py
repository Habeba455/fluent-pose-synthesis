from typing import Optional
import torch
import torch.nn as nn

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess


class OutputProcessMLP(nn.Module):
    """
    Input:
        output: [T, B, latent_dim]
    Output:
        [B, K, D, T]
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

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        # output: [T, B, latent_dim]
        nframes, bs, _ = output.shape

        output = self.mlp(output)  # [T, B, input_feats]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)  # [T, B, K, D]
        output = output.permute(1, 2, 3, 0).contiguous()  # [B, K, D, T]
        return output


class SignLanguagePoseDiffusion(nn.Module):
    """
    Corrected diffusion denoiser for sign language pose generation.

    Expected tensor shapes:
        fluent_clip (x_t):      [B, K, D, T_chunk]
        disfluent_seq:          [B, K, D, T_cond]
        previous_output:        [B, K, D, T_hist] or None
        t:                      [B]

    Important:
    - fluent_clip is the current noisy sample x_t and MUST be used.
    - disfluent_seq is the condition sequence.
    - previous_output is optional history condition.
    - We do NOT leak clean target as condition.
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
        ablation: Optional[str] = None,
        activation: str = "gelu",
        legacy: bool = False,
        arch: str = "trans_enc",
        cond_mask_prob: float = 0.0,
        device: Optional[torch.device] = None,
        batch_first: bool = True,
    ):
        super().__init__()

        self.input_feats = input_feats
        self.chunk_len = chunk_len
        self.keypoints = keypoints
        self.dims = dims
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.legacy = legacy
        self.arch = arch
        self.cond_mask_prob = cond_mask_prob
        self.device = device
        self.batch_first = batch_first

        self.sequence_pos_encoder = PositionalEncoding(
            d_model=latent_dim,
            dropout=dropout,
        )

        self.embed_timestep = TimestepEmbedder(
            latent_dim,
            self.sequence_pos_encoder,
        )

        # Separate encoders for clarity
        self.noisy_encoder = MotionProcess(input_feats, latent_dim)       # x_t
        self.disfluent_encoder = MotionProcess(input_feats, latent_dim)   # condition
        self.history_encoder = MotionProcess(input_feats, latent_dim)     # history

        if self.arch == "trans_enc":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
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
            )
            self.sequence_encoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers,
            )

        elif self.arch == "gru":
            self.sequence_encoder = nn.GRU(
                latent_dim,
                latent_dim,
                num_layers=num_layers,
                batch_first=True,
            )

        else:
            raise ValueError("arch must be one of: ['trans_enc', 'trans_dec', 'gru']")

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

        if self.device is not None:
            self.to(self.device)

    def _encode_sequence(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, K, D, T]
        Output:
            if batch_first=True  -> [B, T, latent_dim]
            else                 -> [T, B, latent_dim]

        Assumes MotionProcess returns [T, B, latent_dim].
        """
        x_raw = encoder(x)

        if x_raw.dim() != 3:
            raise ValueError(f"Unexpected encoder output shape: {x_raw.shape}")

        if self.batch_first:
            return x_raw.permute(1, 0, 2).contiguous()  # [B, T, D]
        return x_raw.contiguous()  # [T, B, D]

    def forward(
        self,
        fluent_clip: torch.Tensor,
        disfluent_seq: torch.Tensor,
        t: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        fluent_clip = x_t noisy target
        """
        if self.device is not None:
            fluent_clip = fluent_clip.to(self.device)
            disfluent_seq = disfluent_seq.to(self.device)
            t = t.to(self.device)
            if previous_output is not None:
                previous_output = previous_output.to(self.device)

        # noisy target chunk length
        T_chunk = fluent_clip.shape[-1]

        # timestep embedding
        t_emb_raw = self.embed_timestep(t)
        # expected either [1, B, D] or [B, D]
        if t_emb_raw.dim() == 2:
            t_emb_raw = t_emb_raw.unsqueeze(0)  # [1, B, D]

        if self.batch_first:
            t_emb = t_emb_raw.permute(1, 0, 2).contiguous()  # [B, 1, D]
        else:
            t_emb = t_emb_raw.contiguous()  # [1, B, D]

        # === IMPORTANT FIX ===
        # Encode x_t itself
        noisy_emb = self._encode_sequence(self.noisy_encoder, fluent_clip)         # [B,T,D] or [T,B,D]
        disfluent_emb = self._encode_sequence(self.disfluent_encoder, disfluent_seq)

        tokens = [t_emb, disfluent_emb]

        if previous_output is not None and previous_output.shape[-1] > 0:
            prev_emb = self._encode_sequence(self.history_encoder, previous_output)
            tokens.append(prev_emb)

        if self.arch == "trans_dec":
            # decoder mode:
            #   tgt = noisy tokens
            #   memory = condition tokens
            if self.batch_first:
                memory = torch.cat(tokens, dim=1)  # [B, T_cond_total, D]
                memory = self.sequence_pos_encoder(memory)

                tgt = self.sequence_pos_encoder(noisy_emb)  # [B, T_chunk, D]
                x_encoded = self.sequence_encoder(tgt=tgt, memory=memory)  # [B, T_chunk, D]

                x_target = self.fusion_proj(x_encoded)
                x_out = x_target.permute(1, 0, 2).contiguous()  # [T, B, D]

            else:
                memory = torch.cat(tokens, dim=0)  # [T_cond_total, B, D]
                memory = self.sequence_pos_encoder(memory)

                tgt = self.sequence_pos_encoder(noisy_emb)  # [T_chunk, B, D]
                x_encoded = self.sequence_encoder(tgt=tgt, memory=memory)

                x_target = self.fusion_proj(x_encoded)
                x_out = x_target

        else:
            # encoder / gru mode:
            # concatenate noisy tokens with condition tokens
            if self.batch_first:
                cond_tokens = torch.cat(tokens, dim=1)              # [B, T_cond_total, D]
                xseq = torch.cat([noisy_emb, cond_tokens], dim=1)   # [B, T_total, D]
                xseq = self.sequence_pos_encoder(xseq)

                if self.arch == "trans_enc":
                    x_encoded = self.sequence_encoder(xseq)
                elif self.arch == "gru":
                    x_encoded, _ = self.sequence_encoder(xseq)
                else:
                    raise ValueError("Unsupported architecture")

                # first T_chunk tokens correspond to x_t path
                x_target = x_encoded[:, :T_chunk, :]                # [B, T_chunk, D]
                x_target = self.fusion_proj(x_target)
                x_out = x_target.permute(1, 0, 2).contiguous()      # [T, B, D]

            else:
                cond_tokens = torch.cat(tokens, dim=0)              # [T_cond_total, B, D]
                xseq = torch.cat([noisy_emb, cond_tokens], dim=0)   # [T_total, B, D]
                xseq = self.sequence_pos_encoder(xseq)

                if self.arch == "trans_enc":
                    x_encoded = self.sequence_encoder(xseq)
                elif self.arch == "gru":
                    x_encoded, _ = self.sequence_encoder(xseq)
                else:
                    raise ValueError("Unsupported architecture")

                x_target = x_encoded[:T_chunk, :, :]                # [T_chunk, B, D]
                x_target = self.fusion_proj(x_target)
                x_out = x_target

        output = self.pose_projection(x_out)  # [B, K, D, T]
        return output

    def interface(
        self,
        fluent_clip: torch.Tensor,
        t: torch.Tensor,
        y: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        fluent_clip هنا هو x_t من diffusion process
        """
        batch_size = fluent_clip.size(0)

        disfluent_seq = y["input_sequence"]
        previous_output = y.get("previous_output", None)

        # CFG masking only on conditioning paths
        if self.cond_mask_prob > 0:
            keep = (
                torch.rand(batch_size, device=disfluent_seq.device)
                < (1 - self.cond_mask_prob)
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
