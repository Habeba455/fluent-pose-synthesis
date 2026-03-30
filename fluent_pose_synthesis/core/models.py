from typing import Optional
import torch
import torch.nn as nn

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess


class OutputProcessMLP(nn.Module):
    """
    Output process for the Sign Language Pose Diffusion model.
    Expects input shape: [T, B, D]
    Returns output shape: [B, K, D_feat, T]
    """

    def __init__(self, input_feats, latent_dim, njoints, nfeats, hidden_dim=512):
        super().__init__()

        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.input_feats),
        )

    def forward(self, output):
        # output: [T, B, D]
        nframes, bs, d = output.shape

        output = self.mlp(output)  # [T, B, input_feats]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)  # [T, B, K, D_feat]
        output = output.permute(1, 2, 3, 0).contiguous()  # [B, K, D_feat, T]

        return output


class SignLanguagePoseDiffusion(nn.Module):
    """
    Sign Language Pose Diffusion model.

    Correct behavior:
    - fluent_clip is the current noisy sample x_t and MUST be used by the denoiser.
    - disfluent_seq is the conditioning sequence (updated_clean).
    - previous_output is optional fluent history.
    - timestep embedding is used as an additional token.
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
        cond_mask_prob: float = 0,
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

        # positional encoding
        self.sequence_pos_encoder = PositionalEncoding(
            d_model=latent_dim,
            dropout=dropout,
        )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim,
            self.sequence_pos_encoder,
        )

        # encoders
        # noisy encoder for x_t
        self.noisy_encoder = MotionProcess(input_feats, latent_dim)

        # condition encoders
        self.disfluent_encoder = MotionProcess(input_feats, latent_dim)
        self.history_encoder = MotionProcess(input_feats, latent_dim)

        # optional fusion after encoder
        self.fusion_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # sequence encoder
        if self.arch == "trans_enc":
            print(f"Initializing Transformer Encoder (batch_first={self.batch_first})")

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=self.batch_first,
            )

            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

        elif self.arch == "trans_dec":
            print(f"Initializing Transformer Decoder (batch_first={self.batch_first})")

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=self.batch_first,
            )

            self.sequence_encoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers,
            )

        elif self.arch == "gru":
            print("Initializing GRU Encoder (batch_first=True)")

            self.sequence_encoder = nn.GRU(
                latent_dim,
                latent_dim,
                num_layers=num_layers,
                batch_first=True,
            )

        else:
            raise ValueError("Please choose correct architecture [trans_enc, trans_dec, gru]")

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
        x shape: [B, K, D_feat, T]
        expected encoder output: typically [T, B, latent_dim]
        return shape: [B, T, latent_dim] if batch_first else [T, B, latent_dim]
        """
        x_raw = encoder(x)

        # original code pattern assumed MotionProcess returns [T, B, D]
        # convert to batch_first if needed
        if x_raw.dim() != 3:
            raise ValueError(f"Unexpected encoder output shape: {x_raw.shape}")

        if self.batch_first:
            return x_raw.permute(1, 0, 2).contiguous()  # [B, T, D]
        else:
            return x_raw.contiguous()  # [T, B, D]

    def forward(
        self,
        fluent_clip: torch.Tensor,
        disfluent_seq: torch.Tensor,
        t: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            fluent_clip: current noisy sample x_t, shape [B, K, D_feat, T_chunk]
            disfluent_seq: conditioning sequence, shape [B, K, D_feat, T_cond]
            t: diffusion timestep, shape [B]
            previous_output: optional fluent history, shape [B, K, D_feat, T_hist]

        Returns:
            output: shape [B, K, D_feat, T_chunk]
        """
        if self.device is not None:
            fluent_clip = fluent_clip.to(self.device)
            disfluent_seq = disfluent_seq.to(self.device)
            t = t.to(self.device)
            if previous_output is not None:
                previous_output = previous_output.to(self.device)

        # IMPORTANT:
        # Time is the LAST dimension here.
        T_chunk = fluent_clip.shape[-1]

        # timestep embedding
        _t_emb_raw = self.embed_timestep(t)
        # expected: [1, B, D] or [B, D]
        if _t_emb_raw.dim() == 2:
            _t_emb_raw = _t_emb_raw.unsqueeze(0)  # [1, B, D]

        if self.batch_first:
            t_emb = _t_emb_raw.permute(1, 0, 2).contiguous()  # [B, 1, D]
        else:
            t_emb = _t_emb_raw.contiguous()  # [1, B, D]

        # encode noisy current sample x_t
        noisy_emb = self._encode_sequence(self.noisy_encoder, fluent_clip)

        # encode disfluent condition
        disfluent_emb = self._encode_sequence(self.disfluent_encoder, disfluent_seq)

        # conditioning tokens
        if self.batch_first:
            embeddings_to_concat = [t_emb, disfluent_emb]
        else:
            embeddings_to_concat = [t_emb, disfluent_emb]

        # optional history
        if previous_output is not None and previous_output.shape[-1] > 0:
            prev_out_emb = self._encode_sequence(self.history_encoder, previous_output)
            embeddings_to_concat.append(prev_out_emb)

        # concatenate condition stream
        if self.batch_first:
            cond_tokens = torch.cat(embeddings_to_concat, dim=1)  # [B, T_cond_total, D]
            # final sequence = noisy tokens + condition tokens
            xseq = torch.cat([noisy_emb, cond_tokens], dim=1)     # [B, T_chunk + T_cond_total, D]
            xseq = self.sequence_pos_encoder(xseq)
        else:
            cond_tokens = torch.cat(embeddings_to_concat, dim=0)  # [T_cond_total, B, D]
            xseq = torch.cat([noisy_emb, cond_tokens], dim=0)     # [T_chunk + T_cond_total, B, D]
            xseq = self.sequence_pos_encoder(xseq)

        # run sequence encoder
        if self.arch == "trans_enc":
            x_encoded = self.sequence_encoder(xseq)

        elif self.arch == "gru":
            x_encoded, _ = self.sequence_encoder(xseq)

        elif self.arch == "trans_dec":
            # in decoder mode, use condition as memory and noisy tokens as target
            if self.batch_first:
                memory = cond_tokens
                tgt = noisy_emb
            else:
                memory = cond_tokens
                tgt = noisy_emb
            x_encoded = self.sequence_encoder(tgt=tgt, memory=memory)

        else:
            raise ValueError("Unsupported architecture")

        # We only want the output tokens corresponding to the noisy target chunk
        if self.batch_first:
            # x_encoded: [B, T_total, D]
            x_target = x_encoded[:, :T_chunk, :]   # first T_chunk tokens correspond to noisy input
            x_target = self.fusion_proj(x_target)
            x_out = x_target.permute(1, 0, 2).contiguous()  # [T, B, D]
        else:
            # x_encoded: [T_total, B, D]
            x_target = x_encoded[:T_chunk, :, :]
            x_target = self.fusion_proj(x_target)
            x_out = x_target  # already [T, B, D]

        output = self.pose_projection(x_out)  # [B, K, D_feat, T]
        return output

    def interface(
        self,
        fluent_clip: torch.Tensor,
        t: torch.Tensor,
        y: dict[str, torch.Tensor],
    ):
        """
        fluent_clip here is x_t from the diffusion process, NOT clean target.
        """
        batch_size = fluent_clip.size(0)

        disfluent_seq = y["input_sequence"]
        previous_output = y.get("previous_output", None)

        # classifier-free guidance masking on conditions only
        if self.cond_mask_prob > 0:
            keep_batch_idx = torch.rand(
                batch_size,
                device=disfluent_seq.device,
            ) < (1 - self.cond_mask_prob)

            keep_mask = keep_batch_idx.view((batch_size, 1, 1, 1)).float()

            disfluent_seq = disfluent_seq * keep_mask

            if previous_output is not None:
                previous_output = previous_output * keep_mask

        return self.forward(
            fluent_clip=fluent_clip,
            disfluent_seq=disfluent_seq,
            t=t,
            previous_output=previous_output,
        )
