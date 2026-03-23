from typing import Optional
import torch
import torch.nn as nn

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess


class OutputProcessMLP(nn.Module):
    """
    Output process for the Sign Language Pose Diffusion model.
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
            nn.Linear(self.hidden_dim // 2, self.input_feats)
        )

    def forward(self, output):
        nframes, bs, d = output.shape

        output = self.mlp(output)
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)

        return output


class SignLanguagePoseDiffusion(nn.Module):
    """
    Sign Language Pose Diffusion model.

    Important fix:
    - DO NOT condition on fluent_clip (target).
    - Only condition on:
        1) timestep embedding
        2) disfluent input sequence
        3) previous_output (optional history)
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
            dropout=dropout
        )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim,
            self.sequence_pos_encoder
        )

        # encoders
        self.fluent_encoder = MotionProcess(input_feats, latent_dim)
        self.disfluent_encoder = MotionProcess(input_feats, latent_dim)

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
                num_layers=num_layers
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
                num_layers=num_layers
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
            input_feats,
            latent_dim,
            keypoints,
            dims,
            hidden_dim=1024,
        )

        if self.device is not None:
            self.to(self.device)

    def forward(
        self,
        fluent_clip: torch.Tensor,
        disfluent_seq: torch.Tensor,
        t: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
    ):
        """
        fluent_clip:
            target/noisy target sequence used by diffusion training objective.
            It should NOT be used as a conditioning signal.
        """
        fluent_clip = fluent_clip.to(self.device)
        disfluent_seq = disfluent_seq.to(self.device)
        t = t.to(self.device)

        if previous_output is not None:
            previous_output = previous_output.to(self.device)

        T_chunk = fluent_clip.shape[-1]

        # timestep embedding
        _t_emb_raw = self.embed_timestep(t)
        if _t_emb_raw.dim() == 2:
            _t_emb_raw = _t_emb_raw.unsqueeze(1)

        # shape => [B, 1, D]
        t_emb = _t_emb_raw.permute(1, 0, 2).contiguous()

        # disfluent encoding
        _disfluent_emb_raw = self.disfluent_encoder(disfluent_seq)
        disfluent_emb = _disfluent_emb_raw.permute(1, 0, 2).contiguous()

        # conditioning tokens
        embeddings_to_concat = [t_emb, disfluent_emb]

        # previous fluent history encoding (optional)
        if previous_output is not None and previous_output.shape[-1] > 0:
            _prev_out_emb_raw = self.fluent_encoder(previous_output)
            prev_out_emb = _prev_out_emb_raw.permute(1, 0, 2).contiguous()
            embeddings_to_concat.append(prev_out_emb)

        # IMPORTANT:
        # Do NOT append fluent_clip encoding here.
        # The old version leaked target information into the conditioning stream.

        xseq = torch.cat(embeddings_to_concat, dim=1)

        # positional encoding
        if self.batch_first:
            # xseq: [B, T_total, D]
            xseq_encoded = self.sequence_pos_encoder(xseq)
            xseq = xseq_encoded
        else:
            # convert to [T_total, B, D]
            xseq = xseq.permute(1, 0, 2).contiguous()
            xseq = self.sequence_pos_encoder(xseq)

        # encoder
        if self.arch == "trans_enc":
            x_encoded = self.sequence_encoder(xseq)

        elif self.arch == "gru":
            x_encoded, _ = self.sequence_encoder(xseq)

        elif self.arch == "trans_dec":
            memory = xseq
            tgt = xseq
            x_encoded = self.sequence_encoder(tgt=tgt, memory=memory)

        else:
            raise ValueError("Unsupported architecture")

        # We need T_chunk output tokens.
        # Since conditioning length may be < T_chunk, expand/repeat as needed.
        if self.batch_first:
            current_len = x_encoded.shape[1]

            if current_len < T_chunk:
                repeat_count = (T_chunk + current_len - 1) // current_len
                x_encoded = x_encoded.repeat(1, repeat_count, 1)

            x_out = x_encoded[:, -T_chunk:, :]
            x_out = x_out.permute(1, 0, 2).contiguous()

        else:
            current_len = x_encoded.shape[0]

            if current_len < T_chunk:
                repeat_count = (T_chunk + current_len - 1) // current_len
                x_encoded = x_encoded.repeat(repeat_count, 1, 1)

            x_out = x_encoded[-T_chunk:, :, :]

        output = self.pose_projection(x_out)
        return output

    def interface(
        self,
        fluent_clip: torch.Tensor,
        t: torch.Tensor,
        y: dict[str, torch.Tensor],
    ):
        batch_size = fluent_clip.size(0)

        disfluent_seq = y["input_sequence"]
        previous_output = y.get("previous_output", None)

        # CFG masking on condition only
        if self.cond_mask_prob > 0:
            keep_batch_idx = torch.rand(
                batch_size,
                device=disfluent_seq.device
            ) < (1 - self.cond_mask_prob)

            disfluent_seq = disfluent_seq * keep_batch_idx.view(
                (batch_size, 1, 1, 1)
            )

            if previous_output is not None:
                previous_output = previous_output * keep_batch_idx.view(
                    (batch_size, 1, 1, 1)
                )

        return self.forward(
            fluent_clip=fluent_clip,
            disfluent_seq=disfluent_seq,
            t=t,
            previous_output=previous_output,
        )
