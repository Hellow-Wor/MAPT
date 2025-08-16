import torch
from torch import nn
from models.unit import Embedding, AttentionLayer, Encoder, EncoderLayer, Decoder, DecoderLayer, Embedding_inverted
from models.atten import ProbAttention, FullAttention

class BatteryTransformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, session, label_len, pred_len, factor=5,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu', inverse=True,
                 output_attention=False, distil=True, device=torch.device('cuda:0')
                 ):
        super(BatteryTransformer, self).__init__()
        self.d_model = d_model
        self.inverse = inverse
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.features_enc = enc_in // 2
        self.features_dec = dec_in
        self.session = session
        self.enc_embedding = Embedding_inverted(enc_in//2, d_model, dropout=dropout)
        self.dec_embedding = Embedding(dec_in, d_model, dropout=dropout)
        self.dec_out_embedding = Embedding(dec_in*2, d_model, dropout=dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.encoder = Encoder(
            attn_layers = [
                EncoderLayer(
                    AttentionLayer(Attn(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            layers = [
                DecoderLayer(
                    AttentionLayer(Attn(mask_flag=True, factor=factor, attention_dropout=dropout, output_attention=False), d_model, n_heads, mix=True),
                    AttentionLayer(FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention=False), d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection1 = nn.Linear(d_model, c_out, bias=True)
        self.projection2 = nn.Linear(d_model, session * 2, bias=True)
        nn.init.orthogonal_(self.projection2.weight)

    def forward(self, x_enc, x_dec):
        means = torch.zeros_like(x_dec)
        stdev = torch.ones_like(x_dec)
        if self.inverse:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            means = x_dec.mean(1, keepdim=True).detach()
            x_dec = x_dec - means
            stdev = torch.sqrt(torch.var(x_dec, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_dec /= stdev

        x_enc1_embedding = self.enc_embedding(x_enc[:,:,:self.features_enc])  # voltage
        x_enc2_embedding = self.enc_embedding(x_enc[:,:,self.features_enc:])  # diff-voltage

        x_dec_embedding = self.dec_embedding(x_dec)
        enc1_out, attns = self.encoder(x_enc1_embedding)
        enc2_out, attns = self.encoder(x_enc2_embedding)
        dec1_out = self.decoder(x_dec_embedding, enc1_out)
        dec2_out = self.decoder(x_dec_embedding, enc2_out)

        dec1_out = self.projection1(dec1_out)
        dec2_out = self.projection1(dec2_out)
        dec_out = torch.cat((dec1_out, dec2_out), dim=2)
        x_dec_out_embedding = self.dec_out_embedding(dec_out)

        dec_out = self.projection2(x_dec_out_embedding)
        dec_out = dec_out.reshape(dec_out.shape[0], self.label_len+self.pred_len, self.features_dec, -1)
        dec_out = dec_out * (stdev.unsqueeze(-1).repeat(1, self.label_len+self.pred_len, 1, self.session))
        dec_out = dec_out + (means.unsqueeze(-1).repeat(1, self.label_len+self.pred_len, 1, self.session))

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :, :], attns
        else:
            return dec_out[:, -self.pred_len:, :, :]
