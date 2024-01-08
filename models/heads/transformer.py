import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomTransformerDecoder(nn.Module):
    def __init__(self, in_dim=768, d_model=768, nhead=8, num_layers=6, num_classes=15):
        super(CustomTransformerDecoder, self).__init__()
        self.num_classes = num_classes
        self.inp_proj = nn.Linear(in_dim, d_model)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        # Learnable embedding vectors (initialized here but could be set to a pre-trained value)
        #self.rot_vec = nn.Parameter(
        #    torch.rand((1, 1, d_model))
        #)  # One of the learned vectors
        #self.trans_vec = nn.Parameter(
        #    torch.rand((1, 1, d_model))
        #)  # The other learned vector
        if num_classes > 0:
            self.rot_vec = nn.Embedding(num_classes, d_model)
            self.trans_vec = nn.Embedding(num_classes, d_model)

        self.act = nn.ReLU()

        self.fc_rot1 = nn.Linear(d_model, 256)
        self.fc_rot2 = nn.Linear(256, 6)
        self.fc_trans = nn.Linear(d_model, 256)
        self.fc_cen = nn.Linear(256, 2)
        self.fc_z = nn.Linear(256, 1)

    def forward(self, inp):
        features = inp["roi_img"]
        assert features.ndim == 3, f"features.shape: {features.shape}"
        bs = features.shape[0]
        features= self.inp_proj(features)

        if self.num_classes > 0:
            rot_vec = self.rot_vec(inp["roi_cls"])
            trans_vec = self.trans_vec(inp["roi_cls"]) 
        else:
            rot_vec = self.rot_vec.repeat(bs, 1, 1)
            trans_vec = self.trans_vec.repeat(bs, 1, 1)
        tgt = torch.cat((rot_vec, trans_vec), dim=1)

        output = self.transformer_decoder(tgt,features)

        rot_feature = output[:, 0, :]
        rot_feature = self.fc_rot1(self.act(rot_feature))
        rot = self.fc_rot2(self.act(rot_feature))

        trans_feature = output[:, 1, :]
        trans_feature = self.fc_trans(self.act(trans_feature))
        centroid = self.fc_cen(self.act(trans_feature))
        z = self.fc_z(self.act(trans_feature))
        trans = torch.cat((centroid, z), dim=1)
        return rot, trans


model = CustomTransformerDecoder(in_dim=768, d_model=512)
tgt = torch.randn(10, 32, 768)
inp = {"roi_img":tgt, "roi_cls":torch.randint(0, 15, (10,1))}
output = model(inp)
print(
    "Number of parameters: ",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)
print(output.shape)


class PoseRegTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ex_feats = torch.randn(16, 768, 48)
