import torch
import torch.nn as nn
from torch.nn import init


class TransformerPool(nn.Module):
    def __init__(self, dim=2048, part_num=6, head_num=8) -> None:
        super().__init__()

        self.part_num = part_num
        self.head_num = head_num
        self.scale = (dim // head_num) ** -0.5

        self.part_tokens = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(1, head_num, part_num, dim // head_num), mode='fan_out'))
        self.pos_embeding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(18 * 9, dim), mode='fan_out'))
        self.kv = nn.Linear(dim, dim * 2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        x = x + self.pos_embeding

        kv = self.kv(x).reshape(B, H * W, 2, self.head_num, C // self.head_num).permute(2, 0, 3, 1,
                                                                                        4)  # [2, B, head_num, HW, C//head_num]
        k, v = kv[0], kv[1]  # [B, head_num, H*W, C//head_num]

        sim = self.part_tokens @ k.transpose(-1, -2) * self.scale
        sim = sim.softmax(dim=-1)

        x = (sim @ v).transpose(1, 2).reshape(B, self.part_num, C)

        return x.view(B, -1)


class PLM(nn.Module):
    def __init__(self, dim=2048, part_num=6, H=24, W=12, dataset='sysu') -> None:
        super().__init__()

        self.proto_num = part_num
        self.learnable_proto = nn.Parameter(nn.init.kaiming_normal_(torch.empty(self.proto_num, dim)))
        self.pos_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(H * W, dim)))

        self.active = nn.Sigmoid()
        self.dataset = dataset

        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        x_pos = x + self.pos_embedding  # 位置编码

        sim = self.learnable_proto @ x_pos.transpose(-1, -2)
        sim = self.active(sim)

        x = sim @ x / H / W
        return x.view(B, -1), sim


if __name__ == '__main__':
    x1 = torch.randn(4, 2048, 18, 9)
    attn_pool = PLM(2048, 8, 18, 9)
    out, attn1 = attn_pool(x1)
    pass
