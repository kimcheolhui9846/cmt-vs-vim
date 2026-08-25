"""D칸 — CMT 골격에서 LMHSA만 양방향 Mamba로 바꾼 모델.

신규 아키텍처 제안이 아니라 요인 분리를 위한 통제된 재구현이다. VMamba·LocalMamba가
같은 방향을 이미 제시했고, 논문에서 그 관계를 인용으로 명시한다.

VMamba식 4방향 scan을 쓰지 않는 이유는 대칭성이다. B(Vim-Ti)가 양방향이므로 D를
4방향으로 만들면 B->D가 "계층 + conv + scan 방향수"를 한꺼번에 바꾸게 되어 A->C와
조작이 달라지고, 상호작용 항에 scan 방향 효과가 섞인다.
"""
import torch.nn as nn
from timm.models.layers import DropPath

from models.cmt_official import CMT, Mlp


class MambaBlock(nn.Module):
    """CMT Block에서 attention 자리만 바꾼 것. LPU와 IRFFN은 그대로 둔다.

    CMT Block의 forward와 한 줄만 다르다 — self.attn(self.norm1(x), H, W, relative_pos)
    자리에 self.mixer(self.norm1(x))가 들어간다. Mamba는 (B, L, D)를 받아 같은 모양을
    돌려주므로 시퀀스 축의 계약이 attention과 동일하다.
    """

    def __init__(self, dim, mlp_ratio=3.6, drop=0.0, drop_path=0.0, d_state=16,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba

        self.norm1 = norm_layer(dim)
        self.mixer = Mamba(d_model=dim, d_state=d_state, bimamba_type="v2",
                           if_divide_out=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        # LPU — CMT Block과 같은 depth-wise conv
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W, relative_pos=None):
        B, N, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class HierarchicalVim(CMT):
    """CMT를 그대로 짓고 블록만 갈아 끼운다.

    상속으로 짓는 이유는 stem·patch_embed·_fc·head를 C와 비트 단위로 같게 두기
    위해서다. 다시 구현하면 조용히 갈라진다.

    **알려진 비대칭**: 그 상속 때문에 D의 블록은 CMT 골격의 `nn.LayerNorm`을 그대로
    쓰고, 잔차는 fp16으로 흐른다. 반면 B(Vim-Ti)는 `rms_norm=True`,
    `fused_add_norm=True`, `residual_in_fp32=True`로 만들어진다(Vim은 fp16 안정성을
    위해 residual_in_fp32를 쓴다). 즉 A->C는 LayerNorm -> LayerNorm인데 B->D는
    RMSNorm+fp32 잔차 -> LayerNorm+fp16 잔차다. 두 조작이 norm 종류와 잔차 정밀도에서
    다르므로, 상호작용 항에 그 차이가 섞일 수 있다. "D는 C의 골격을 공유한다"는 설계
    자체에서 나오는 제약이라 코드로 없앨 수 없고(없애려면 D가 C의 골격을 벗어난다),
    설계 문서의 한계 목록에 폭 차이와 나란히 적어 둔다.
    """

    def __init__(self, img_size=64, num_classes=200, embed_dims=(52, 104, 208, 416),
                 stem_channel=16, depths=(2, 10, 2, 2), mlp_ratios=(3.6, 3.6, 3.6, 3.6),
                 d_state=16, drop_path_rate=0.1, dp=0.1):
        # CMT에서 dp는 stochastic depth가 아니라 head 앞 classifier dropout이다
        # (cmt_official.py:283의 self._drop). stochastic depth는 drop_path_rate이고
        # 기본값이 0이다. 두 인자를 섞으면 C와 D의 레시피가 조용히 갈라진다.
        super().__init__(
            img_size=img_size, num_classes=num_classes, embed_dims=list(embed_dims),
            stem_channel=stem_channel, num_heads=[1, 2, 4, 8], depths=list(depths),
            mlp_ratios=list(mlp_ratios), qkv_bias=True, qk_ratio=1,
            sr_ratios=[8, 4, 2, 1], drop_path_rate=drop_path_rate, dp=dp,
        )

        # CMT와 같은 stochastic depth 감쇠 규칙(cmt_official.py:238의 linspace)
        total = sum(depths)
        rates = (
            [drop_path_rate * i / (total - 1) for i in range(total)]
            if total > 1 else [0.0]
        )
        cursor = 0
        for name, dim, depth, ratio in zip(
            ("blocks_a", "blocks_b", "blocks_c", "blocks_d"),
            embed_dims, depths, mlp_ratios,
        ):
            blocks = nn.ModuleList([
                MambaBlock(dim=dim, mlp_ratio=ratio, drop_path=rates[cursor + i],
                           d_state=d_state)
                for i in range(depth)
            ])
            # super().__init__()의 self.apply(self._init_weights)는 CMT의 원래
            # attention 블록에 대해 이미 끝난 뒤이므로, 방금 새로 만든 MambaBlock의
            # Conv2d(proj의 LPU, mlp/IRFFN의 세 conv)는 그 초기화를 받지 못하고
            # PyTorch 기본값(kaiming_uniform_, mode='fan_in')으로 남는다. proj와
            # mlp 서브트리에만 좁혀서 CMT와 같은 kaiming_normal_(mode='fan_out')을
            # 다시 걸어준다 — self.mixer(Mamba)는 건드리지 않는다. Mamba는 dt_proj
            # 등 자체 초기화가 있고, 거기에 Conv2d용 kaiming_normal_을 걸면 그 초기화가
            # 깨진다.
            for block in blocks:
                block.proj.apply(self._init_weights)
                block.mlp.apply(self._init_weights)
            setattr(self, name, blocks)
            cursor += depth

        # attention이 사라졌으므로 상대 위치 파라미터도 사라져야 한다. 남겨 두면
        # 학습되지 않으면서 파라미터 예산만 먹어 "6.86M 대역 안"이라는 판정이
        # 거짓이 된다.
        for suffix in ("a", "b", "c", "d"):
            delattr(self, f"relative_pos_{suffix}")

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_norm1(self.stem_relu1(self.stem_conv1(x)))
        x = self.stem_norm2(self.stem_relu2(self.stem_conv2(x)))
        x = self.stem_norm3(self.stem_relu3(self.stem_conv3(x)))

        for embed, blocks in (
            (self.patch_embed_a, self.blocks_a), (self.patch_embed_b, self.blocks_b),
            (self.patch_embed_c, self.blocks_c), (self.patch_embed_d, self.blocks_d),
        ):
            if blocks is not self.blocks_a:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x, (H, W) = embed(x)
            for block in blocks:
                x = block(x, H, W)

        B, N, C = x.shape
        x = self._fc(x.permute(0, 2, 1).reshape(B, C, H, W))
        x = self._swish(self._bn(x))
        x = self._avg_pooling(x).flatten(start_dim=1)
        return self.pre_logits(self._drop(x))


def build_hvim(img_size=64, num_classes=200, embed_dims=(52, 104, 208, 416),
               depths=(2, 10, 2, 2), stem_channel=16,
               mlp_ratios=(3.6, 3.6, 3.6, 3.6), d_state=16,
               drop_path_rate=0.1, dp=0.1):
    return HierarchicalVim(
        img_size=img_size, num_classes=num_classes, embed_dims=embed_dims,
        stem_channel=stem_channel, depths=depths, mlp_ratios=mlp_ratios,
        d_state=d_state, drop_path_rate=drop_path_rate, dp=dp,
    )


def stage_grids(model) -> list[int]:
    """stage별 격자 한 변. C와 D가 같은 격자를 쓰는지 대조하는 데 쓴다."""
    return [
        int(embed.num_patches ** 0.5)
        for embed in (model.patch_embed_a, model.patch_embed_b,
                      model.patch_embed_c, model.patch_embed_d)
    ]
