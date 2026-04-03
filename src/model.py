import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, num_heads=8, hidden_dim=1024, dropout=0.15):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels

        # Linear projections cho Query, Key, Value
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)

        # Feed Forward Network (FFN) như bài báo đề xuất
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 1. Patch Tokenization (Đưa về chuỗi 1D: B, N, C)
        x_flat = x.view(B, C, N).permute(0, 2, 1)

        # 2. Multi-Head Self Attention
        # Tính Q, K, V
        qkv = self.qkv(self.norm1(x_flat)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Kích thước mỗi cái: (B, Heads, N, Head_dim)

        # Tính Attention Score: (Q * K^T) / sqrt(d_k)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / (k.shape[-1] ** 0.5))
        attn_weights = F.softmax(attn_scores, dim=-1)  # Kích thước: (B, Heads, N, N)

        # Nhân với Value
        x_attn = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)

        # 3. Residual Connection & FFN
        x_flat = x_flat + x_attn
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        # 4. Reshape lại thành ảnh (B, C, H, W)
        out = x_flat.permute(0, 2, 1).view(B, C, H, W)

        return out, attn_weights


# --- CÁC KHỐI CƠ BẢN (BUILDING BLOCKS) ---

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpFlexible(nn.Module):
    """
    Up block linh hoạt cho Dense Connections.
    Cho phép chỉ định rõ số kênh đầu vào (in_ch) và số kênh skip (skip_ch).
    """

    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()

        # Upsample giữ nguyên số kênh
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

        # Conv nhận vào tổng số kênh của (Input đã upsample + Skip connection)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample input từ dưới lên

        # Xử lý padding nếu kích thước không khớp
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Nối với Skip Connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- MODULE SELF-AWARE ATTENTION (SAA) ---
# (Giữ nguyên không đổi)
class SelfAwareAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(SelfAwareAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = in_channels // num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.reduced_channels = in_channels // 8
        self.gsa_conv_m = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.gsa_conv_n = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.gsa_conv_w = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gsa_conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma_tsa = nn.Parameter(torch.zeros(1))
        self.gamma_gsa = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, self.num_heads, self.d_k, -1)
        proj_key = self.key_conv(x).view(batch_size, self.num_heads, self.d_k, -1)
        proj_value = self.value_conv(x).view(batch_size, self.num_heads, self.d_k, -1)
        energy = torch.matmul(proj_query.permute(0, 1, 3, 2), proj_key)
        attention = F.softmax(energy / (self.d_k ** 0.5), dim=-1)
        tsa_out = torch.matmul(attention, proj_value.permute(0, 1, 3, 2))
        tsa_out = tsa_out.permute(0, 1, 3, 2).contiguous().view(batch_size, channels, height, width)
        m = self.gsa_conv_m(x).view(batch_size, self.reduced_channels, -1).permute(0, 2, 1)
        n = self.gsa_conv_n(x).view(batch_size, self.reduced_channels, -1)
        position_attention = F.softmax(torch.matmul(m, n), dim=-1)
        w = self.gsa_conv_w(x).view(batch_size, channels, -1)
        gsa_out = torch.matmul(w, position_attention.permute(0, 2, 1))
        gsa_out = gsa_out.view(batch_size, channels, height, width)
        gsa_out = self.gsa_conv_out(gsa_out)
        out = (self.gamma_tsa * tsa_out) + (self.gamma_gsa * gsa_out) + x
        return out


# --- KIẾN TRÚC CHÍNH TRANSATTUNET (DENSE VERSION) ---


# --- KIẾN TRÚC CHÍNH TRANSATTUNET (DENSE VERSION + TRANSFORMER BOTTLENECK) ---

class TransAttUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=3, bilinear=True):
        super(TransAttUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 1. Encoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )

        # 2. Bridge (Transformer + SAA)
        self.transformer_bottleneck = TransformerBottleneck(in_channels=512)
        self.saa_bridge = SelfAwareAttention(in_channels=512)

        # ==========================================================
        # 3A. NHÁNH CHÍNH (SEGMENTATION) - DENSE CONNECTIONS
        # ==========================================================
        self.up1 = UpFlexible(in_ch=512, skip_ch=256, out_ch=256, bilinear=bilinear)
        self.up2 = UpFlexible(in_ch=768, skip_ch=128, out_ch=128, bilinear=bilinear)
        self.up3 = UpFlexible(in_ch=384, skip_ch=64, out_ch=64, bilinear=bilinear)
        self.up4 = UpFlexible(in_ch=192, skip_ch=32, out_ch=32, bilinear=bilinear)
        self.outc = nn.Conv2d(96, n_classes, kernel_size=1)

        # Deep Supervision
        self.ds_out1 = nn.Conv2d(128, n_classes, kernel_size=1)
        self.ds_out2 = nn.Conv2d(64, n_classes, kernel_size=1)

        # ==========================================================
        # 3B. NHÁNH PHỤ (RECONSTRUCTION) - HỌC GIẢI PHẪU PHỔI
        # Dùng lại UpFlexible nhưng theo phong cách U-Net cơ bản (không Dense)
        # ==========================================================
        self.recon_up1 = UpFlexible(in_ch=512, skip_ch=256, out_ch=256, bilinear=bilinear)
        self.recon_up2 = UpFlexible(in_ch=256, skip_ch=128, out_ch=128, bilinear=bilinear)
        self.recon_up3 = UpFlexible(in_ch=128, skip_ch=64, out_ch=64, bilinear=bilinear)
        self.recon_up4 = UpFlexible(in_ch=64, skip_ch=32, out_ch=32, bilinear=bilinear)
        # Đầu ra là 1 kênh (Ảnh CT Grayscale gốc)
        self.recon_outc = nn.Conv2d(32, n_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoding ---
        x1 = self.inc(x)         # 32
        x2 = self.down1(x1)      # 64
        x3 = self.down2(x2)      # 128
        x4 = self.down3(x3)      # 256
        x5 = self.down4_conv(x4) # 512

        # --- Bridge ---
        x5_trans, attn_weights = self.transformer_bottleneck(x5)
        x5_att = self.saa_bridge(x5_trans)

        # ==========================================================
        # DECODING NHÁNH 1: SEGMENTATION (TÌM KHỐI U)
        # ==========================================================
        x6 = self.up1(x5_att, x4)
        x5_scale = F.interpolate(x5_att, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat((x5_scale, x6), 1)

        x7 = self.up2(x6_cat, x3)
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat((x6_scale, x7), 1)

        x8 = self.up3(x7_cat, x2)
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat((x7_scale, x8), 1)

        x9 = self.up4(x8_cat, x1)
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9_cat = torch.cat((x8_scale, x9), 1)

        logits = self.outc(x9_cat)

        # ==========================================================
        # DECODING NHÁNH 2: RECONSTRUCTION (VẼ LẠI ẢNH PHỔI)
        # Bắt đầu từ cùng Bottleneck x5_att để ép nó phải học giải phẫu
        # ==========================================================
        r6 = self.recon_up1(x5_att, x4)
        r7 = self.recon_up2(r6, x3)
        r8 = self.recon_up3(r7, x2)
        r9 = self.recon_up4(r8, x1)
        recon_logits = self.recon_outc(r9)

        # ==========================================================
        # RETURN LOGIC
        # ==========================================================
        if self.training:
            out_x7 = self.ds_out1(x7)
            out_x7 = F.interpolate(out_x7, size=logits.shape[2:], mode='bilinear', align_corners=False)

            out_x8 = self.ds_out2(x8)
            out_x8 = F.interpolate(out_x8, size=logits.shape[2:], mode='bilinear', align_corners=False)

            # Trả về: [List Output nhánh 1], Output nhánh 2, Attention Weights
            return [logits, out_x7, out_x8], recon_logits, attn_weights
        else:
            # Khi Test, vẫn trả về nhánh 2 nhưng ta có thể bỏ qua nó
            return logits, recon_logits, attn_weights
