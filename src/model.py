import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Bài báo sử dụng Bilinear Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            # Input của Conv phải là tổng của (in_channels từ dưới lên) + (in_channels // 2 từ skip connection)
            # Up1: 1024 (dưới lên) + 512 (skip) = 1536
            self.conv = DoubleConv(in_channels + (in_channels // 2), out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Xử lý trường hợp kích thước không khớp (do padding khi conv)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Standard Skip Connection (Cascade)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- MODULE SELF-AWARE ATTENTION (SAA) ---

class SelfAwareAttention(nn.Module):
    """
    Module cầu nối chứa TSA và GSA.
    """

    def __init__(self, in_channels, num_heads=8):
        super(SelfAwareAttention, self).__init__()

        # 1. Transformer Self Attention (TSA)
        self.num_heads = num_heads
        self.d_k = in_channels // num_heads

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 2. Global Spatial Attention (GSA)
        self.reduced_channels = in_channels // 8
        self.gsa_conv_m = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.gsa_conv_n = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.gsa_conv_w = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gsa_conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 3. Embedding Fusion
        self.gamma_tsa = nn.Parameter(torch.zeros(1))
        self.gamma_gsa = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # --- TSA Logic ---
        # (B, C, H, W) -> (B, Heads, d_k, N)
        proj_query = self.query_conv(x).view(batch_size, self.num_heads, self.d_k, -1)
        proj_key = self.key_conv(x).view(batch_size, self.num_heads, self.d_k, -1)
        proj_value = self.value_conv(x).view(batch_size, self.num_heads, self.d_k, -1)

        # Scaled Dot-product Attention: Softmax(Q * K^T / sqrt(d_k))
        energy = torch.matmul(proj_query.permute(0, 1, 3, 2), proj_key)
        attention = F.softmax(energy / (self.d_k ** 0.5), dim=-1)

        tsa_out = torch.matmul(attention, proj_value.permute(0, 1, 3, 2))
        tsa_out = tsa_out.permute(0, 1, 3, 2).contiguous().view(batch_size, channels, height, width)

        # --- GSA Logic ---
        m = self.gsa_conv_m(x).view(batch_size, self.reduced_channels, -1).permute(0, 2, 1)  # (B, N, c')
        n = self.gsa_conv_n(x).view(batch_size, self.reduced_channels, -1)  # (B, c', N)
        position_attention = F.softmax(torch.matmul(m, n), dim=-1)  # (B, N, N)
        w = self.gsa_conv_w(x).view(batch_size, channels, -1)  # (B, C, N)

        gsa_out = torch.matmul(w, position_attention.permute(0, 2, 1))
        gsa_out = gsa_out.view(batch_size, channels, height, width)
        gsa_out = self.gsa_conv_out(gsa_out)

        # --- Fusion ---
        out = (self.gamma_tsa * tsa_out) + (self.gamma_gsa * gsa_out) + x
        return out


# --- KIẾN TRÚC CHÍNH TRANSATTUNET ---

class TransAttUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(TransAttUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 1. Encoder (Standard CNN)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Lớp chuyển tiếp trước khi vào Bridge (để tăng channel lên 1024)
        self.down4_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)  # Feature Map đáy: 1024 x 32 x 32 (nếu input 512)
        )

        # 2. Bridge (Self-aware Attention Module)
        # Input của SAA là feature map sâu nhất (1024 channels)
        self.saa_bridge = SelfAwareAttention(in_channels=1024)

        # 3. Decoder (Multi-scale Skip Connections - Cascade style)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 4. Output Layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoding ---
        x1 = self.inc(x)  # (B, 64, 512, 512)
        x2 = self.down1(x1)  # (B, 128, 256, 256)
        x3 = self.down2(x2)  # (B, 256, 128, 128)
        x4 = self.down3(x3)  # (B, 512, 64, 64)

        # Bottom layer
        x5 = self.down4_conv(x4)  # (B, 1024, 32, 32)

        # --- Bridge (SAA) ---
        # Áp dụng Attention để nắm bắt ngữ cảnh toàn cục & không gian
        x5_att = self.saa_bridge(x5)

        # --- Decoding ---
        # Truyền output của SAA (x5_att) vào Decoder
        x = self.up1(x5_att, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits