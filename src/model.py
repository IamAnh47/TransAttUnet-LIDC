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

class TransAttUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(TransAttUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 1. Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        # 2. Bridge
        self.saa_bridge = SelfAwareAttention(in_channels=1024)

        # 3. Decoder (Multi-scale Dense Connections)
        # SỬ DỤNG UpFlexible để kiểm soát chính xác Channels

        # Up1: Input từ Bridge (1024) + Skip x4 (512)
        self.up1 = UpFlexible(in_ch=1024, skip_ch=512, out_ch=512, bilinear=bilinear)

        # Up2:
        #   Input Vertical = x6_cat = x5_scale (1024) + x6 (512) = 1536 channels
        #   Skip x3 = 256
        self.up2 = UpFlexible(in_ch=1536, skip_ch=256, out_ch=256, bilinear=bilinear)

        # Up3:
        #   Input Vertical = x7_cat = x6_scale (512) + x7 (256) = 768 channels
        #   (Lưu ý: x6_scale lấy từ output của up1 là 512)
        #   Skip x2 = 128
        self.up3 = UpFlexible(in_ch=768, skip_ch=128, out_ch=128, bilinear=bilinear)

        # Up4:
        #   Input Vertical = x8_cat = x7_scale (256) + x8 (128) = 384 channels
        #   Skip x1 = 64
        self.up4 = UpFlexible(in_ch=384, skip_ch=64, out_ch=64, bilinear=bilinear)

        # 4. Output Layer
        #   Input = x9_cat = x8_scale (128) + x9 (64) = 192 channels
        self.outc = nn.Conv2d(192, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoding ---
        x1 = self.inc(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4_conv(x4)  # 1024

        # --- Bridge ---
        x5_att = self.saa_bridge(x5)  # 1024

        # --- Decoding (Dense Logic) ---

        # Block 1
        x6 = self.up1(x5_att, x4)  # Out: 512

        # Dense connect 1: Bridge(1024) + x6(512) -> 1536
        x5_scale = F.interpolate(x5_att, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat((x5_scale, x6), 1)

        # Block 2
        x7 = self.up2(x6_cat, x3)  # Out: 256

        # Dense connect 2: x6(512) + x7(256) -> 768
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat((x6_scale, x7), 1)

        # Block 3
        x8 = self.up3(x7_cat, x2)  # Out: 128

        # Dense connect 3: x7(256) + x8(128) -> 384
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat((x7_scale, x8), 1)

        # Block 4
        x9 = self.up4(x8_cat, x1)  # Out: 64

        # Dense connect 4: x8(128) + x9(64) -> 192
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9_cat = torch.cat((x8_scale, x9), 1)

        logits = self.outc(x9_cat)
        return logits