"""
Export CAM++ 3D-Speaker model (speech_campplus_sv_zh_en_16k-common_advanced) to ONNX.
Model: 192-dim embeddings, trained on 200k speakers (VoxCeleb + CNCeleb + 3D-Speaker).

Source: https://github.com/modelscope/3D-Speaker
Paper: arXiv 2403.19971
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np

# ============================================================
# layers.py from 3D-Speaker
# ============================================================

def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False, config_str="batchnorm-relu"):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding,
                 dilation, bias, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size,
                 stride=1, dilation=1, bias=False, config_str="batchnorm-relu",
                 memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers, in_channels, out_channels, bn_channels,
                 kernel_size, stride=1, dilation=1, bias=False,
                 config_str="batchnorm-relu", memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels, bn_channels=bn_channels,
                kernel_size=kernel_size, stride=stride, dilation=dilation,
                bias=bias, config_str=config_str, memory_efficient=memory_efficient)
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ============================================================
# DTDNN.py from 3D-Speaker (CAMPPlus model)
# ============================================================

class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    def __init__(self, feat_dim=80, embedding_size=512, growth_rate=32,
                 bn_size=4, init_channels=128, config_str="batchnorm-relu",
                 memory_efficient=True):
        super(CAMPPlus, self).__init__()
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(OrderedDict([
            ("tdnn", TDNNLayer(channels, init_channels, 5, stride=2,
                               dilation=1, padding=-1, config_str=config_str)),
        ]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
                zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers, in_channels=channels,
                out_channels=growth_rate, bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size, dilation=dilation,
                config_str=config_str, memory_efficient=memory_efficient)
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str))
            channels //= 2
        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))
        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x


# ============================================================
# Export to ONNX
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export 3D-Speaker CAM++ (200k speakers, 192-dim) PyTorch checkpoint to ONNX."
    )
    parser.add_argument(
        "--pt_path",
        default="modelscope_cache/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt",
        help="Path to PyTorch .pt checkpoint from ModelScope. Download via: "
             "modelscope download --model iic/speech_campplus_sv_zh_en_16k-common_advanced",
    )
    parser.add_argument(
        "--onnx_path",
        default="models/campp-3dspeaker/campplus_cn_en_common_200k.onnx",
        help="Output ONNX file path",
    )
    args = parser.parse_args()
    pt_path = args.pt_path
    onnx_path = args.onnx_path

    print("=" * 60)
    print("CAM++ 3D-Speaker ONNX Export")
    print("Model: speech_campplus_sv_zh_en_16k-common_advanced")
    print("=" * 60)

    # 1. Build model with correct config
    print("\n[1] Building CAMPPlus model (192-dim, 80-dim fbank)...")
    model = CAMPPlus(
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
    )

    # 2. Load weights
    print("[2] Loading weights from:", os.path.basename(pt_path))
    state_dict = torch.load(pt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print("  WARNING: Missing keys:", missing)
    if unexpected:
        print("  WARNING: Unexpected keys:", unexpected)
    print("  Loaded %d parameters successfully!" % len(state_dict))

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print("  Total parameters: {:,} ({:.2f}M)".format(total_params, total_params / 1e6))

    # 3. Set eval mode
    model.eval()

    # 4. Test forward pass
    print("[3] Testing forward pass...")
    batch_size = 1
    time_steps = 200  # ~2 seconds of audio
    dummy_input = torch.randn(batch_size, time_steps, 80)
    with torch.no_grad():
        output = model(dummy_input)
    print("  Input:  {} (batch, time, 80)".format(list(dummy_input.shape)))
    print("  Output: {} (batch, {})".format(list(output.shape), output.shape[-1]))
    assert output.shape[-1] == 192, "Expected 192-dim embeddings, got %d" % output.shape[-1]

    # 5. Export to ONNX
    print("\n[4] Exporting to ONNX: %s" % onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Use legacy TorchScript exporter (dynamo=False) for compatibility
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["feats"],
        output_names=["embs"],
        dynamic_axes={
            "feats": {0: "batch", 1: "time"},
            "embs": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print("  ONNX file size: %.1f MB" % file_size)

    # 6. Verify with onnxruntime
    print("\n[5] Verifying with ONNX Runtime...")
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print("\n  === ONNX Model Info ===")
    for inp in sess.get_inputs():
        print("  Input:  name='%s', shape=%s, dtype=%s" % (inp.name, inp.shape, inp.type))
    for out in sess.get_outputs():
        print("  Output: name='%s', shape=%s, dtype=%s" % (out.name, out.shape, out.type))

    # Run inference
    test_input = np.random.randn(1, 200, 80).astype(np.float32)
    ort_output = sess.run(["embs"], {"feats": test_input})[0]
    print("\n  Test inference:")
    print("    Input:  feats shape=%s" % str(test_input.shape))
    print("    Output: embs  shape=%s" % str(ort_output.shape))

    # Compare PyTorch vs ONNX outputs
    pt_output = model(torch.from_numpy(test_input)).detach().numpy()
    max_diff = np.abs(pt_output - ort_output).max()
    print("    Max diff (PyTorch vs ONNX): %.6e" % max_diff)

    # Test with different time lengths
    print("\n  Dynamic time axis test:")
    for t in [100, 300, 500]:
        test_t = np.random.randn(1, t, 80).astype(np.float32)
        out_t = sess.run(["embs"], {"feats": test_t})[0]
        print("    time=%4d => embs shape=%s" % (t, str(out_t.shape)))

    # Test with batch > 1
    test_batch = np.random.randn(3, 200, 80).astype(np.float32)
    out_batch = sess.run(["embs"], {"feats": test_batch})[0]
    print("    batch=3, time=200 => embs shape=%s" % str(out_batch.shape))

    print("\n" + "=" * 60)
    print("DONE! ONNX model saved to:")
    print("  %s" % onnx_path)
    print("=" * 60)
