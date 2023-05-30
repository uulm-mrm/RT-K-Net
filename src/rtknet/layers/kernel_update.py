import torch
from torch import nn

from .batch_norm import get_norm
from .ffn import FFN

__all__ = [
    "group_feature_assembling",
    "AdaptiveKernelUpdate",
    "KernelInteraction",
]


def group_feature_assembling(mask_pred, feature_map, normalize_sigmoid_masks):
    """
    Group feature assembling as proposed in "K-Net: Towards Unified Image Segmentation"
    (https://arxiv.org/abs/2106.14855)

    Extended by normalize_sigmoid_masks to normalize the group features by their respective area

    :param mask_pred: tensor of N mask predictions, tensor shape [B, N, H, W]
    :param feature_map:  tensor of C features, shape [B, C, H, W]
    :param normalize_sigmoid_masks: bool, if true normalize sigmoid masks by their respective area
    :return: group features, shape [B, N, C]
    """
    sigmoid_masks = (mask_pred.sigmoid() > 0.5).to(feature_map.dtype)
    if normalize_sigmoid_masks:
        # Normalize masks by pixel count
        pixel_norm = sigmoid_masks.sum(-1).sum(-1, keepdim=True).unsqueeze(-1)
        pixel_norm[pixel_norm == 0] = 1
        sigmoid_masks /= pixel_norm
    else:
        sigmoid_masks = sigmoid_masks.float()
        feature_map = feature_map.float()
    with torch.cuda.amp.autocast(enabled=normalize_sigmoid_masks):
        group_features = torch.einsum("bnhw,bchw->bnc", sigmoid_masks, feature_map)

    return group_features


class AdaptiveKernelUpdate(nn.Module):
    """
    Adaptive kernel update as proposed in "K-Net: Towards Unified Image Segmentation"
    (https://arxiv.org/abs/2106.14855)
    """

    def __init__(
        self,
        in_channels=256,
        feat_channels=256,
        out_channels=None,
        gate_sigmoid=True,
        gate_norm_act=False,
        activate_out=False,
        norm="LN",
    ):
        super(AdaptiveKernelUpdate, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels, self.num_params_in + self.num_params_out)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels)
        if self.gate_norm_act:
            self.gate_norm = get_norm(norm, self.feat_channels)

        self.norm_in = get_norm(norm, self.feat_channels)
        self.norm_out = get_norm(norm, self.feat_channels)
        self.input_norm_in = get_norm(norm, self.feat_channels)
        self.input_norm_out = get_norm(norm, self.feat_channels)

        self.activation = nn.ReLU(inplace=True)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels)
        self.fc_norm = get_norm(norm, self.out_channels)

    def forward(self, update_feature, input_feature):
        update_feature = update_feature.reshape(-1, self.in_channels)
        num_proposals = update_feature.size(0)
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, : self.num_params_in].view(-1, self.feat_channels)
        param_out = parameters[:, -self.num_params_out :].view(-1, self.feat_channels)

        input_feats = self.input_layer(input_feature.reshape(num_proposals, -1, self.feat_channels))
        input_in = input_feats[..., : self.num_params_in]
        input_out = input_feats[..., -self.num_params_out :]

        gate_feats = input_in * param_in.unsqueeze(-2)
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(-2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features


class KernelInteraction(nn.Module):
    """
    Kernel interaction as proposed in "K-Net: Towards Unified Image Segmentation"
    (https://arxiv.org/abs/2106.14855)
    """

    def __init__(
        self,
        in_channels=256,
        num_heads=8,
        dropout=0.0,
        norm="LN",
    ):
        super().__init__()
        self.in_channels = in_channels

        self.attention = nn.MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = get_norm(norm, in_channels)
        self.ffn = FFN(in_channels)

    def forward(self, kernels, B, N):
        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        kernels = kernels.reshape(B, N, -1).permute(1, 0, 2)
        kernels = self.attention_norm(
            kernels + self.attention(query=kernels, key=kernels, value=kernels)[0]
        )
        # [N, B, K*K*C] -> [B, N, K*K*C] -> [B, N, K*K, C]
        kernels = kernels.permute(1, 0, 2).reshape(B, N, -1, self.in_channels)
        kernels = self.ffn(kernels)
        return kernels
