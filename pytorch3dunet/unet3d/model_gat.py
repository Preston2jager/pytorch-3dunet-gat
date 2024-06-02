import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level


class AbstractUNet(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, graph_data, edge_index,f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, is3d=True, gat_in_features =3 , gat_out_features = 128, gat_layers = 3, gat_heads_per_layer = 4):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
                                        conv_padding, conv_upscale, dropout_prob,
                                        layer_order, num_groups, pool_kernel_size, is3d)

        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding,
                                        layer_order, num_groups, upsample, dropout_prob,
                                        is3d)

        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None
        
        self.gat_module = GAT(gat_in_features, gat_out_features, gat_layers, gat_heads_per_layer)

    def forward(self, x, graph_data, edge_index):

        encoders_features = []

        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        # GAT integration
        gat_features = self.gat_module(graph_data, edge_index)
        x = torch.cat((x, gat_features), dim=1)  # Concatenate features along the feature dimension

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(AbstractUNet):

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=True)


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=True)


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on 
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_padding=conv_padding,
                                               conv_upscale=conv_upscale,
                                               upsample=upsample,
                                               dropout_prob=dropout_prob,
                                               is3d=True)


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=False)


class ResidualUNet2D(AbstractUNet):
    """
    Residual 2DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet2D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=False)

class GAT(nn.Module):
    def __init__(self, in_features, out_features, num_layers, heads_per_layer, dropout=0.6, concat=True):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.concat = concat  # 确定最后一层是拼接还是平均

        for i in range(num_layers):
            in_channels = in_features if i == 0 else out_features * heads_per_layer[i - 1]
            out_channels = out_features if i < num_layers - 1 or not concat else out_features * heads_per_layer[-1]
            concat = True if i < num_layers - 1 else self.concat
            self.layers.append(GATConv(in_channels, out_features, heads=heads_per_layer[i],
                                       dropout=dropout, concat=concat))

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_include))
            x = F.dropout(x, p=0.6, training=self.training)
        # 最后一层不使用激活和dropout
        x = self.layers[-1](x, edge_index)
        return x

def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model'
    ])
    return model_class(**model_config)
