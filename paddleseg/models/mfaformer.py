import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle.nn import Conv2D
from paddleseg.models.backbones.transformer_utils import (DropPath, ones_,
                                                          to_2tuple, zeros_)
from paddle import Tensor
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.models.ddrnet import DAPPM
from paddleseg.cvlibs import manager, param_init
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

class ECA(nn.Layer):
    def __init__(self, k_size=7, dilations=[1,2,4,8]):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.dilations_conv = nn.LayerList([nn.Conv1D(1,1, kernel_size=k_size, dilation=dilation, padding=(k_size - 1)//2 * dilation,bias_attr=False )
                                            for dilation in dilations])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4,1, bias_attr=False)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose([0, 2, 1])
        conv_outputs = [conv(y) for conv in self.dilations_conv]
        y = paddle.concat(conv_outputs, axis=1)
        y = y.transpose([0,2,1])
        y = self.relu(y)
        y = self.fc(y)
        y = self.sigmoid(y)
        y = y.unsqueeze(-1)
        return x * y

class SAM(nn.Layer):
    def __init__(self):  
        super(SAM, self).__init__()
        self.conv_after_concat = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)  
        self.sigmoid_spatial = nn.Sigmoid()  

    def forward(self, x):  
        # Spatial Attention Module  
        module_input = x  
        avg = paddle.mean(x, axis=1, keepdim=True)  
        mx = paddle.argmax(x, axis=1, keepdim=True)
        mx = paddle.cast(mx, 'float32')
        x = paddle.concat([avg, mx], axis=1)
        x = self.conv_after_concat(x)  
        x = self.sigmoid_spatial(x)  
        x = module_input * x  

        return x

class ECAM(nn.Layer):
    def __init__(self):
        super().__init__()
        self.eca = ECA()
        self.sam = SAM()
        
    def forward(self, inputs):
        x = self.eca(inputs)
        x = self.sam(x)
        return x
    

def get_depthwise_conv(dim, kernel_size=3):
    if isinstance(kernel_size, int):
        kernel_size = to_2tuple(kernel_size)
    padding = tuple([k // 2 for k in kernel_size])
    return Conv2D(
        dim, dim, kernel_size, padding=padding, bias_attr=True, groups=dim)

class AttentionModule(nn.Layer):
    """
    AttentionModule Layer, which contains some depth-wise strip convolutions.

    Args:
        dim (int): Number of input channels.
        kernel_sizes (list[int], optional): The height or width of each strip convolution kernel. Default: [7, 11, 21].
    """

    def __init__(self, dim, kernel_sizes=[7, 11, 21]):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)

        self.dwconvs = nn.LayerList([
            nn.Sequential((f"conv{i+1}_1", get_depthwise_conv(dim, (1, k))),
                          (f"conv{i+1}_2", get_depthwise_conv(dim, (k, 1))))
            for i, k in enumerate(kernel_sizes)
        ])

        self.conv_out = nn.Conv2D(dim, dim, 1)
        
    def forward(self, x):
        u = paddle.clone(x)
        attn = self.conv0(x)

        attns = [m(attn) for m in self.dwconvs]

        attn += sum(attns)

        attn = self.conv_out(attn)

        return attn * u

class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x     

    
class ConvBNAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 norm=nn.BatchNorm2D,
                 act=None,
                 bias_attr=False):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=None if bias_attr else False)
        self.act = act() if act is not None else nn.Identity()
        self.bn = norm(out_channels, bias_attr=None) \
            if norm is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InjectionMultiSumallmultiallsum(nn.Layer):
    def __init__(self, in_channels=(64, 128, 256, 384), out_channels=256):
        super(InjectionMultiSumallmultiallsum, self).__init__()
        self.embedding_list = nn.LayerList()
        self.act_embedding_list = nn.LayerList()
        self.mg = AttentionModule(512)
        self.dappm = DAPPM(512, 128, 512)
        self.act_list = nn.LayerList()
        for i in range(len(in_channels)):
            self.embedding_list.append(
                ConvBNAct(
                    in_channels[i], out_channels, kernel_size=1))
            self.act_embedding_list.append(
                ConvBNAct(
                    in_channels[i], out_channels, kernel_size=1))
            self.act_list.append(nn.Sigmoid())

    def forward(self, inputs):  # x_x8, x_x16, x_x32, x_x64
        low_feat1 = F.interpolate(inputs[0], scale_factor=1.0, mode="bilinear")
        low_feat1_act = self.act_list[0](self.act_embedding_list[0](low_feat1))
        low_feat1 = self.embedding_list[0](low_feat1)

        low_feat2 = F.interpolate(
            inputs[1], size=low_feat1.shape[-2:], mode="bilinear")
        low_feat2_act = self.act_list[1](
            self.act_embedding_list[1](low_feat2))  # x16
        low_feat2 = self.embedding_list[1](low_feat2)
        high_feat_act = F.interpolate(
            self.act_list[2](self.act_embedding_list[2](inputs[2])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat = F.interpolate(
            self.embedding_list[2](self.dappm(inputs[2])),
            size=low_feat2.shape[2:],
            mode="bilinear")

        res = low_feat1_act * low_feat2_act * high_feat_act * (
            low_feat1 + low_feat2) + high_feat
        res = self.mg(res)
        return res

@manager.MODELS.add_component
classMFAFormer(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 stride_attention=True,
                 pretrained=None):
        super(MSAFormer, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels
        self.ecam = ECAM()
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=48)
        self.feature_fusion = InjectionMultiSumallmultiallsum(in_channels=[c2_in_channels, c3_in_channels, c4_in_channels], out_channels=embedding_dim)
        self.dropout = nn.Dropout2D(0.1)
        self.low_linear_fuse = layers.ConvBNReLU(in_channels=embedding_dim + 48 ,
                                             out_channels=embedding_dim,
                                             kernel_size=1,
                                             bias_attr=False)
        self.linear_pred = nn.Conv2D(embedding_dim,
                                     self.num_classes,
                                     kernel_size=1)
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats
        c1_shape = c1.shape
        ff = self.feature_fusion([c2, c3, c4])
        ff = F.interpolate(ff,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)
        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])
        _c = self.low_linear_fuse(paddle.concat([ff, _c1], axis=1))
        _c = self.ecam(_c)
        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return [
            F.interpolate(logit,
                          size=x.shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
        ]
    
            
