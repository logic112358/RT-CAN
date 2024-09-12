import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchinfo import summary
from thop import profile

# from CrossViewAttention import CMAtt,CMAtt2
# from model.CrossViewAttention import CMAtt,CMAtt2
# from model.fuseDiff import FusionNetwork
# from fuseDiff import FusionNetwork

class Att_Enhance(nn.Module):

    def __init__(self, out_channels):

        super().__init__()

        self.w = nn.Sequential(

            nn.Conv2d(out_channels *2 , out_channels, kernel_size=1, stride=1, padding=0, bias=True),

            nn.BatchNorm2d(out_channels),

            nn.Sigmoid()

        )

        self.cse =  Channel_Attention(out_channels)

        self.sse = Spatial_Attention(out_channels)


        self.relu = nn.ReLU(inplace=True)


    def forward(self, thermal, rgb):

        weight = self.relu(torch.cat([thermal, rgb], dim=1))

        weight = self.w(weight)

        out = thermal * weight

        ca = self.cse(out)

        out = out * ca

        sa = self.sse(out)

        out = out * sa

        return out

class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gmp_pool(x).permute(0, 2, 3, 1)
        x = self.down(x)
        max_out = self.up(self.act(x)).permute(0, 3, 1, 2)
        out = self.sigmoid(max_out)
        return out

class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv1(x)
        out = self.softmax(x1)
        return out

class Att_avg_pool(nn.Module):
    def __init__(self, dim, reduction):
        super(Att_avg_pool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5, force_dropout: bool = False):
        super().__init__()
        self.force_dropout = force_dropout
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.training or self.force_dropout)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AttentionConcat(nn.Module):
    def __init__(self, in_channels=4):
        super(AttentionConcat, self).__init__()
        self.attention_weights = nn.Parameter(torch.ones(in_channels))

    def forward(self, x_list):
        # Normalize attention weights
        attn_weights = torch.softmax(self.attention_weights, dim=0)
        
        # Element-wise multiplication with attention weights

        for i in range(len(x_list)):
            x_list[i] = x_list[i] * attn_weights[i]

        # Concatenate along channel dimension
        x_cat = torch.cat(tuple(x_list), dim=1)

        return x_cat

# Global Contextual module

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = BasicConv2d(in_channel, out_channel, 1)
        self.branch1_1 = BasicConv2d(
            out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        self.branch1_2 = BasicConv2d(
            out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))

        self.branch2 = BasicConv2d(in_channel, out_channel, 1)
        self.branch2_1 = BasicConv2d(
            out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2))
        self.branch2_2 = BasicConv2d(
            out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0))

        self.branch3 = BasicConv2d(in_channel, out_channel, 1)
        self.branch3_1 = BasicConv2d(
            out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_2 = BasicConv2d(
            out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0))

        # self.se = Att_avg_pool(out_channel, 4)
        # self.catt_list =nn.ModuleList()
        # for i in range(4):
        #     self.catt_list.append(Channel_Attention(out_channel))
        # self.attncat = AttentionConcat()
        self.catt = Channel_Attention(4*out_channel,4)
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x0 = self.branch0(x)
        # x0 = self.se(x0)
        x1 = self.branch1_2(self.branch1_1(self.branch1(x)))
        # x1 = self.se(x1)
        x2 = self.branch2_2(self.branch2_1(self.branch2(x)))
        # x2 = self.se(x2)
        x3 = self.branch3_2(self.branch3_1(self.branch3(x)))
        # x3 = self.se(x3)

        # x_list = [x0,x1,x2,x3]

        # for i in range(4):
        #     catt = self.catt_list[i]
        #     x_list[i] = catt(x_list[i])
        
        # x_cat =self.attncat(x_list)
        # x_cat = self.conv_cat(x_cat)
        # x_add = x0 + x1 + x2 + x3

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = x_cat * self.catt(x_cat)
        x_cat = self.conv_cat(x_cat)
        
        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation_init(nn.Module):
    def __init__(self, channel,n_class=2, mode='None'):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        #########################################################################################################
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        # self.se = Att_avg_pool(3 * channel, 4)
        if mode == 'out':
            self.conv = nn.Conv2d(3*channel, n_class, 1)
        if mode == 'splat':
            self.conv = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        ##########################################################################################################
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(
            x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        ##########################################################################################################
        x2_2 = self.conv_concat2(
            torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1))
        x3_2 = self.conv_concat3(
            torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        x = self.conv(self.conv4(x3_2))
        return x

# Refinement flow

class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
               * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, attention, x1, x2, x3):
        x1 = x1 + torch.mul(x1, self.upsample2(attention))
        x2 = x2 + torch.mul(x2, self.upsample2(attention))
        # x1 = x1 + torch.mul(x1, attention)
        # x2 = x2 + torch.mul(x2, attention)
        x3 = x3 + torch.mul(x3, attention)
        return x1, x2, x3
####################################################################################################


class mini_Aspp(nn.Module):
    def __init__(self, channel):
        super(mini_Aspp, self).__init__()
        self.conv_6 = nn.Conv2d(
            channel, channel, kernel_size=3,  stride=1, padding=6,  dilation=6)
        self.conv_12 = nn.Conv2d(
            channel, channel, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv_18 = nn.Conv2d(
            channel, channel, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.bn(self.conv_6(x))
        x2 = self.bn(self.conv_12(x))
        x3 = self.bn(self.conv_18(x))
        feature_map = x1 + x2 + x3
        return feature_map
####################################################################################################



class FA_encoder(nn.Module):
    def __init__(self, num_resnet_layers,dropout_rate: float = .09):
        super(FA_encoder, self).__init__()
        self.num_resnet_layers = num_resnet_layers
        if self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048
        ########  Thermal ENCODER  ########
        self.encoder_thermal_conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(
            torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encode_dropout = MCDropout(dropout_rate)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        self.ae0 = Att_Enhance(64)
        self.ae1 = Att_Enhance(256)
        self.ae2 = Att_Enhance(512)
        self.ae3 = Att_Enhance(1024)
        self.ae4 = Att_Enhance(2048)

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        # 增加MCD机制
        self.encode_dropout = MCDropout(dropout_rate)
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

    def forward(self, input):
        rgb = input[:, :3]
        thermal = input[:, 3:]
        ######################################################################
        # layer0
        ######################################################################
        rgb = self.encoder_rgb_conv1(rgb)
     
        rgb = self.encoder_rgb_bn1(rgb)

        rgb = self.encoder_rgb_relu(rgb)

        rgb = self.encoder_rgb_maxpool(rgb)

        thermal = self.encoder_thermal_conv1(thermal)

        thermal = self.encoder_thermal_bn1(thermal)

        thermal = self.encoder_thermal_relu(thermal)

        thermal = self.encoder_thermal_maxpool(thermal)

        ######################################################################

        # fuse0 = thermal - rgb # 减
        # cut0 = thermal + rgb # 加
        # fuse0,weight0 = self.fn0(thermal,rgb)
        # cut0 = thermal - fuse1
        # cut0 = self.cmatt_0(T= thermal, RGB=rgb)

        fuse0 = self.ae0(thermal,rgb)

        ######################################################################
        # layer1
        ######################################################################
        rgb1 = self.encoder_rgb_layer1(rgb)
        thermal1 = self.encoder_thermal_layer1(thermal)

        # fuse1 = thermal1 - rgb1 #剪
        # cut1 = thermal1 + rgb1 #加
        # cut1 = torch.abs(cut1)
        # fuse1,weight1 = self.fn1(thermal1,rgb1)
        # cut1 = self.cmatt_1(T= thermal1, RGB=rgb1)

        fuse1 = self.ae1(thermal1,rgb1)

        ######################################################################
        # layer2
        ######################################################################
        rgb2 = self.encoder_rgb_layer2(rgb1)
        thermal2 = self.encoder_thermal_layer2(thermal1)

        # fuse2 = thermal2 - rgb2 # 减
        # cut2 = thermal2 + rgb2 # 加尝试
        # cut2 = self.cmatt_2(T= thermal2,RGB=rgb2)
        # fuse2,weight2 = self.fn2(thermal2,rgb2)

        fuse2 = self.ae2(thermal2,rgb2)


        ######################################################################
        # layer3
        ######################################################################
        rgb3 = self.encoder_rgb_layer3(rgb2)
        thermal3 = self.encoder_thermal_layer3(thermal2)

        # fuse3 = thermal3 - rgb3 # 减
        # cut3 = thermal3 + rgb3 # 加
        # cut3 = self.cmatt_3(T= thermal3,RGB=rgb3)
        # fuse3,weight3 = self.fn3(thermal3,rgb3)

        fuse3 = self.ae3(thermal3,rgb3)

        ######################################################################
        # layer4
        ######################################################################
        rgb4 = self.encoder_rgb_layer4(rgb3)
        thermal4 = self.encoder_thermal_layer4(thermal3)
        
        # fuse4 = thermal4 - rgb4# 减
        # cut4 = thermal4 + rgb4# 加
        # cut4 = self.cmatt_4(T= thermal4,RGB=rgb4)
        # fuse4,weight4 = self.fn4(thermal4,rgb4)

        fuse4 = self.ae4(thermal4,rgb4)


        ######################################################################
        fuse = [fuse0, fuse1, fuse2, fuse3, fuse4]
        thermal = [thermal, thermal1, thermal2, thermal3, thermal4]
        rgb = [rgb, rgb1, rgb2, rgb3, rgb4]
        # weight = [weight0,weight1,weight2,weight3,weight4]
        return fuse, thermal, rgb


class TransBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(
                planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out

class Cascaded_decoder(nn.Module):
    def __init__(self, n_class=2, channel=64):
        super(Cascaded_decoder, self).__init__()
        ########  DECODER  ########
        self.rfb2_1 = GCM(512, channel)
        self.rfb3_1 = GCM(1024, channel)
        self.rfb4_1 = GCM(2048, channel)
        self.rfb0_2 = GCM(64, channel)
        self.rfb1_2 = GCM(256, channel)
        self.rfb2_2 = GCM(512, channel)
        self.agg1 = aggregation_init(channel,n_class, mode='out')
        self.agg1_splat = aggregation_init(channel, mode='splat')
        self.agg2 = aggregation_final(channel)
        self.miniaspp = mini_Aspp(channel)
        self.HA = Refine()
        ######################################################################
        # upsample function
        self.upsample = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        ######################################################################
        # Components of PTM module
        self.inplanes = channel
        self.agant1 = self._make_agant_layer(channel*3, channel)
        self.deconv1 = self._make_transpose_layer(
            TransBottleneck, channel, 3, stride=2)
        # self.inplanes = channel/2
        self.agant2 = self._make_agant_layer(channel, channel)
        self.deconv2 = self._make_transpose_layer(
            TransBottleneck, channel, 3, stride=2)
        ######################################################################
        # test3
        ######################################################################
        self.out2_conv = nn.Conv2d(channel, n_class, kernel_size=1)
        ######################################################################

    def _make_transpose_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes))
        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))

        self.inplanes = planes
        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def forward(self, x):
        ######################################################################
        rgb, rgb1, rgb2, rgb3, rgb4 = x[0], x[1], x[2], x[3], x[4]
        ######################################################################
        # produce initial saliency map by decoder1
        ######################################################################
        x2_1 = self.rfb2_1(rgb2)
        ux2_1 = self.upsample2(x2_1)
        x3_1 = self.rfb3_1(rgb3)
        ux3_1 = self.upsample4(x3_1)
        x4_1 = self.rfb4_1(rgb4)
        ux4_1 = self.upsample(x4_1)
        attention_gate = torch.sigmoid(self.agg1_splat(x4_1, x3_1, x2_1))
        ##############################################################################
        x, x1, x5 = self.HA(attention_gate, rgb, rgb1, rgb2)
        x0_2 = self.rfb0_2(x)
        ux0_2 = x0_2
        x1_2 = self.rfb1_2(x1)
        ux1_2 = x1_2
        x5_2 = self.rfb2_2(x5)
        # ux5_2 = self.upsample2(x5_2)
        ##############################################################################
        # feature_map = ux5_2 + ux1_2 + ux2_1 + ux3_1 + ux0_2 + ux4_1
        # feature_map = self.miniaspp(feature_map)
        ##############################################################################
        hight_output = self.upsample(self.agg1(x4_1, x3_1, x2_1))
        ##############################################################################
        # Refine low-layer features by initial map
        ##############################################################################
        # PTM module
        ##############################################################################
        # y = feature_map
        y = self.agg2(x5_2,x1_2,x0_2)
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)
        ######################################################################
        return hight_output, y
        

class GasSegNet(nn.Module):
    def __init__(self, n_class,num_resnet_layers=50):
        super(GasSegNet, self).__init__()
        self.FA_encoder = FA_encoder(num_resnet_layers)
        self.CD = Cascaded_decoder(n_class)


    def forward(self, x):
        fuse,thermal,rgb = self.FA_encoder(x)
        # fuse,thermal,rgb = self.FA_encoder(x)
        diff = fuse

        out, out_1 = self.CD(diff)
        return out, out_1, fuse,thermal,rgb


def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 512, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 512, 640).cuda(0)
    RTCAN_Net = GasSegNet(2).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    out = RTCAN_Net(input)
    print(out)

def summarize():
    model = GasSegNet(2).cuda(0)
    summary(model, input_size=(4,4,512,640))

def model_stat():
    model = GasSegNet(2).cuda(0)
    input = torch.randn(1,4,512,640).cuda()
    from thop import profile

    flops, params = profile(model, inputs=(input, ))
    # flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    print("params=", str(params/1e6)+'{}'.format("M"))


if __name__ == '__main__':
    # summarize()
    unit_test()
    # model_stat()