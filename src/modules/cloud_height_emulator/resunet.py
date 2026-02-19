import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet34_Weights



class ConvRelu(nn.Module):
    """ Convolution -> ReLU.

        Args:   
            in_channels : number of input channels
            out_channels : number of output channels
            kernel_size : size of convolution kernel
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)





class OutHead(nn.Module):
    def __init__(self, hidden_channels_list, out_channels):
        super(OutHead, self).__init__()

        self.convs = nn.ModuleList()
        for i in range(len(hidden_channels_list) - 1):
            self.convs.append(
                nn.Conv2d(hidden_channels_list[i], hidden_channels_list[i + 1], kernel_size=1)
            )
            self.convs.append(nn.ReLU(inplace=True))
        self.convs.append(nn.Conv2d(hidden_channels_list[-1], out_channels, kernel_size=1))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        return self.convs(x)


class Out(nn.Module):
    def __init__(self, hidden_channels_list, out_channels, n_rh):
        super(Out, self).__init__()
        self.n_rh = n_rh
        self.out_heads = nn.ModuleList()
        for i in range(n_rh):
            self.out_heads.append(OutHead(hidden_channels_list, out_channels[i]))
        
    def forward(self, x):
        out = []
        for i in range(self.n_rh):
            out.append(self.out_heads[i](x))
        return torch.cat(out, dim=1)

class Res34_Unet(nn.Module):
    """Unet model with a resnet34 encoder used for single image. 

        Args:
            pretrained : if True, use pretrained resnet34 weights.

    """

    def __init__(self, pretrained: bool = True, in_channels: int = 3, out_channels: int = 1, heads= ["regression"], heads_hidden_channels=None, **kwargs) -> None:
        super().__init__()

        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])
        self.heads = heads
        self.nb_heads = len(heads)

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(
            decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(
            decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(
            decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(
            decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        if self.nb_heads ==1:
            self.out = nn.Conv2d(
                decoder_filters[-5], out_channels, 1, stride=1, padding=0)
        else:
            assert len(out_channels)==self.nb_heads 
            self.out = Out(heads_hidden_channels, out_channels, self.nb_heads)



        self._initialize_weights()
        # pretrained argument was deprecated so we changed to weights

        # throw error if pretrained is not a bool
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained argument should be a bool")
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        if weights is not None:
            print(f"using weights from {weights}")
        encoder = torchvision.models.resnet34(weights=weights)
        if in_channels != 3:
            # reproduce the behavior of torchvision for different in_channels
            kernel_size1 = encoder.conv1.kernel_size
            stride1 = encoder.conv1.stride
            padding1 = encoder.conv1.padding
            bias1 = encoder.conv1.bias is not None
            conv1_new = nn.Conv2d(
                in_channels, 64, kernel_size=kernel_size1, stride=stride1, padding=padding1, bias=bias1)

            _w = encoder.conv1.state_dict()
            if in_channels < 3:
                conv1_new.weight.data[:, :in_channels, :,
                                      :] = _w['weight'][:, :in_channels, :, :]
            else:
                conv1_new.weight.data[:, :3, :, :] = _w['weight']
        else:
            conv1_new = encoder.conv1

        self.conv1 = nn.Sequential(
            conv1_new,
            encoder.bn1,
            encoder.relu)
        self.conv2 = nn.Sequential(
            encoder.maxpool,
            encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    
    def _initialize_weights(self) -> None:
        """ Initialize weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], 1))
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        if self.nb_heads > 1:
            dec10 = self.out(dec10)
        output =  {"logits": dec10}
        for head in self.heads:
            if "regression" in head:
                regression_dim = np.where(np.array(self.heads)=="regression")[0][0]
                output["regression"] = dec10[:,regression_dim]
            elif "segmentation" in head:
                segmentation_dim = np.where(np.array(self.heads)=="segmentation")[0][0]
                output["segmentation"] = dec10[:,segmentation_dim]
        return output

