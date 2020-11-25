'''Creating Two dimensional convolutional neural net layers'''

import torch.nn as nn
from collections import OrderedDict

def make_conv_layer(in_channels,
                    out_channels,
                    filter_size,
                    non_linearity=True,
                    batch_norm=False,
                    atrou_rate=1):
    
    '''Create a convolutional layer with relevant padding and batchnorm + elu
    
    Parameters
    -------
    in_channels,
    out_channels,
    filter_size,
    non_linearity=True,
    batch_norm=False,
    atrou_rate=1
    
    Return
    ------
    Sequential of layers
    '''
    
    layers = []
    if filter_size == 1:
        padding_size = 0
    elif filter_size == 3:
        padding_size = atrou_rate
    else:
        raise

    if batch_norm:
        layers.append(
            ('conv', nn.Conv2d(in_channels, out_channels, filter_size,
                               padding=padding_size, dilation=atrou_rate, bias=False))
        )
        layers.append(('bn', nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)))
    else:
        layers.append(
            ('conv', nn.Conv2d(in_channels, out_channels, filter_size,
                               padding=padding_size, dilation=atrou_rate, bias=True))
        )

    if non_linearity:
        layers.append(('elu', nn.ELU()))

    return nn.Sequential(OrderedDict(layers))

class SeparableConv2d(nn.Module):
    ''''Create sep conv 2d as not a method in Pytorch (but is in Tensorflow)'
    Separable conv = depthwise followed by pointwise convolution
    Variable bias by default set to True'''
    
    def __init__(self,in_channels,
                 out_channels,
                 filter_size=1,
                 filter_size_2=None,
                 stride=1,
                 padding=0,
                 dilation=1,
                 ):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, filter_size,
                               stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1)

    
    def forward(self,in_channels, out_channels, filter_size,
                               stride, padding, dilation):
        x = self.depthwise(in_channels, in_channels, filter_size,
                               stride, padding, dilation, groups=in_channels)
        x = self.pointwise(in_channels, out_channels)
        return x



def make_conv_sep2d_layer(in_channels,
                          out_channels,
                          channel_multiplier,
                          filter_size,
                          filter_size_2=None,
                          batch_norm=False,
                          atrou_rate=1):
    
    '''Create a convolutional seperable layer with relevant padding and batchnorm + elu
    
    Parameters
    -------
    in_channels,
    out_channels,
    channel_multiplier,
    filter_size,
    filter_size_2=None,
    batch_norm=False,
    atrou_rate=1
    
    Return
    ------
    Sequential of layers
    '''
    layers = []
    
    if filter_size_2 is None:
        filter_size_2 = filter_size
    
    h_conv = SeparableConv2d().forward(in_channels, out_channels, filter_size, 1, 0, 1)
    
    if batch_norm:
        layers.append('conv', h_conv)#bias needs to be False here
        
        layers.append(('bn', nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)))
    else:
        layers.append('conv', h_conv) #bias needs to be True here
        
    return nn.Sequential(OrderedDict(layers))

