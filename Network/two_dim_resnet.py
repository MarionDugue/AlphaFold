'''Creates a 2D residual network

See 'DeepMind's algorithm to create a seperable layer' figure to get the big picture'''


import torch.nn as nn
from two_dim_convnet import make_conv_layer, make_conv_sep2d_layer
from collections import OrderedDict

class ResidualBlock(nn.Module):
    """Creates a residual block
    
    AlphaFold is composed of 220 residual blocks (number of blocks with 
    highest accuracy with least computational power) cycling through dilations 1,2,4,8
    
    """
    def __init__(self, in_channels,
                       out_channels,
                       layer_name,
                       filter_size,
                       batch_norm=False,
                       divide_channels_by=2,
                       atrou_rate=1,
                       channel_multiplier=0,
                       dropout_keep_prob=1.0):
        """A separable resnet block."""
        super().__init__()

        self.batch_norm = batch_norm
        self.dropout_keep_prob = dropout_keep_prob
        self.channel_multiplier = channel_multiplier
        
        
        #Batchnorm 
        if batch_norm:
            self.bn = nn.BatchNorm2d(in_channels, momentum=0.001, eps=0.001)
        self.elu = nn.ELU()

        # Project down: 1x1 with half size (64*64)
        self.conv_1x1h = make_conv_layer(in_channels=in_channels,
                                         out_channels=in_channels // divide_channels_by,
                                         filter_size=1,
                                         non_linearity=True,
                                         batch_norm=batch_norm)

        # Dilate: 3x3 with half size
        if channel_multiplier == 0:
            self.conv_3x3h = make_conv_layer(in_channels=in_channels // divide_channels_by,
                                             out_channels=in_channels // divide_channels_by,
                                             filter_size=filter_size,
                                             non_linearity=True,
                                             batch_norm=batch_norm,
                                             atrou_rate=atrou_rate)
        else:
            self.conv_sep3x3h = make_conv_sep2d_layer(in_channels=in_channels // divide_channels_by,
                                                      out_channels=in_channels // divide_channels_by,
                                                      channel_multiplier=channel_multiplier,
                                                      filter_size=filter_size,
                                                      batch_norm=batch_norm,
                                                      atrou_rate=atrou_rate)

        # Project up: 1x1 back to normal size without relu
        self.conv_1x1 = make_conv_layer(in_channels=in_channels // divide_channels_by,
                                        out_channels=out_channels,
                                        filter_size=1,
                                        non_linearity=False,
                                        batch_norm=False)
        #dropout
        if dropout_keep_prob < 1.0:
            self.dropout = nn.Dropout(1-dropout_keep_prob)

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(x)
            out = self.elu(out)
        else:
            out = self.elu(x)

        out = self.conv_1x1h(out)
        if self.channel_multiplier == 0:
            out = self.conv_3x3h(out)
        else:
            out = self.conv_sep3x3h(out)
        out = self.conv_1x1(out)

        if self.dropout_keep_prob < 1.0:
            out = self.dropout(out)

        out += x
        return out


def make_two_dim_resnet(num_features,
                        num_predictions=1,
                        num_channels=32,
                        num_layers=2,
                        filter_size=3,
                        final_non_linearity=False,
                        batch_norm=False,
                        atrou_rates=None,
                        channel_multiplier=0,
                        divide_channels_by=2,
                        resize_features_with_1x1=False,
                        dropout_keep_prob=1.0):
    '''
      Create a 2d resnet thanks to make_conv_layer method and ResidualBlock class
      
      Parameters
    ----------
    num_features,
    num_predictions=1,
    num_channels=32,
    num_layers=2,
    filter_size=3,
    final_non_linearity=False,
    batch_norm=False,
    atrou_rates=None,
    channel_multiplier=0,
    divide_channels_by=2,
    resize_features_with_1x1=False,
    dropout_keep_prob=1.0
    
    Returns
    -------
    Sequential of layers consituting the 2d resnet'''
            
    #Note: à trou rate is equivalent to dilution factor
    if atrou_rates is None: atrou_rates = [1]

    layers = []
    non_linearity = True
    for i_layer in range(num_layers):
        in_channels = num_channels
        out_channels = num_channels
        curr_atrou_rate = atrou_rates[i_layer % len(atrou_rates)]

        if i_layer == 0:
            in_channels = num_features
        if i_layer == num_layers - 1:
            out_channels = num_predictions
            non_linearity = final_non_linearity

        #For first and penulitmate layer: Create convolutional layer
        if i_layer == 0 or i_layer == num_layers - 1:
            layer_name = f'conv{i_layer+1}'
            initial_filter_size = 1 if resize_features_with_1x1 else filter_size
            conv_layer = make_conv_layer(in_channels=in_channels,
                                        out_channels=out_channels,
                                        filter_size=initial_filter_size,
                                        non_linearity=non_linearity,
                                        atrou_rate=curr_atrou_rate)
            
        #For in-between layers, create residual blocks
        else:
            layer_name = f'res{i_layer+1}'
            conv_layer = ResidualBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       layer_name=layer_name,
                                       filter_size=filter_size,
                                       batch_norm=batch_norm,
                                       atrou_rate=curr_atrou_rate,
                                       channel_multiplier=channel_multiplier,
                                       divide_channels_by=divide_channels_by,
                                       dropout_keep_prob=dropout_keep_prob)
        layers.append((layer_name, conv_layer))
    
    return nn.Sequential(OrderedDict(layers))
