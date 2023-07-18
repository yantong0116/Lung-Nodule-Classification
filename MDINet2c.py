import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
# from torchsummary import summary

def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True)
    )
    return layer

class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()

        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=5, stride=1, padding=2, bias=False)),

        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class MDINet2c(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False, num_init_features=64, 
                block_config=(6), bn_size=4, growth_rate=32, 
                drop_rate=0.3, memory_efficient=False):
        super(MDINet2c, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channel=48, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(2, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        
        # num_layers = 2
        # num_layers = 3
        # num_layers = 4
        num_layers = 5
        # num_layers = 6
        # num_layers = 7
        # num_layers = 8

        # for num_layers in enumerate(block_config):
        block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.features.add_module('denseblock%d' % (0 + 1), block)
        num_features = num_features + num_layers * growth_rate

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.block2 = nn.Sequential(
            # conv_relu(48, 48, kernel=1),

            # layers = 8
            # conv_relu(160, 144, kernel=3, padding=1),

            # layers = 7
            # conv_relu(144, 144, kernel=3, padding=1),

            # layers = 6
            # conv_relu(256, 128, kernel=3, padding=1),

            # layers = 5
            conv_relu(224, 128, kernel=3, padding=1),

            # layers = 4
            # conv_relu(192, 128, kernel=3, padding=1), 

            # layers = 3
            # conv_relu(160, 128, kernel=3, padding=1), 

            # layers = 2
            # conv_relu(128, 128, kernel=3, padding=1), 
            nn.MaxPool2d(3, 2, 1), 

            conv_relu(128, 48, kernel=3, padding=1), 
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            inception(48, 12, 24, 12, 24, 12, 12),
            # nn.MaxPool2d(3, 2)
        )

        self.classifier = nn.Sequential(
            # 0.5
            nn.Dropout(p=0.6),
            nn.Linear(5184, 1024),
            nn.ReLU(inplace=True),
            # 0.5
            nn.Dropout(p=0.6),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            # 0.5
            nn.Dropout(p=0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out1 = self.features(x[:, 0])
        out1 = self.block2(out1)
        out1 = self.block3(out1)

        out2 = self.features(x[:, 1])
        out2 = self.block2(out2)
        out2 = self.block3(out2)

        out3 = self.features(x[:, 2])
        out3 = self.block2(out3)
        out3 = self.block3(out3)

        out = torch.cat((out1, out2, out3), 1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        classify_result = torch.sigmoid(out)
        return classify_result
