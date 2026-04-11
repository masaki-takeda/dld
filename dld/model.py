import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from model_utils import GradExtractor, Flatten, fix_module


def calc_coverage_length(kernel_size, dilation_sizes):
    """ Calculate the coverage """
    coverage_length = 1
    
    for dilation_size in reversed(dilation_sizes):
        coverage_length += 2 * dilation_size * (kernel_size-1)

    return coverage_length


def calc_coverage_length_with_level(kernel_size, level_size):
    """ Calculate the coverage when the level is specified """
    dilation_sizes = []
    for i in range(level_size):
        dilation_size = 2 ** i
        dilation_sizes.append(dilation_size)        
    coverage_length = calc_coverage_length(kernel_size, dilation_sizes)    
    return coverage_length


def get_frame_size_with_duration_type(duration_type):
    if duration_type == 'normal':
        frame_size = 250
    elif duration_type == 'short':
        frame_size = 125
    elif duration_type == 'long':
        frame_size = 375
    else:
        assert False
    return frame_size
    

def calc_required_level(kernel_size, duration_type='normal'):
    """ Calculate the number of levels required according to the kernel size """
    frame_size = get_frame_size_with_duration_type(duration_type)
    
    max_level = 100
    for i in range(max_level):
        coverage_length = calc_coverage_length_with_level(kernel_size, i)
        if coverage_length >= frame_size:
            print('Required level_size={} for kernel_size={} (duration_type={}, frame_size={})'.format(
                i, kernel_size, duration_type, frame_size))
            return i
    return None


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout,
                 use_residual):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs,
                                           n_outputs,
                                           kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs,
                                           n_outputs,
                                           kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1,
                                 self.chomp1,
                                 nn.ReLU(),
                                 self.dropout1,
                                 self.conv2,
                                 self.chomp2,
                                 nn.ReLU(),
                                 self.dropout2)

        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        else:
            # When "n_input" and "n_output" are both 63 (default)
            self.downsample = None
        self.last_relu = nn.ReLU()
        self.use_residual = use_residual
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.use_residual:
            # When using Residual connection (default)
            if self.downsample is None:
                # When "n_input" and "n_output" are both 63 (default)
                res = x
            else:
                res = self.downsample(x)
            return self.last_relu(out + res)
        else:
            # When Residual connection is not used
            return self.last_relu(out)

    def store_grad(self, grad):
        self.grad = grad
    
    def clear_grad_cam(self):
        self.grad = None
        self.feature = None
        self.hook.remove()
        self.hook = None
        
    def forward_grad_cam(self, x):
        if x.requires_grad == False:
            x.requires_grad = True
        self.feature = x
        self.hook = self.feature.register_hook(self.store_grad)
        return self.forward(x)


class EEGTCNModelSub(nn.Module):
    def __init__(self,
                 num_channels,
                 kernel_size,
                 dropout,
                 use_residual,
                 duration_type):
        super(EEGTCNModelSub, self).__init__()
        
        self.linear = nn.Linear(num_channels[-1], 1)
        
        layers = []
        num_levels = len(num_channels)
        dilation_sizes = []
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            dilation_sizes.append(dilation_size)
            
            if i == 0:
                in_channels = 63
            else:
                in_channels = num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout,
                                     use_residual=use_residual)]
        
        coverage_length = calc_coverage_length(kernel_size, dilation_sizes)
        print('TCN coverage_length={}'.format(coverage_length))

        frame_size = get_frame_size_with_duration_type(duration_type)
        
        if coverage_length < frame_size:
            print('[WARNING] coverage length is shorter than total EEG sequence length: {}',
                  coverage_length)
        
        self.network = nn.Sequential(*layers)
        # (batch, 63, 250)

        # Leave layers (for "forward_grad_cam()")
        self.layers = layers

    def clear_grad_cam(self):
        for layer in self.layers:
            layer.clear_grad_cam()
        
    def forward(self, x):
        y1 = self.network(x)
        # When using only the last output as well as in STNN
        h = self.linear(y1[:,:,-1])
        return h

    def forward_for_combined(self, x):
        """ Output with the last Linear removed for Combined """
        y1 = self.network(x)
        return y1[:,:,-1]

    def forward_grad_cam(self, x, for_combined=False):
        h = x
        for i,layer in enumerate(self.layers):
            h = layer.forward_grad_cam(h)
        if not for_combined:
            h = self.linear(h[:,:,-1])
        else:
            h = h[:,:,-1]
        return h


class EEGTCNModel(nn.Module):
    """ TCN using only the last time step of the last level """
    def __init__(self,
                 kernel_size=7,
                 level_size=7,
                 level_hidden_size=63,
                 dropout=0.2,
                 use_residual=True,
                 duration_type='normal'):
        
        super(EEGTCNModel, self).__init__()

        if level_size < 0:
            # Automatically calculate the optimal level
            level_size = calc_required_level(kernel_size, duration_type)
        
        num_channels=[level_hidden_size] * level_size
        self.eeg_net = EEGTCNModelSub(num_channels=num_channels,
                                      kernel_size=kernel_size,
                                      dropout=dropout,
                                      use_residual=use_residual,
                                      duration_type=duration_type)
        
    def forward(self, x):
        h = self.eeg_net(x)
        return torch.sigmoid(h)
    
    def forward_raw(self, x):
        """ Outputs up to logits """
        h = self.eeg_net(x)
        return h

    def forward_grad_cam(self, x):
        h = self.eeg_net.forward_grad_cam(x)
        # Without sigmoid() here
        return h

    def get_cam_gradients(self):
        grads = []
        for layer in self.eeg_net.layers:
            grads.append(layer.grad)
        return grads
            
    def get_cam_features(self):
        features = []
        for layer in self.eeg_net.layers:
            features.append(layer.feature)
        return features

    def clear_grad_cam(self):
        self.eeg_net.clear_grad_cam()


class FMRIModel(nn.Module):
    """
    With BatchNorm3d (Former: FMRIMiddleModel2)
    """
    def __init__(self, fmri_ch_size=1):
        super(FMRIModel, self).__init__()
        self.fmri_net = torch.nn.Sequential(
            nn.Conv3d(in_channels=fmri_ch_size,
                      out_channels=8,
                      kernel_size=7,
                      stride=2,
                      padding=0), # (-1, 8, 37, 45, 37)
            nn.BatchNorm3d(8), # BatchNormalization
            nn.ReLU(),
            nn.Conv3d(in_channels=8,
                      out_channels=16,
                      kernel_size=5,
                      stride=2,
                      padding=0), # (-1, 16, 17, 21, 17)
            nn.BatchNorm3d(16), # BatchNormalization
            nn.ReLU(),
            nn.Conv3d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=3,
                      padding=1), # (-1, 32, 6, 7, 6)
            nn.BatchNorm3d(32), # BatchNormalization
            nn.ReLU(),
            nn.Dropout3d(p=0.5),
            Flatten(),
            # Large unit size (8064x128), (OHBM's unit size is smaller (2304x128))
            nn.Linear(in_features=6*7*6*32,
                      out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
    def forward(self, x):
        h = self.fmri_net(x)
        return torch.sigmoid(h)

    def forward_raw(self, x):
        h = self.fmri_net(x)
        return h

    def forward_grad_cam(self, x):
        self.grad_extractor = GradExtractor()
        h = self.grad_extractor.forward(self.fmri_net, x, '8')
        # Relu() after [8]=Conv3d
        return h # Without sigmoid() here

    def get_cam_gradients(self):
        return self.grad_extractor.grad

    def get_cam_features(self):
        return self.grad_extractor.feature


class CombinedTCNModel(nn.Module):
    """ Combined model using TCN """
    def __init__(self,
                 fmri_ch_size=1,
                 kernel_size=7,
                 level_size=7,
                 level_hidden_size=63,
                 dropout=0.2,
                 use_residual=True,
                 combined_hidden_size=256,
                 combined_layer_size=1):
        super(CombinedTCNModel, self).__init__()
        
        # FMRIModel based
        self.fmri_net = torch.nn.Sequential(
            nn.Conv3d(in_channels=fmri_ch_size,
                      out_channels=8,
                      kernel_size=7,
                      stride=2,
                      padding=0), # (-1, 8, 37, 45, 37)
            nn.BatchNorm3d(8), # BatchNormalization
            nn.ReLU(),
            nn.Conv3d(in_channels=8,
                      out_channels=16,
                      kernel_size=5,
                      stride=2,
                      padding=0), # (-1, 16, 17, 21, 17)
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=3,
                      padding=1), # (-1, 32, 6, 7, 6)
            nn.BatchNorm3d(32), # BatchNormalization
            nn.ReLU(),
            nn.Dropout3d(p=0.5),
            Flatten(),
            # This is quite large (8064x128)
            nn.Linear(in_features=6*7*6*32,
                      out_features=128),
            nn.ReLU(),
        )

        # EEGTCNModel based
        if level_size < 0:
            # Automatically calculate the optimal level
            level_size = calc_required_level(kernel_size)
        num_channels=[level_hidden_size] * level_size
        
        self.eeg_net = EEGTCNModelSub(num_channels=num_channels,
                                      kernel_size=kernel_size,
                                      dropout=dropout,
                                      use_residual=use_residual,
                                      duration_type='normal')

        combined_layers = []
        combined_layers += [nn.Linear(in_features=128 + level_hidden_size,
                                      out_features=combined_hidden_size),
                            nn.ReLU()]

        for i in range(combined_layer_size):
            combined_layers += [nn.Linear(in_features=combined_hidden_size,
                                          out_features=combined_hidden_size),
                                nn.ReLU()]
        
        combined_layers += [nn.Dropout(p=0.5),
                            nn.Linear(in_features=combined_hidden_size,
                                      out_features=1)]
        
        self.output = torch.nn.Sequential(*combined_layers)

    def parameters_fc(self):
        return self.output.parameters()

    def parameters_eeg(self):
        return self.eeg_net.parameters()

    def parameters_fmri(self):
        return self.fmri_net.parameters()

    def fix_preloads(self):
        # Fixing on fMRI side
        for i in range(11):
            name = "{}".format(i)
            m = self.fmri_net._modules[name]
            fix_module(m)
            # Up to Flatten()

        # Fixing on EEG side
        # (Although the last linear is also fixed, "forward_for_combined()" does not pass through there and is not affected)
        fix_module(self.eeg_net)
    
    def forward(self, x_fmri, x_eeg):
        h = self.forward_raw(x_fmri, x_eeg)
        return torch.sigmoid(h)

    def forward_raw(self, x_fmri, x_eeg):
        h0 = self.fmri_net(x_fmri)
        # (batch_size, 128)
        # Call "forward_for_combined()" instead of "forward()" to prevent passing the last linear 
        h1 = self.eeg_net.forward_for_combined(x_eeg)
        # (batch_size, 63)
        h = torch.cat([h0, h1], dim=1)
        h = self.output(h)
        return h

    def forward_grad_cam(self, x_fmri, x_eeg):
        self.fmri_grad_extractor = GradExtractor()
        h0 = self.fmri_grad_extractor.forward(self.fmri_net, x_fmri, '8')
        # Relu() after [8]=Conv3d
        # (batch_size, 128)
        
        # Use "for_combined" to prevent passing the last linear
        h1 = self.eeg_net.forward_grad_cam(x_eeg, for_combined=True)
        # (batch_size, 63)
        
        h = torch.cat([h0, h1], dim=1)
        h = self.output(h)
        return h

    def get_cam_gradients(self):
        fmri_grad = self.fmri_grad_extractor.grad
        eeg_grads = []
        for layer in self.eeg_net.layers:
            eeg_grads.append(layer.grad)
        return [fmri_grad, eeg_grads]

    def get_cam_features(self):
        fmri_feature = self.fmri_grad_extractor.feature
        eeg_features = []
        for layer in self.eeg_net.layers:
            eeg_features.append(layer.feature)
        return [fmri_feature, eeg_features]

    def clear_grad_cam(self):
        self.eeg_net.clear_grad_cam()


def get_eeg_model(model_type,
                  parallel,
                  kernel_size,
                  level_size,
                  level_hidden_size,
                  use_residual,
                  duration_type,
                  device):
    
    if model_type == "tcn1":
        print("using tcn model1")
        model = EEGTCNModel(kernel_size=kernel_size,
                            level_size=level_size,
                            level_hidden_size=level_hidden_size,
                            use_residual=use_residual,
                            duration_type=duration_type).to(device)
    else:
        assert False

    if parallel:
        model = torch.nn.DataParallel(model).to(device)

    return model


def get_fmri_model(fmri_ch_size, parallel, device):
    model = FMRIModel(fmri_ch_size).to(device)
    if parallel:
        model = torch.nn.DataParallel(model).to(device)
    return model


def get_combined_model(model_type, fmri_ch_size,
                       kernel_size,
                       level_size,
                       level_hidden_size,
                       use_residual,
                       combined_hidden_size,
                       combined_layer_size,
                       parallel,
                       device):

    if model_type == 'combined_tcn1':
        print("using combined_tcn1")
        model = CombinedTCNModel(
            kernel_size=kernel_size,
            level_size=level_size,
            level_hidden_size=level_hidden_size,
            use_residual=use_residual,
            combined_hidden_size=combined_hidden_size,
            combined_layer_size=combined_layer_size).to(device)
    else:
        assert False
    
    if parallel:
        model = torch.nn.DataParallel(model).to(device)
    return model


if __name__ == '__main__':
    pass
