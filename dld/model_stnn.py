import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from model_utils import Flatten, fix_module


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Spatial_Block(nn.Module):
    def __init__(self, n_inputs, n_channels):
        super(Spatial_Block, self).__init__()

        self.dropout = nn.Dropout(0.1)
        
        self.net_a = weight_norm(nn.Conv1d(n_inputs,  128, 1))
        self.net_b = weight_norm(nn.Linear(n_channels,128))
        self.net_c = weight_norm(nn.Conv1d(128, n_inputs, 1))
        self.net_d = weight_norm(nn.Linear(128, n_channels))
        
        self.net = nn.Sequential(self.net_a,
                                 nn.ReLU(),
                                 self.net_b,
                                 nn.ReLU(),
                                 self.dropout,
                                 self.net_c,
                                 nn.ReLU(),
                                 self.net_d,
                                 nn.ReLU(),
                                 self.dropout)
        self.init_weights()
        
    def init_weights(self):
        self.net_a.weight.data.normal_(0, 0.01)
        self.net_b.weight.data.normal_(0, 0.01)
        self.net_c.weight.data.normal_(0, 0.01)
        self.net_d.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        y = self.net(x)
        return y


class Spatial_Unit(nn.Module):
    def __init__(self, n_inputs, n_channels):
        super(Spatial_Unit, self).__init__()
        
        # 最後に利用するReLU
        self.last_relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.net_a = weight_norm(nn.Conv1d(n_inputs, 128, 1))
        self.net_b = weight_norm(nn.Conv1d(128, 128, 1))
        self.net_c = weight_norm(nn.Linear(n_channels,128))
        self.net_d = weight_norm(nn.Linear(128,128))
        self.pool1 = nn.MaxPool2d(2,2)

        self.net_e = weight_norm(nn.Conv1d(64, 64, 1))
        self.net_f = weight_norm(nn.Conv1d(64, 32, 1))
        self.net_g = weight_norm(nn.Linear(64, 64))
        self.net_h = weight_norm(nn.Linear(64, 32))
        self.pool2 = nn.MaxPool2d(2,2)

        self.net_i = weight_norm(nn.Conv1d(16, 16, 1))
        self.net_j = weight_norm(nn.Conv1d(16, 4, 1))
        self.net_k = weight_norm(nn.Linear(16, 16))
        self.net_l = weight_norm(nn.Linear(16, 4))
        
        self.net = nn.Sequential(self.net_a,
                                 nn.ReLU(),
                                 self.dropout,
                                 self.net_b,
                                 nn.ReLU(),
                                 self.dropout,
                                 self.net_c,
                                 nn.ReLU(),
                                 self.dropout,
                                 self.net_d,
                                 nn.ReLU(),
                                 self.dropout,
                                 self.pool1,                                 
                                 self.net_e,
                                 nn.ReLU(),
                                 self.net_f,
                                 nn.ReLU(),
                                 self.net_g,
                                 nn.ReLU(),
                                 self.net_h,
                                 nn.ReLU(),
                                 self.pool2,
                                 self.net_i,
                                 nn.ReLU(),
                                 self.dropout,
                                 self.net_j,
                                 nn.ReLU(),
                                 self.dropout,
                                 self.net_k,
                                 nn.ReLU(),
                                 self.dropout,
                                 self.net_l,
                                 nn.ReLU(),
                                 self.dropout)
        
        self.fc = weight_norm(nn.Linear(16, 1))
        self.init_weights()
        
    def init_weights(self):
        self.net_a.weight.data.normal_(0, 0.01)
        self.net_b.weight.data.normal_(0, 0.01)
        self.net_c.weight.data.normal_(0, 0.01)
        self.net_d.weight.data.normal_(0, 0.01)
        
        self.net_e.weight.data.normal_(0, 0.01)
        self.net_f.weight.data.normal_(0, 0.01)
        self.net_g.weight.data.normal_(0, 0.01)
        self.net_h.weight.data.normal_(0, 0.01)
        
        self.net_i.weight.data.normal_(0, 0.01)
        self.net_j.weight.data.normal_(0, 0.01)
        self.net_k.weight.data.normal_(0, 0.01)
        self.net_l.weight.data.normal_(0, 0.01)
        self.fc.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        y = self.net(x).squeeze(2)
        y = self.fc(y.view(-1, 16))
        y = self.last_relu(y) # last_reluを利用する
        return y

    
class Temporal_module(nn.Module):
    def __init__(self,
                 n_inputs,
                 input_length,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding):
        
        super(Temporal_module, self).__init__()
        
        self.dropout = nn.Dropout(0.1)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs,
                                           n_outputs,
                                           kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        
        self.temporal_block = nn.Sequential(self.conv1,
                                            self.chomp1,
                                            nn.ReLU(),
                                            self.dropout)
        self.spatial_block = Spatial_Block(n_inputs,
                                           input_length)
                                                
        self.last_relu = nn.ReLU() # 最後に利用するReLU
        self.init_weights()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        y1 = self.temporal_block(x)
        y2 = self.spatial_block(x)
        return self.last_relu(x + y1 + y2) # last_reluを利用する

    
class Temporal_Unit(nn.Module):
    def __init__(self,
                 num_inputs,
                 input_length,
                 num_channels,
                 kernel_size):
        super(Temporal_Unit, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [Temporal_module(in_channels,
                                       input_length,
                                       out_channels,
                                       kernel_size,
                                       stride=1, 
                                       dilation=dilation_size,
                                       padding=(kernel_size-1) * dilation_size)]
    
        self.Temporal_Unit = nn.Sequential(*layers)

    def forward(self, x):
        return self.Temporal_Unit(x)

    
class EEGSTNNModelSub(nn.Module):
    def __init__(self,
                 input_channel,
                 input_length,
                 num_channels,
                 kernel_size):
        super(EEGSTNNModelSub, self).__init__()        
        
        self.Temporal_Unit = Temporal_Unit(input_channel,
                                           input_length,
                                           num_channels,
                                           kernel_size)
        self.Spatial_Unit = Spatial_Unit(input_channel,
                                         input_length)
        self.linear = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        y = self.Temporal_Unit(x)
        y1 = self.linear(y[:, :, -1]) 
        y2 = self.Spatial_Unit(x)
        out = y1 + y2
        return out


class EEGSTNNModel(nn.Module):
    def __init__(self,
                 input_channel=63,
                 input_length=250,
                 num_channels=[63,63,63,63],
                 kernel_size=3):
        
        super(EEGSTNNModel, self).__init__()
        
        self.eeg_net = EEGSTNNModelSub(input_channel=input_channel,
                                       input_length=input_length,
                                       num_channels=num_channels,
                                       kernel_size=kernel_size)
        
    def forward(self, x):
        h = self.eeg_net(x)
        return torch.sigmoid(h)

    def forward_raw(self, x):
        """ logitsまでのところまでの出力 """
        h = self.eeg_net(x)
        return h


class PlainTemporalBlock(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout,
                 use_residual):
        super(PlainTemporalBlock, self).__init__()
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
            # n_inputとn_outputが供に63である場合は、こちら(デフォルト)
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
            # Residual connectionを使う場合(通常)
            if self.downsample is None:
                # n_inputとn_outputが供に63である場合は、こちら(デフォルト)
                res = x
            else:
                res = self.downsample(x)
            return self.last_relu(out + res)
        else:
            # Residual connectionを使わない場合
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


def calc_coverage_length(kernel_size, dilation_sizes):
    """ カバーできる範囲を算出する """
    coverage_length = 1
    
    for dilation_size in reversed(dilation_sizes):
        coverage_length += 2 * dilation_size * (kernel_size-1)

    return coverage_length


def calc_coverage_length_with_level(kernel_size, level_size):
    """ レベルを指定した時のカバレッジ範囲を算出 """
    dilation_sizes = []
    for i in range(level_size):
        dilation_size = 2 ** i
        dilation_sizes.append(dilation_size)        
    coverage_length = calc_coverage_length(kernel_size, dilation_sizes)    
    return coverage_length
    

def calc_required_level(kernel_size):
    """ カーネルサイズに対して、必要なレベル数を算出 """
    max_level = 100
    for i in range(max_level):
        coverage_length = calc_coverage_length_with_level(kernel_size, i)
        if coverage_length >= 250:
            print("Required level_size={} for kernel_size={}".format(i, kernel_size))
            return i
    return None



class EEGTCNModelSub(nn.Module):
    def __init__(self,
                 num_channels,
                 kernel_size,
                 dropout,
                 use_residual):
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
            layers += [PlainTemporalBlock(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=1,
                                          dilation=dilation_size,
                                          padding=(kernel_size-1) * dilation_size,
                                          dropout=dropout,
                                          use_residual=use_residual)]
        
        coverage_length = calc_coverage_length(kernel_size, dilation_sizes)
        print("TCN coverage_length={}".format(coverage_length))
        if coverage_length < 250:
            print("[WARNING] coverage length is shorter than total EEG sequence length: {}",
                  coverage_length)
        
        self.network = nn.Sequential(*layers)
        # (batch, 63, 250)

        # layersを残しておく (forward_grad_cam()用)
        self.layers = layers

    def clear_grad_cam(self):
        for layer in self.layers:
            layer.clear_grad_cam()
        
    def forward(self, x):
        y1 = self.network(x)
        # STNNと同じく最後の出力のみを利用する場合
        h = self.linear(y1[:,:,-1])
        return h

    def forward_for_combined(self, x):
        """ Combined用に最後のLinearを外した出力を出す """
        y1 = self.network(x)
        return y1[:,:,-1]

    def forward_grad_cam(self, x):
        h = x
        for i,layer in enumerate(self.layers):
            h = layer.forward_grad_cam(h)
        h = self.linear(h[:,:,-1])
        return h


class EEGTCNModelSub2(nn.Module):
    def __init__(self,
                 num_channels,
                 kernel_size,
                 dropout,
                 use_residual):
        super(EEGTCNModelSub2, self).__init__()
        
        self.linear1 = nn.Linear(num_channels[-1]*250, 128)
        self.linear2 = nn.Linear(128, 1)
        self.last_relu = nn.ReLU()
        
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
            layers += [PlainTemporalBlock(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=1,
                                          dilation=dilation_size,
                                          padding=(kernel_size-1) * dilation_size,
                                          dropout=dropout,
                                          use_residual=use_residual)]
        
        coverage_length = calc_coverage_length(kernel_size, dilation_sizes)
        print("TCN coverage_length={}".format(coverage_length))
        if coverage_length < 250:
            print("[WARNING] coverage length is shorter than total EEG sequence length: {}",
                  coverage_length)
        
        self.network = nn.Sequential(*layers)
        # (batch, 63, 250)
        
    def forward(self, x):
        y1 = self.network(x)
        
        # 最終層の全time stepを利用する
        batch_size = y1.shape[0]
        # Flatten
        y1 = y1.view(batch_size, -1)
        # FC 2層
        h = self.linear1(y1)
        h = self.last_relu(h)
        h = self.linear2(h)
        return h


class EEGTCNModel(nn.Module):
    """ 最終Levelの最後の時間stepのみを利用するTCN """
    def __init__(self,
                 kernel_size=7,
                 level_size=7,
                 level_hidden_size=63,
                 dropout=0.2,
                 use_residual=True):
        
        super(EEGTCNModel, self).__init__()

        if level_size < 0:
            # 最適なレベルを自動で算出
            level_size = calc_required_level(kernel_size)
        
        num_channels=[level_hidden_size] * level_size
        self.eeg_net = EEGTCNModelSub(num_channels=num_channels,
                                      kernel_size=kernel_size,
                                      dropout=dropout,
                                      use_residual=use_residual)
        
    def forward(self, x):
        h = self.eeg_net(x)
        return torch.sigmoid(h)
    
    def forward_raw(self, x):
        """ logitsまでのところまでの出力 """
        h = self.eeg_net(x)
        return h

    def forward_grad_cam(self, x):
        h = self.eeg_net.forward_grad_cam(x)
        # ここはsigmoidを通さない
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


class EEGTCNModel2(nn.Module):
    """ 最終Levelの最後の全時間stepを利用するTCN """
    def __init__(self,
                 kernel_size=7,
                 level_size=7,
                 level_hidden_size=63,
                 dropout=0.2,
                 use_residual=True):
        
        super(EEGTCNModel2, self).__init__()

        if level_size < 0:
            # 最適なレベルを自動で算出
            level_size = calc_required_level(kernel_size)

        num_channels=[level_hidden_size] * level_size
        self.eeg_net = EEGTCNModelSub2(num_channels=num_channels,
                                       kernel_size=kernel_size,
                                       dropout=dropout,
                                       use_residual=use_residual)
        
    def forward(self, x):
        h = self.eeg_net(x)
        return torch.sigmoid(h)
    
    def forward_raw(self, x):
        """ logitsまでのところまでの出力 """
        h = self.eeg_net(x)
        return h


class CombinedTCNModel(nn.Module):
    """ TCNを利用したCombinedモデル """
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
        
        # FMRIModelベース
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
            # ここが8064x128となっていて結構大きい
            nn.Linear(in_features=6*7*6*32,
                      out_features=128),
            nn.ReLU(),
        )

        # EEGTCNModelベース
        if level_size < 0:
            # 最適なレベルを自動で算出
            level_size = calc_required_level(kernel_size)
        num_channels=[level_hidden_size] * level_size
        
        self.eeg_net = EEGTCNModelSub(num_channels=num_channels,
                                      kernel_size=kernel_size,
                                      dropout=dropout,
                                      use_residual=use_residual)

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
        # fMRI側の固定
        for i in range(11):
            name = "{}".format(i)
            m = self.fmri_net._modules[name]
            fix_module(m)
            # Flatten()のところまで固定

        # EEG側の固定
        # (最後のlinearも固定しているが、forward_for_combined()はそこは通らないので影響がない)
        fix_module(self.eeg_net)
    
    def forward(self, x_fmri, x_eeg):
        h0 = self.fmri_net(x_fmri)
        # (batch_size, 128)
        # forward()ではなく、最後のlinearを通さない用にforward_for_combined()を呼び出す.
        h1 = self.eeg_net.forward_for_combined(x_eeg)
        # (batch_size, 63)
        h = torch.cat([h0, h1], dim=1)
        h = self.output(h)
        return torch.sigmoid(h)


if __name__ == '__main__':
    # Debug code
    
    input_channel  = 63  # EEG channel (original 64)
    input_length   = 250 # Time-series length  (original 128)
    output_channel = 1
    
    kernel_size = 2
    batch_size = 10

    level_size = -1
    level_hidden_size = 63
    use_residual = False
    
    # EEG data
    x_eeg = torch.randn(batch_size, input_channel, input_length)

    model = EEGTCNModel(kernel_size=kernel_size,
                        level_size=level_size,
                        level_hidden_size=level_hidden_size,
                        use_residual=use_residual)

    #out = model(x_eeg)
    #print(out.shape) # (10, 1)

    out = model.forward_grad_cam(x_eeg)
    torch.sum(out).backward()
    
    print(model.get_cam_features()[1].shape)
    print(model.get_cam_gradients()[1].shape)
    model.clear_grad_cam()

    """
    x_fmri = torch.ones(batch_size, 1, 79, 95, 79)
    
    model = CombinedTCNModel(fmri_ch_size=1,
                             kernel_size=kernel_size,
                             level_size=level_size,
                             level_hidden_size=level_hidden_size)
    
    out = model(x_fmri, x_eeg)
    print(out.shape) # (10, 1)

    model.fix_preloads()

    out = model(x_fmri, x_eeg)
    print(out.shape) # (10, 1)
    """
