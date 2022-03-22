import torch
import torch.nn as nn
import torch.nn.functional as F

from model_stnn import EEGSTNNModel, EEGTCNModel, EEGTCNModel2, CombinedTCNModel
from model_utils import GradExtractor, Flatten, fix_module

    
class EEGModel(nn.Module):
    """
    EEG Model
    """
    def __init__(self):
        super(EEGModel, self).__init__()
        self.eeg_net = torch.nn.Sequential(
            nn.Conv1d(in_channels=63,
                      out_channels=10,
                      kernel_size=5,
                      stride=1,
                      padding=2), # (10,10,250) #(10, 10, 375)
            nn.ReLU(),
            nn.Conv1d(in_channels=10,
                      out_channels=8,
                      kernel_size=5,
                      stride=3,
                      padding=2), # (10,8,84) #(10, 8, 125)
            nn.ReLU(),
            nn.Conv1d(in_channels=8,
                      out_channels=32,
                      kernel_size=3,
                      stride=3,
                      padding=1),# (10,32,28) #(10, 32, 42)
            nn.ReLU(),
            nn.Conv1d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1, # If stride=2, it will not work correctly
                      padding=1), # (10,32,28) #(10, 32, 42)
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32*28,
                      out_features=128),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=128,
                      out_features=1)
        )
        
    def forward(self, x):
        h = self.eeg_net(x)
        h = torch.sigmoid(h)
        return h

    def forward_grad_cam(self, x, cam_level=0):
        self.grad_extractor = GradExtractor()

        if cam_level == 0:
            e_layer = '7'
            # Relu() after [7]=Conv1d
        elif cam_level == 1:
            e_layer = '5'
        elif cam_level == 2:
            e_layer = '3'
        elif cam_level == 3:
            e_layer = '1'
        elif cam_level == -1:
            e_layer = None
        
        h = self.grad_extractor.forward(self.eeg_net, x, e_layer)
        return h

    def get_cam_gradients(self):
        return self.grad_extractor.grad

    def get_cam_features(self):
        return self.grad_extractor.feature


class EEGFilterModel(nn.Module):
    """
    EEG Filter Model (tentative)
    A model with 2D convolution: the input data is considered as an image of width:250, height:63, and color:5ch
    (c)
    Since the 63-channels of EEG does not represent the spatial location information in the order, it is necessary to examine whether the data is appropriate or not
    The kernel size (vertical and horizontal) and stride size should be considered accordingly
    """
    def __init__(self):
        super(EEGFilterModel, self).__init__()
        self.eeg_net = torch.nn.Sequential(
            nn.Conv2d(in_channels=5,
                      out_channels=10,
                      kernel_size=(5,5),
                      stride=(1,1),
                      padding=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=10,
                      out_channels=8,
                      kernel_size=(5,5),
                      stride=(3,3),
                      padding=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(3,3),
                      padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=(1,1)),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32*196,
                      out_features=128),
            nn.ReLU(), # Fixed a missing ReLU here
            nn.Linear(in_features=128,
                      out_features=1)
        )
        
    def forward(self, x):
        # x=(10, 5, 63, 250)
        h = self.eeg_net(x)
        h = torch.sigmoid(h)
        return h


class EEGFilterModel2(nn.Module):
    """
    EEG Filter Model2
    A model with 2D convolution: the input data is considered as an image of width:250, height:5, and color:63ch
    (This model could be more appropriate because the order of the 5 channels obtained by filter is meaningful)
    The kernel size (vertical and horizontal) and stride size should be considered
    """
    def __init__(self):
        super(EEGFilterModel2, self).__init__()
        self.eeg_net = torch.nn.Sequential(
            nn.Conv2d(in_channels=63,
                      out_channels=10,
                      kernel_size=(3,5),
                      stride=(1,1),
                      padding=(1,2)), # (10, 63, 5, 250)
            nn.ReLU(),
            nn.Conv2d(in_channels=10,
                      out_channels=8,
                      kernel_size=(3,5),
                      stride=(1,3),
                      padding=(1,2)), # (10, 8, 5, 84)
            nn.ReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(1,3),
                      padding=(1,1)), # (10, 8, 5, 28)
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(2,1),
                      padding=(1,1)), # (10, 32, 3, 28)
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32*3*28,
                      out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,
                      out_features=1)
        )
        
    def forward(self, x):
        # x=(10, 5, 63, 250)
        # Swap the locations of 5ch and 63ch
        x = torch.transpose(x, 1, 2)
        # x=(10, 63, 5, 250) -> the input data is considered as an image of width:250, height:5, and 63ch, batch_size=10
        h = self.eeg_net(x)
        h = torch.sigmoid(h)
        return h

    def forward_grad_cam(self, x, cam_level=0):
        # x=(10, 5, 63, 250)
        # Swap the locations of 5ch and 63ch
        x = torch.transpose(x, 1, 2)
        
        self.grad_extractor = GradExtractor()

        if cam_level == 0:
            e_layer = '7'
            # Relu() after [7]=Conv1d
        elif cam_level == 1:
            e_layer = '5'
        elif cam_level == 2:
            e_layer = '3'
        elif cam_level == 3:
            e_layer = '1'
        elif cam_level == -1:
            e_layer = None
        
        h = self.grad_extractor.forward(self.eeg_net, x, e_layer)
        return h

    def get_cam_gradients(self):
        return self.grad_extractor.grad

    def get_cam_features(self):
        return self.grad_extractor.feature


class EEGFilterModel3(nn.Module):
    """
    EEG Filter Model3
    A model that first performs FC and then 1d conv
    (Parameters and others are specified as appropriate values for now)
    """
    def __init__(self):
        super(EEGFilterModel3, self).__init__()
        self.pre_fc = nn.Linear(in_features=5*63,
                                out_features=32)
        
        self.eeg_net = torch.nn.Sequential(
            nn.Conv1d(in_channels=32,
                      out_channels=10,
                      kernel_size=5,
                      stride=1,
                      padding=2), # (10,10,250)
            nn.ReLU(),
            nn.Conv1d(in_channels=10,
                      out_channels=8,
                      kernel_size=5,
                      stride=3,
                      padding=2), # (10,8,84)
            nn.ReLU(),
            nn.Conv1d(in_channels=8,
                      out_channels=32,
                      kernel_size=3,
                      stride=3,
                      padding=1),# (10,32,28)
            nn.ReLU(),
            nn.Conv1d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1, # If stride=2, it will not work correctly
                      padding=1), # (10,32,28) #(10, 32, 42)
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32*28,
                      out_features=128),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=128,
                      out_features=1)
            )
        
    def forward(self, x):
        # x=(10, 5, 63, 250)
        x = torch.transpose(x, 1, 3)
        # x=(10, 250, 63, 5)
        x = torch.reshape(x, (-1, 250, 63*5))
        x2 = self.pre_fc(x)
        # x2=(10*250, 32)
        x2 = x2.view(-1, 250, 32)
        # x2=(10, 250, 32)
        x2 = torch.transpose(x2, 1, 2)
        # x2=(10, 32, 250)
        h = self.eeg_net(x2)
        h = torch.sigmoid(h)
        return h


class EEGModel2(nn.Module):
    """
    EEG Model2
    """
    def __init__(self):
        super(EEGModel2, self).__init__()
        self.eeg_net = torch.nn.Sequential(
            nn.Conv1d(in_channels=63,
                      out_channels=10,
                      kernel_size=5,
                      stride=3, # Set the first Stride larger
                      padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=10,
                      out_channels=8,
                      kernel_size=5,
                      stride=3,
                      padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8,
                      out_channels=32,
                      kernel_size=3,
                      stride=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32*10,
                      out_features=128),
            Flatten(),
            nn.Linear(in_features=128,
                      out_features=1)
        )
        
    def forward(self, x):
        h = self.eeg_net(x)
        h = torch.sigmoid(h)
        return h

    def forward_grad_cam(self, x):
        self.grad_extractor = GradExtractor()
        h = self.grad_extractor.forward(self.eeg_net, x, '7')
        # Relu() after [7]=Conv1d
        return h # Without sigmoid() here

    def get_cam_gradients(self):
        return self.grad_extractor.grad

    def get_cam_features(self):
        return self.grad_extractor.feature


class EEGRNNModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, bidirectional=True):
        super(EEGRNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_size=63,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, input, state):
        # Ignore "state"
        output, _ = self.lstm(input, state)
        #output = self.fc(output)
        output = self.fc(output[:,-1,:]) # For using only the last output
        output = torch.sigmoid(output)
        return output
    
    def init_state(self, batch_size, device):
        if self.bidirectional:
            h = torch.zeros(self.num_layers*2, batch_size, self.hidden_size)
            c = torch.zeros(self.num_layers*2, batch_size, self.hidden_size)
        else:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h = h.to(device)
        c = c.to(device)
        return (h, c)



class EEGConvRNNModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, bidirectional=True):
        super(EEGConvRNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.conv_net = torch.nn.Sequential(
            nn.Conv1d(in_channels=63,
                      out_channels=10,
                      kernel_size=5,
                      stride=3, # Set the first Stride larger
                      padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=10,
                      #out_channels=8,
                      out_channels=32,
                      kernel_size=5,
                      stride=3,
                      padding=2),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, input, state):
        # Since "input" belongs to "conv" (batch, dim, seq)
        h = self.conv_net(input)
        h = torch.transpose(h, 1, 2)
        # (batch, seq, dim)
        output, _ = self.lstm(h, state)
        output = self.fc(output[:,-1,:]) # For using only the last output
        output = torch.sigmoid(output)
        return output
    
    def init_state(self, batch_size, device):
        if self.bidirectional:
            h = torch.zeros(self.num_layers*2, batch_size, self.hidden_size)
            c = torch.zeros(self.num_layers*2, batch_size, self.hidden_size)
        else:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h = h.to(device)
        c = c.to(device)
        return (h, c)


class EEGFtModel(nn.Module):
    def __init__(self):
        super(EEGFtModel, self).__init__()
        self.eeg_net = torch.nn.Sequential( 
            nn.Conv2d(in_channels=63,
                      out_channels=10,
                      kernel_size=(3,5),
                      stride=(1,1),
                      padding=(1,2)), # (10, 10, 17, 125)
            nn.ReLU(),
            nn.Conv2d(in_channels=10,
                      out_channels=8,
                      kernel_size=(3,5),
                      stride=(1,3),
                      padding=(1,2)), # (10, 8, 17, 42)
            nn.ReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(2,3),
                      padding=(1,1)), # (10, 8, 9, 14)
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(2,1),
                      padding=(1,1)), # (10, 32, 5, 14)
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32*5*14,
                      out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,
                      out_features=1)
        )
        
    def forward(self, x):
        # x=(10, 17, 63, 125)
        # Swap the locations of 5ch and 63ch
        x = torch.transpose(x, 1, 2)
        # x=(10, 63, 17, 125)
        h = self.eeg_net(x)
        h = torch.sigmoid(h)
        return h



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
            # This data is quite large (8064x128), (OHBM's input data is smaller (2304x128))
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


class CombinedModel(nn.Module):
    def __init__(self, fmri_ch_size=1):
        super(CombinedModel, self).__init__()
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
            # This data is quite large (8064x128), (OHBM's input data is smaller (2304x128))
            nn.Linear(in_features=6*7*6*32,
                      out_features=128),
            nn.ReLU(),
        )

        # EEGModel based
        self.eeg_net = torch.nn.Sequential(
            nn.Conv1d(in_channels=63,
                      out_channels=10,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=10,
                      out_channels=8,
                      kernel_size=5,
                      stride=3,
                      padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8,
                      out_channels=32,
                      kernel_size=3,
                      stride=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32*28,
                      out_features=128),
            nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            nn.Linear(in_features=256,
                      out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )

    def fix_preloads(self):
        for i in range(11):
            name = "{}".format(i)
            m = self.fmri_net._modules[name]
            fix_module(m)
            # Up to Flatten()

        for i in range(9):
            name = "{}".format(i)
            m = self.eeg_net._modules[name]
            fix_module(m)
            # Up to Flatten()
    
    def forward(self, x_fmri, x_eeg):
        h0 = self.fmri_net(x_fmri)
        h1 = self.eeg_net(x_eeg)
        h = torch.cat([h0, h1], dim=1)
        h = self.output(h)
        return torch.sigmoid(h)

    def forward_grad_cam(self, x_fmri, x_eeg, cam_level=0):
        self.fmri_grad_extractor = GradExtractor()
        self.eeg_grad_extractor  = GradExtractor()

        if cam_level == 0:
            f_layer = '8'
            e_layer = '7'
        elif cam_level == 1:
            f_layer = '5'
            e_layer = '5'
        elif cam_level == 2:
            f_layer = '3'
            e_layer = '3'
        
        h0 = self.fmri_grad_extractor.forward(self.fmri_net, x_fmri, f_layer)
        # Relu() after [8]=BatchNorm3d
        h1 = self.eeg_grad_extractor.forward(self.eeg_net, x_eeg, e_layer)
        # Relu() after [7]=Conv1d
        h = torch.cat([h0, h1], dim=1)
        h = self.output(h)
        # Without sigmoid() here
        return h

    def get_cam_gradients(self):
        return (self.fmri_grad_extractor.grad, self.eeg_grad_extractor.grad)

    def get_cam_features(self):
        return (self.fmri_grad_extractor.feature, self.eeg_grad_extractor.feature)


class CombinedFilterModel(nn.Module):
    def __init__(self, fmri_ch_size=1):
        super(CombinedFilterModel, self).__init__()
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
            # This data is quite large (8064x128), (OHBM's input data is smaller (2304x128))
            nn.Linear(in_features=6*7*6*32,
                      out_features=128),
            nn.ReLU(),
        )

        # EEGFilterModel2 based
        self.eeg_net = torch.nn.Sequential(
            nn.Conv2d(in_channels=63,
                      out_channels=10,
                      kernel_size=(3,5),
                      stride=(1,1),
                      padding=(1,2)), # (10, 63, 5, 250)
            nn.ReLU(),
            nn.Conv2d(in_channels=10,
                      out_channels=8,
                      kernel_size=(3,5),
                      stride=(1,3),
                      padding=(1,2)), # (10, 8, 5, 84)
            nn.ReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(1,3),
                      padding=(1,1)), # (10, 8, 5, 28)
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(2,1),
                      padding=(1,1)), # (10, 32, 3, 28)
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32*3*28,
                      out_features=128),
            nn.ReLU(),
        )
        
        self.output = torch.nn.Sequential(
            nn.Linear(in_features=256,
                      out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )

    def fix_preloads(self):
        for i in range(11):
            name = "{}".format(i)
            m = self.fmri_net._modules[name]
            fix_module(m)
            # Up to Flatten()

        for i in range(9):
            name = "{}".format(i)
            m = self.eeg_net._modules[name]
            fix_module(m)
            # Up to Flatten()
    
    def forward(self, x_fmri, x_eeg):
        h0 = self.fmri_net(x_fmri)
        
        # x_eeg=(10, 5, 63, 250)
        # Swap the locations of 5ch and 63ch
        x_eeg = torch.transpose(x_eeg, 1, 2)
        # x_eeg=(10, 63, 5, 250) -> the input data is considered as an image of width:250, height:5, and 63ch, batch_size=10
        h1 = self.eeg_net(x_eeg)
        
        h = torch.cat([h0, h1], dim=1)
        h = self.output(h)
        return torch.sigmoid(h)

    def forward_grad_cam(self, x_fmri, x_eeg, cam_level=0):
        self.fmri_grad_extractor = GradExtractor()
        self.eeg_grad_extractor  = GradExtractor()

        if cam_level == 0:
            f_layer = '8'
            e_layer = '7'
        elif cam_level == 1:
            f_layer = '5'
            e_layer = '5'
        elif cam_level == 2:
            f_layer = '3'
            e_layer = '3'

        x_eeg = torch.transpose(x_eeg, 1, 2)
        
        h0 = self.fmri_grad_extractor.forward(self.fmri_net, x_fmri, f_layer)
        # Relu() after [8]=BatchNorm3d
        h1 = self.eeg_grad_extractor.forward(self.eeg_net, x_eeg, e_layer)
        # Relu() after [7]=Conv1d
        h = torch.cat([h0, h1], dim=1)
        h = self.output(h)
        # Without sigmoid() here
        return h

    def get_cam_gradients(self):
        return (self.fmri_grad_extractor.grad, self.eeg_grad_extractor.grad)

    def get_cam_features(self):
        return (self.fmri_grad_extractor.feature, self.eeg_grad_extractor.feature)


def get_eeg_model(model_type, parallel, kernel_size, level_size, level_hidden_size, use_residual, device):
    if model_type == "model1":
        print("using model1")
        model = EEGModel().to(device)
    elif model_type == "model2":
        print("using model2")
        model = EEGModel2().to(device)
    elif model_type == "rnn1":
        print("using rnn1")
        model = EEGRNNModel().to(device)
        transpose_input = True
        use_state = True
    elif model_type == "convrnn1":
        print("using convrnn1")
        model = EEGConvRNNModel().to(device)
        transpose_input = False
        use_state = True
    elif model_type == "filter1":
        print("using filter model1")
        model = EEGFilterModel().to(device)
    elif model_type == "filter2":
        print("using filter model2")
        model = EEGFilterModel2().to(device)
    elif model_type == "filter3":
        print("using filter model3")
        model = EEGFilterModel3().to(device)
    elif model_type == "ft1":
        print("using ft model1")
        model = EEGFtModel().to(device)
    elif model_type == "stnn1":
        print("using stnn model1")
        model = EEGSTNNModel(kernel_size=kernel_size).to(device)
    elif model_type == "tcn1":
        print("using tcn model1")
        model = EEGTCNModel(kernel_size=kernel_size,
                            level_size=level_size,
                            level_hidden_size=level_hidden_size,
                            use_residual=use_residual).to(device)
    elif model_type == "tcn2":
        print("using tcn model2")
        model = EEGTCNModel2(kernel_size=kernel_size,
                             level_size=level_size,
                             level_hidden_size=level_hidden_size,
                             use_residual=use_residual).to(device)

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
    if model_type == "combined1":
        print("using combined1")
        model = CombinedModel(fmri_ch_size).to(device)
    elif model_type == "combined_filter1":
        print("using combined_filter1")
        model = CombinedFilterModel().to(device)
    elif model_type == "combined_tcn1":
        print("using combined_tcn1")
        model = CombinedTCNModel(
            kernel_size=kernel_size,
            level_size=level_size,
            level_hidden_size=level_hidden_size,
            use_residual=use_residual,
            combined_hidden_size=combined_hidden_size,
            combined_layer_size=combined_layer_size).to(device)
    else:
        print("using default combined_tcn1")
        model = CombinedTCNModel(
            kernel_size=kernel_size,
            level_size=level_size,
            level_hidden_size=level_hidden_size,
            use_residual=use_residual,
            combined_hidden_size=combined_hidden_size,
            combined_layer_size=combined_layer_size).to(device)
    if parallel:
        model = torch.nn.DataParallel(model).to(device)
    return model


if __name__ == '__main__':
    # Example code for debugging
    model = EEGModel()

    batch_size = 10
    #x = torch.ones(batch_size, 5, 63, 250)
    #x = torch.ones(batch_size, 17, 63, 125)
    x = torch.ones(batch_size, 63, 250)
    out = model(x)
    print(out.shape) # (10, 1)
