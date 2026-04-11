import numpy as np
import unittest

import torch
from model import FMRIModel, EEGTCNModel


class FMRIModelTest(unittest.TestCase):
    def check_tensor_shape(self, t, shape):
        for i in range(len(shape)):
            self.assertEqual(t.shape[i], shape[i])

    def test_model(self):
        model = FMRIModel()

        batch_size = 10
        z_dim = 79
        y_dim = 95
        x_dim = 79
        x = torch.ones(batch_size, 1, z_dim, y_dim, x_dim) # (10, 1, 79, 95, 79)
        out = model(x)
        self.check_tensor_shape(out, (batch_size, 1))


class EEGModelTest(unittest.TestCase):
    def check_tensor_shape(self, t, shape):
        for i in range(len(shape)):
            self.assertEqual(t.shape[i], shape[i])

    def test_model(self):
        def check_model(duration_type, input_length):
            input_channel  = 63  # EEG channel
    
            output_channel = 1
    
            kernel_size = 2
            batch_size = 1

            level_size = -1
            level_hidden_size = 63
            use_residual = True
        
            # EEG data
            x_eeg = torch.randn(batch_size, input_channel, input_length)

            model = EEGTCNModel(kernel_size=kernel_size,
                                level_size=level_size,
                                level_hidden_size=level_hidden_size,
                                use_residual=use_residual,
                                duration_type=duration_type)
        
            out = model(x_eeg)
            self.check_tensor_shape(out, (batch_size, 1))

        check_model('normal', 250)
        check_model('long',   375)
        check_model('short',  125)

    

'''
class CombinedModelTest(unittest.TestCase):
    def check_tensor_shape(self, t, shape):
        for i in range(len(shape)):
            self.assertEqual(t.shape[i], shape[i])
            
    def test_model(self):
        model = CombinedModel()
        
        batch_size = 10
        fmri_z_dim = 79
        fmri_y_dim = 95
        fmri_x_dim = 79
        x_fmri = torch.ones(batch_size, 1, fmri_z_dim, fmri_y_dim, fmri_x_dim)
        
        eeg_x_dim = 63
        eeg_seq_length = 250
        x_eeg = torch.ones(batch_size, eeg_x_dim, eeg_seq_length)
        
        model.zero_grad()
        out = model(x_fmri, x_eeg)
        self.check_tensor_shape(out, (batch_size, 1))
        
        model.eval()
        model.zero_grad()
        out = model.forward_grad_cam(x_fmri, x_eeg)
        
        out = torch.sum(out)
        out.backward()
        cam_grads = model.get_cam_gradients()
        
        self.check_tensor_shape(cam_grads[0], (10, 32, 6, 7, 6))
        self.check_tensor_shape(cam_grads[1], (10, 32, 28))

        model.fix_preloads()
'''



if __name__ == '__main__':
    unittest.main()
