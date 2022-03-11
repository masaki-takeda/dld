import numpy as np
import unittest

import torch
from model import EEGModel, EEGRNNModel, EEGConvRNNModel, FMRIModel, CombinedModel


class EEGModelTest(unittest.TestCase):
    def check_tensor_shape(self, t, shape):
        for i in range(len(shape)):
            self.assertEqual(t.shape[i], shape[i])

    def test_model(self):
        model = EEGModel()

        batch_size = 10
        x_dim = 63
        seq_length = 250

        x = torch.ones(batch_size, x_dim, seq_length)
        out = model(x)
        self.check_tensor_shape(out, (batch_size, 1))
        
        
class EEGRNNModelTest(unittest.TestCase):
    def check_tensor_shape(self, t, shape):
        for i in range(len(shape)):
            self.assertEqual(t.shape[i], shape[i])

    def test_init(self):
        model = EEGRNNModel()

        batch_size = 10
        x_dim = 63
        seq_length = 250

        x = torch.ones(batch_size, seq_length, x_dim) # RNNはここがseq, x_dim逆になっている
        device = "cpu"
        state = model.init_state(batch_size, device)
        output = model(x, state)
        self.check_tensor_shape(output, (batch_size, 1))


class EEGConvRNNModelTest(unittest.TestCase):
    def check_tensor_shape(self, t, shape):
        for i in range(len(shape)):
            self.assertEqual(t.shape[i], shape[i])

    def test_init(self):
        model = EEGConvRNNModel()

        batch_size = 10
        x_dim = 63
        seq_length = 250

        x = torch.ones(batch_size, x_dim, seq_length) # Conv系のinput
        device = "cpu"
        state = model.init_state(batch_size, device)
        output = model(x, state)
        self.check_tensor_shape(output, (batch_size, 1))

        
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


if __name__ == '__main__':
    unittest.main()
