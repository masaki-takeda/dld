import torch
import torch.nn as nn


class ReLUHook():
    """
    # 1) Save the output values of forward pass
    # 2) Clipping to 0 when the gradient is less than 0
    """
    def __init__(self):
        pass
        
    def backward_hook(self, module, grad_in, grad_out):
        # If there is a negative gradient, change it to zero
        self.forward_output[self.forward_output > 0] = 1
        modified_grad_out = self.forward_output * torch.clamp(grad_in[0], min=0.0)
        return (modified_grad_out,)

    def forward_hook(self, module, ten_in, ten_out):
        # Store results of forward pass
        self.forward_output = ten_out

    def clear(self):
        # Clear reference to "forward_output"
        self.forward_output = None

    
class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.relu_hooks = []
        self.hook_handles = []        
        
        self.model.eval()
        self.apply_relu_hooks(self.model)
        
    def apply_relu_hooks(self, module):
        for child in module.children():
            if len(list(child.children())) == 0 and isinstance(child, nn.ReLU):
                relu_hook = ReLUHook()
                handle_backward = child.register_backward_hook(relu_hook.backward_hook)
                handle_forward = child.register_forward_hook(relu_hook.forward_hook)
                self.relu_hooks.append(relu_hook)
                
                self.hook_handles.append(handle_backward)
                self.hook_handles.append(handle_forward)
            else:
                self.apply_relu_hooks(child)
    
    def generate_gradients(self, input_image, label, device):
        # Execute forward pass
        input_image = input_image.detach().clone()
        input_image.requires_grad = True

        # Get output up to the point of logits
        model_output = self.model.forward_raw(input_image)
        predicted_prob = torch.sigmoid(model_output)[0][0].cpu().detach().numpy()
        
        # Zero initialize gradient
        self.model.zero_grad()
        
        # Set targets
        output = torch.FloatTensor(1, model_output.size()[-1]).to(device).zero_()
        # (1,1)
        
        if label == 1:
            output[0][0] = 1
        else:
            output[0][0] = -1
        
        # Execute backward pass
        model_output.backward(gradient=output)
        
        # Convert torch tensor to numpy array
        # Get in [0] to eliminate the dimension of batch
        gradients_array = input_image.grad.cpu().numpy()[0]
        
        return gradients_array, predicted_prob

    def generate_gradients_for_combined(self, input_image_f, input_image_e, label, device):
        # Execute forward pass
        input_image_f = input_image_f.detach().clone()
        input_image_f.requires_grad = True

        input_image_e = input_image_e.detach().clone()
        input_image_e.requires_grad = True

        # Get output up to the point of logits
        model_output = self.model.forward_raw(input_image_f, input_image_e)
        predicted_prob = torch.sigmoid(model_output)[0][0].cpu().detach().numpy()
        
        # Zero initialize gradient
        self.model.zero_grad()
        
        # Set targets
        output = torch.FloatTensor(1, model_output.size()[-1]).to(device).zero_()
        # (1,1)
        
        if label == 1:
            output[0][0] = 1
        else:
            output[0][0] = -1
        
        # Execute backward pass
        model_output.backward(gradient=output)
        
        # Convert torch tensor to numpy array
        # Get in [0] to eliminate the dimension of batch
        gradients_array_f = input_image_f.grad.cpu().numpy()[0]
        gradients_array_e = input_image_e.grad.cpu().numpy()[0]
        
        return gradients_array_f, gradients_array_e, predicted_prob

    def clear(self):
        # Clear hook
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        for relu_hook in self.relu_hooks:
            relu_hook.clear()
        self.relu_hooks = []

    
if __name__ == '__main__':
    # Debug code
    from model_stnn import EEGSTNNModel
    model = EEGSTNNModel()
    gbp = GuidedBackprop(model)
    
    input_channel  = 63  # EEG channel (original 64)
    input_length   = 250 # Time-series length  (original 128)
    x = torch.randn(1, input_channel, input_length)

    gradient0, predicted_prob0 = gbp.generate_gradients(x, 0, "cpu")
    # (63, 250)
    print(gradient0.shape)
    print(predicted_prob0)    
    
    gradient1, predicted_prob1 = gbp.generate_gradients(x, 1, "cpu")
    # (63, 250)
    print(gradient1.shape)
    print(predicted_prob1)

    clear()
