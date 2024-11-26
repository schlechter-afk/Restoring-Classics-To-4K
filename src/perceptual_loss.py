# perceptual_loss.py
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=None, weights=None):
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG19 model
        vgg = models.vgg19(pretrained=True).features

        # Freeze VGG19 parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg.eval()

        # Specify layers to extract features from
        if layers is None:
            self.layers = {'3': 'relu1_2',   # After 2nd ReLU
                           '8': 'relu2_2',   # After 4th ReLU
                           '17': 'relu3_3',  # After 9th ReLU
                           '26': 'relu4_2'}  # After 13th ReLU
        else:
            self.layers = layers

        # Weights for each layer's loss contribution
        if weights is None:
            self.weights = {'relu1_2': 1.0,
                            'relu2_2': 1.0,
                            'relu3_3': 1.0,
                            'relu4_2': 1.0}
        else:
            self.weights = weights

    def forward(self, input_img, target_img):

        # normalize input and target images
        mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input_img.device)
        std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input_img.device)
        input_img = (input_img - mean_tensor) / std_tensor
        target_img = (target_img - mean_tensor) / std_tensor

        input_features = self.get_features(input_img)
        target_features = self.get_features(target_img)

        perceptual_loss = 0.0
        for layer in self.layers.values():
            input_feat = input_features[layer]
            target_feat = target_features[layer]
            perceptual_loss += self.weights[layer] * nn.functional.mse_loss(input_feat, target_feat)

        return perceptual_loss

    def get_features(self, x):
        features = {}
        for name, layer in self.vgg_layers._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
            if int(name) >= max([int(k) for k in self.layers.keys()]):
                break
        return features
