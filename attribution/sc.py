import torch
import torch.nn.functional as F


class ScoreCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.gradients = {}
        self.activations = {}

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0].to(self.device)

        def forward_hook(module, input, output):
            self.activations['value'] = output.to(self.device)

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        logit = self.model(input)

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit, dim=1)

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        score_saliency_map = torch.zeros((1, 1, h, w), device=self.device)
        activations = self.activations['value']
        b, k, u, v = activations.size()
        activations = activations.to(self.device)

        with torch.no_grad():
          for i in range(k):
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(
                saliency_map, size=(h, w), mode='bilinear', align_corners=False)
              saliency_map = saliency_map.to(self.device)
              if saliency_map.max() == saliency_map.min():
                continue

              norm_saliency_map = (saliency_map - saliency_map.min())
              norm_saliency_map /= saliency_map.max() - saliency_map.min()

              output = self.model(input * norm_saliency_map)
              output = F.softmax(output, dim=1)
              score = output[0][predicted_class]
              score_saliency_map = score_saliency_map.to(self.device)
              score_saliency_map +=  score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min = score_saliency_map.min()
        score_saliency_map_max = score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        ds1 = score_saliency_map - score_saliency_map_min
        ds2 = score_saliency_map_max - score_saliency_map_min
        score_saliency_map = ds1.div(ds2).data
        return score_saliency_map


def sc(model, target_layer, x, class_idx, device):
    """Метод атрибуции Score-CAM.

        Args:
            model (torch.nn.Module): нейронная сеть.
            target_layer (str): целевой слой
            x (torch.tensor): входной вектор для сети.
            class_idx (str): метка класса.

        Returns:
            x (torch.tensor): карта атрибуции для входного вектора.

    """
    score_cam = ScoreCAM(model, target_layer, device)
    return score_cam.forward(x, class_idx)
