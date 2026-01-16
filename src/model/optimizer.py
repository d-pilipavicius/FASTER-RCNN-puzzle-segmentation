from torch.optim import SGD
from torchvision.models.detection import FasterRCNN

def get_custom_fasterrcnn_SGD(model: FasterRCNN) -> SGD:
  backbone_params = []
  rpn_params = []
  roi_params = []

  for name, params in model.named_parameters():
    if not params.requires_grad:
      continue

    if 'backbone.body.layer4' in name:
      backbone_params.append(params)
    elif 'rpn' in name:
      rpn_params.append(params)
    elif 'roi_heads' in name:
      roi_params.append(params)

  return SGD(
    [
      { 'params': backbone_params, 'lr': 1e-5 },
      { 'params': rpn_params, 'lr': 5e-4},
      { 'params': roi_params, 'lr': 5e-4}
    ],
    momentum=0.9,
    weight_decay=1e-4
  )