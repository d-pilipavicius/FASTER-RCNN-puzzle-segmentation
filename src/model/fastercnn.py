import time
import torch

from model import data
from utils import metrics
from utils import imgtools
from PIL.Image import Image
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN

CLASSES_COUNT = 4 # + 1 cause of background

CLASS_COLORS = {
  1: 'blue',
  2: 'green',
  3: 'red'
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model_inst(box_score_thresh: float = 0.85) -> FasterRCNN:
  if box_score_thresh < 0 or box_score_thresh > 1:
    raise Exception('Box score threshhold should be within [0; 1]')

  model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', box_score_thresh=box_score_thresh)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, CLASSES_COUNT) 
  model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, CLASSES_COUNT*4)

  for name, param in model.backbone.body.named_parameters():
    if name.startswith(('conv1', 'bn1', 'layer1', 'layer2', 'layer3')):
        param.requires_grad = False

  return model.to(device)

def prep_batch(batch) -> tuple[list, list]:
  images, targets = batch
  
  images = [img.to(device) for img in images]
  
  new_targets = []
  for i in range(len(images)):
    new_targets.append({ 'boxes': targets['boxes'][i].to(device), 'labels': targets['labels'][i].to(device) })

  return images, new_targets

def train_model(
  model: FasterRCNN,
  train_dataloader: DataLoader,
  valid_dataloader: DataLoader,
  optimizer: Optimizer,
  epoch_count: int = 100
):
  train_time = time.time()
  for i in range(epoch_count):
    running_loss = 0.0
    num_batches = 0

    model.train()
    for items in train_dataloader:
      images, targets = prep_batch(items)

      optimizer.zero_grad()

      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict.values())

      losses.backward()
      optimizer.step()

      running_loss += losses.item()
      num_batches += 1

    if (i+1) % 10 == 0:
      avg_loss = running_loss/num_batches
      epochs_train_time = time.time() - train_time
      secs = int(epochs_train_time % 60)
      mins = int(epochs_train_time / 60 % 60)
      hrs = int(epochs_train_time / 3600)
      print(f'Epoch {i+1} train loss: {avg_loss:.4f} | Train time {hrs:02d}:{mins:02d}:{secs:02d}')

      print('-------------TRAIN_DATA-------------')
      test_and_print_metrics(model, train_dataloader)
      print('-------------VALIDATION_DATA-------------')
      test_and_print_metrics(model, valid_dataloader)
      train_time = time.time()

  
def test_and_print_metrics(
  model: FasterRCNN,
  dataloader: DataLoader
):
  model.eval()

  grnd_bboxes = []
  grnd_labels = []
  pred_bboxes = []
  pred_labels = []
  pred_scores = []

  with torch.no_grad():
    for batch in dataloader:
      images, targets = prep_batch(batch)

      preds = model(images)
      
      for target, pred in zip(targets, preds):
        grnd_bboxes.append(target['boxes'])
        grnd_labels.append(target['labels'])
        pred_bboxes.append(pred['boxes'])
        pred_labels.append(pred['labels'])
        pred_scores.append(pred['scores'])
  
  metrics.print_metrics(grnd_bboxes, grnd_labels, pred_bboxes, pred_labels, pred_scores)
  
def run_model(
  model: FasterRCNN,
  image: Tensor
) -> Image:
  model.eval()

  image = image.to(device)

  with torch.no_grad():
    preds = model([image])
    return add_bounding_boxes(image, preds[0]) 
  

def sim_preds(
  bboxes: list[list[int]],
  labels: list[int]
):
  return {
    'boxes': Tensor(bboxes).to(device),
    'labels': Tensor(labels).to(device),
    'scores': Tensor([1 for _ in labels]).to(device)
  }


def add_bounding_boxes(
  image: Tensor,
  prediction: dict
) -> Image:
  labels_ids = prediction['labels'].tolist()

  labels = [data.classes_ids[i] for i in labels_ids]
  colors = [CLASS_COLORS[i] for i in labels_ids]

  box = draw_bounding_boxes(
    image.to(device), 
    boxes=prediction['boxes'],
    labels=labels,
    colors=colors,
    width=4
  )
  
  return imgtools.tensor_to_pil(box.detach())

def load_model(sourcefile: str, threshold: float = 0.85) -> FasterRCNN:

  model = fasterrcnn_resnet50_fpn_v2(box_score_thresh=threshold)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, CLASSES_COUNT) 
  model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, CLASSES_COUNT*4)
  FasterRCNN.load_state_dict(model, torch.load(sourcefile, weights_only=True, map_location=device))

  for name, param in model.backbone.body.named_parameters():
    if name.startswith(('conv1', 'bn1', 'layer1', 'layer2', 'layer3')):
        param.requires_grad = False
  
  return model.to(device)

def export_model(model: FasterRCNN, filename: str):
  torch.save(model.state_dict(), filename)