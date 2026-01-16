import torch
import numpy as NP
import albumentations as A

from PIL import Image
from torch import Tensor, from_numpy
from torchvision.transforms.functional import to_pil_image, to_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_transforms(
  image: Image.Image | NP.ndarray | Tensor, 
  bboxes: list[list[int]], 
  labels: list[int]
) -> tuple[Tensor, dict[str, Tensor]]:
  transf_fun = A.Compose([
    A.Rotate(limit=(-5, 5)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

  return transforms(image=image, bboxes=bboxes, labels=labels, transf_fun=transf_fun)

def test_transforms(
  image: Image.Image | NP.ndarray | Tensor, 
  bboxes: list[list[int]], 
  labels: list[int]
) -> tuple[Tensor, dict[str, Tensor]]:
  transf_fun = A.Compose([], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

  return transforms(image=image, bboxes=bboxes, labels=labels, transf_fun=transf_fun)

def transforms(
  image: Image.Image | NP.ndarray | Tensor, 
  bboxes: list[list[int]], 
  labels: list[int], 
  transf_fun: A.Compose
) -> tuple[Tensor, dict[str, Tensor]]:
  if isinstance(image, Image.Image):
    image = pil_to_numpy(image)
  elif isinstance(image, Tensor):
    image = tensor_to_numpy(image)

  prepped_bboxes = NP.array(bboxes)

  augmented = transf_fun(image=image, bboxes=prepped_bboxes, class_labels=labels)
  image = numpy_to_tensor(augmented['image']).to(device)
  image = image.float() / 255.0
  
  target = {
    'boxes': torch.from_numpy(augmented['bboxes']).to(device),
    'labels': torch.tensor(augmented['class_labels'], dtype=torch.int64).to(device) 
  }

  return image.to(device), target

def to_pil(image_filename: str) -> Image.Image:
  return Image.open(image_filename).convert('RGB')

def pil_to_numpy(pil_image: Image.Image) -> NP.ndarray:
  return NP.array(pil_image)

def numpy_to_pil(numpy_image: NP.ndarray) -> Image.Image:
  return Image.fromarray(numpy_image).convert('RGB')

def pil_to_tensor(pil_image: Image.Image) -> Tensor:
  return to_tensor(pil_image).to(device)

def tensor_to_pil(tensor_image: Tensor) -> Image.Image:
  return to_pil_image(tensor_image).convert('RGB')

def tensor_to_numpy(tensor_image: Tensor) -> NP.ndarray:
  return tensor_image.detach().cpu().numpy()

def numpy_to_tensor(numpy_image: NP.ndarray) -> Tensor:
  return from_numpy(numpy_image).permute((2, 0, 1)).to(device)