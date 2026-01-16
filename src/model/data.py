import os
import torch

from PIL import Image
from torch import Tensor
from utils import imgtools
from typing import Callable
from torch.utils.data import Dataset

classes_dict = {
  'PUZZLE': 1,
  'WORDLIST': 2,
  'OTHER': 3
}

classes_ids = {
  1: 'PUZZLE',
  2: 'WORDLIST',
  3: 'OTHER'
}

seperator = ';'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The metadata in the file is structured as follows:
# FILE_NAME;TOP_LEFT_X_SCALE;TOP_LEFT_Y_SCALE;WIDTH_SCALE;HEIGHT_SCALE;CLASS
def load_metadata(meta_filename: str) -> dict[str, list[dict]]:
  output = dict()

  with open(meta_filename) as file:
    for line in file:
      data_list = line.rstrip().split(seperator)
      if len(data_list) != 6:
        raise Exception(f'Line "{line}" format in file does not fit metafile standard')

      image_filename = data_list[0]
      
      is_in_output = image_filename in output
      image_data = output[image_filename] if is_in_output else []
      image_data.append({
        'top_x': float(data_list[1]),
        'top_y': float(data_list[2]),
        'bot_x': float(data_list[3]),
        'bot_y': float(data_list[4]), 
        'class': data_list[5], 
      })

      if not is_in_output:
        output[image_filename] = image_data
  return output
 
class ImageDataset(Dataset):
  def __init__(
    self, 
    image_names: list[str], 
    image_metadata: dict[str, list[dict]], 
    transform: Callable[[Image.Image, list[list[int]]], tuple[Tensor, dict]]
  ):
    self.image_names = image_names
    self.image_metadata = image_metadata
    self.transform = transform

  def __len__(self):
    return (len(self.image_names))

  def __getitem__(self, i: int):
    img_name = self.image_names[i]
    img_meta = self.image_metadata[os.path.basename(img_name)]
    
    img = imgtools.to_pil(img_name)
    img_w, img_h = img.size
    bboxes = [[data['top_x']*img_w, data['top_y']*img_h, data['bot_x']*img_w, data['bot_y']*img_h] for data in img_meta]
    labels = [classes_dict[label['class']] for label in img_meta]

    img, target = self.transform(img, bboxes, labels)     

    return img.to(device), target