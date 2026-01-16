import os
import sys
import json
import warnings
import random

from PIL import Image
from torch import cuda
from utils import imgtools
from model import fastercnn, optimizer
from torch.utils.data import DataLoader
from model.data import ImageDataset, load_metadata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
  print(f'Using GPU: {cuda.is_available()}')
  warnings.filterwarnings("ignore", category=UserWarning, message="Got processor for bboxes")
  compute_args()

def compute_args():
  argc = len(sys.argv)
  if argc > 5 or argc < 3:
    raise Exception(f'Expected to start using "python {sys.argv[0]} train/test/run SETUP_NAME [RUN_IMG_NAME] [RUN_OUTPUT_IMG_NAME]"')

  run_type = sys.argv[1].lower()
  if run_type not in ['test', 'train', 'run', 'bboxes_test']:
    raise Exception(f'Run type "{run_type}" not supported')

  if run_type == 'run':
    if argc < 4:
      raise Exception('Expected input image name')

    input = sys.argv[3]
    output = sys.argv[4] if argc == 5 else os.path.join(BASE_DIR, 'output.png')
  elif argc != 3:
    raise Exception(f'"{run_type}" should be run using "python {sys.argv[0]} {run_type} SETUP_NAME"')  

  setup = extract_setup(sys.argv[2])

  if run_type == 'bboxes_test':
    if not os.path.exists('./bboxes'):
      os.mkdir('./bboxes')
    draw_bboxes(setup)
  elif run_type == 'train':
    run_train(setup)
  elif run_type == 'test':
    run_test(setup)
  else:
    run_model(setup, input, output)

def draw_bboxes(setup: dict):
  test_imgs_filename = setup['setup']['test']
  test_imgs_src = os.path.join(setup['ground_dir'], test_imgs_filename)
  
  with open(test_imgs_src) as f:
    test_img_list = [os.path.join(setup['img_dir'], line.strip()) for line in f]

  img_meta = load_metadata(setup['img_metafile'])
  test_ds = ImageDataset(test_img_list, img_meta, imgtools.test_transforms)

  for image, preds in test_ds:
    img = fastercnn.add_bounding_boxes(image, preds) 
    img.save(f'./bboxes/image_{random.randint(0, 99999999)}.jpg')

def run_train(setup: dict):
  train_imgs_filename = setup['setup']['train']
  valid_imgs_filename = setup['setup']['valid']
  train_imgs_src = os.path.join(setup['ground_dir'], train_imgs_filename)
  valid_imgs_src = os.path.join(setup['ground_dir'], valid_imgs_filename)
  
  with open(train_imgs_src) as f:
    train_img_list = [os.path.join(setup['img_dir'], line.strip()) for line in f]
  with open(valid_imgs_src) as f:
    valid_img_list = [os.path.join(setup['img_dir'], line.strip()) for line in f]

  img_meta = load_metadata(setup['img_metafile'])

  train_ds = ImageDataset(train_img_list, img_meta, imgtools.train_transforms)
  valid_ds = ImageDataset(valid_img_list, img_meta, imgtools.test_transforms)
  train_dl = DataLoader(train_ds, shuffle=True, num_workers=setup['use_workers'])
  valid_dl = DataLoader(valid_ds, shuffle=False, num_workers=setup['use_workers'])
  
  model_inst = fastercnn.get_model_inst(box_score_thresh=setup['box_thresh'])

  print('Training started')
  fastercnn.train_model(
    model=model_inst,
    train_dataloader=train_dl,
    valid_dataloader=valid_dl,
    optimizer=optimizer.get_custom_fasterrcnn_SGD(model_inst),
    epoch_count=setup['epoch_count']
  )

  model_name = setup['setup']['model']
  model_src = os.path.join(BASE_DIR, model_name)
  print(f'Saving model to "{model_name}"')
  fastercnn.export_model(model_inst, model_src)

def run_test(setup: dict):
  test_imgs_filename = setup['setup']['test']
  test_imgs_src = os.path.join(setup['ground_dir'], test_imgs_filename)

  with open(test_imgs_src) as f:
    test_img_list = [os.path.join(setup['img_dir'], line.strip()) for line in f]

  img_meta = load_metadata(setup['img_metafile'])

  test_ds = ImageDataset(test_img_list, img_meta, imgtools.test_transforms)
  test_dl = DataLoader(test_ds, shuffle=False, num_workers=setup['use_workers'])

  model_name = setup['setup']['model']
  model_src = os.path.join(BASE_DIR, model_name)
  print(f'Loading model from "{model_name}"')
  model = fastercnn.load_model(model_src, threshold=setup['box_thresh'])

  fastercnn.test_and_print_metrics(model, test_dl)

def run_model(
  setup: dict, 
  input_img: str, 
  output_name: str
):
  model_name = setup['setup']['model']
  model_path = os.path.join(BASE_DIR, model_name)
  print(f'Loading model from "{model_name}"')
  model = fastercnn.load_model(model_path, threshold=setup['box_thresh'])
  
  image = Image.open(input_img)
  t_image, _ = imgtools.test_transforms(image, [], [])
  image.close()

  print('Running model')
  b_image = fastercnn.run_model(model, t_image)

  b_image.save(output_name)
  print(f'Image exported to "{output_name}"')

def extract_setup(setup_name: str) -> dict: 
  data = read_setup()
  setup = next((item for item in data['setups'] if item['name'] == setup_name), None)
  
  box_thresh = data['box_thresh']
  if setup == None:
    raise Exception(f'Setup under name {sys.argv[2]} not found')
  if box_thresh < 0 or box_thresh > 1:
    raise Exception('Box threshold must be between [0;1]')

  ground_dir = os.path.join(BASE_DIR, data['ground_dir'])
  instance_setup = {
    'img_dir': os.path.join(BASE_DIR, data['img_dir']), 
    'ground_dir': ground_dir,
    'img_metafile': os.path.join(ground_dir, data['img_metafile']),
    'use_workers': data['use_workers'],
    'epoch_count': data['epoch_count'],
    'box_thresh': box_thresh,
    'setup': setup
  }

  return instance_setup

def read_setup(setup_filename: str = None) -> dict:
  if setup_filename == None:
    setup_filename = os.path.join(BASE_DIR, 'setup.json')

  with open(setup_filename, 'r') as f:
    data = json.load(f)
    return data

if __name__ == '__main__':
  main()