from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

IOU_THRESH = 0.5

def calc_map_and_class_ap(
  grnd_bboxes: list[Tensor],
  grnd_labels: list[Tensor],
  pred_bboxes: list[Tensor],
  pred_labels: list[Tensor],
  pred_scores: list[Tensor]
):
  metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[IOU_THRESH], class_metrics=True)

  for i in range(len(pred_bboxes)):
    preds = [{
      "boxes": pred_bboxes[i],
      "scores": pred_scores[i],
      "labels": pred_labels[i]
    }]
    targets = [{
      "boxes": grnd_bboxes[i],
      "labels": grnd_labels[i]
    }]

    metric.update(preds, targets)

  results = metric.compute()

  return {
    "map": results["map"],
    "map_per_class": results["map_per_class"]
  }

def calc_iou(
  grnd_bbox: list[float] | Tensor,
  pred_bbox: list[float] | Tensor
) -> float:
  if isinstance(grnd_bbox, Tensor):
    grnd_bbox = [point.item() for point in grnd_bbox]
  if isinstance(pred_bbox, Tensor):
    pred_bbox = [point.item() for point in pred_bbox]

  if not rects_overlap(grnd_bbox, pred_bbox):
    return 0
  
  grnd_area = (grnd_bbox[2]-grnd_bbox[0]) * (grnd_bbox[3]-grnd_bbox[1])
  pred_area = (pred_bbox[2]-pred_bbox[0]) * (pred_bbox[3]-pred_bbox[1])
  overlap = (min(grnd_bbox[2], pred_bbox[2]) - max(grnd_bbox[0], pred_bbox[0])) * (min(grnd_bbox[3], pred_bbox[3]) - max(grnd_bbox[1], pred_bbox[1]))

  iou = overlap / (grnd_area + pred_area - overlap)

  return iou

def calc_image_avg_iou(
  grnd_bboxes: Tensor,
  grnd_labels: Tensor,
  pred_bboxes: Tensor,
  pred_labels: Tensor
) -> float:
  preds = {
    1: [],
    2: [],
    3: []
  }
  for pred_bbox, pred_label in zip(pred_bboxes, pred_labels):
    preds[pred_label.item()].append([item.item() for item in pred_bbox])

  iou_list = [] 
  for grnd_bbox, grnd_label in zip(grnd_bboxes, grnd_labels):
    highest_iou = 0
    for pred_bbox in preds[grnd_label.item()]:
      iou = calc_iou(grnd_bbox, pred_bbox)
      highest_iou = iou if highest_iou < iou else highest_iou
    
    iou_list.append(highest_iou)
  
  return sum(iou_list) / len(iou_list)

def calc_images_avg_iou(
  grnd_bboxes: list[Tensor],
  grnd_labels: list[Tensor],
  pred_bboxes: list[Tensor],
  pred_labels: list[Tensor]
) -> float:
  iou_list = [] 
  for g_b, g_l, p_b, p_l in zip(grnd_bboxes, grnd_labels, pred_bboxes, pred_labels):
    iou_list.append(calc_image_avg_iou(g_b, g_l, p_b, p_l))
    
  return sum(iou_list) / len(iou_list)

def calc_image_conf_matrix_for_class(
  label_id: int,
  grnd_bboxes: Tensor,
  grnd_labels: Tensor,
  pred_bboxes: Tensor,
  pred_labels: Tensor
) -> dict:   
  label_grnd_bboxes = [box for box, label in zip(grnd_bboxes, grnd_labels) if label.item() == label_id]
  grnd_len = len(label_grnd_bboxes)
  
  tp = 0
  fp = 0
  fn = grnd_len - len(pred_bboxes) if grnd_len > len(pred_bboxes) else 0

  iou_matrix = []
  for p_box in pred_bboxes:
    p_ious = []
    for g_box in label_grnd_bboxes:
      p_ious.append(calc_iou(g_box, p_box))
    iou_matrix.append(p_ious)
  
  iou_matrix_flattened = []
  for p, row in zip(range(len(iou_matrix)), iou_matrix):
    for g, cell in zip(range(len(row)), row):
      iou_matrix_flattened.append((p, g, cell))

  check = grnd_len - fn
  for i in range(check):
    p_index, g_index, value = max(iou_matrix_flattened, key=lambda x: x[2])
    
    pred_label = pred_labels[p_index].item()
    is_thresh = value >= IOU_THRESH
    if pred_label == label_id and is_thresh:
      tp += 1
    elif is_thresh:
      fp += 1

    if i + 1 != grnd_len:
      iou_matrix_flattened = [(p, g, value) for p, g, value in iou_matrix_flattened if p == p_index or g == g_index]

  fn += grnd_len - tp 

  return {
    'tp': tp,
    'fp': fp,
    'fn': fn
  }

SMALL_VALUE = 1e-30

def calc_images_metrics_for_class(
  label_id: int,
  grnd_bboxes: list[Tensor],
  grnd_labels: list[Tensor],
  pred_bboxes: list[Tensor],
  pred_labels: list[Tensor]
) -> dict:
  matrixes = [calc_image_conf_matrix_for_class(label_id, g_b, g_l, p_b, p_l) for g_b, g_l, p_b, p_l in zip(grnd_bboxes, grnd_labels, pred_bboxes, pred_labels)]
  tp = sum([matrix['tp'] for matrix in matrixes])
  fp = sum([matrix['fp'] for matrix in matrixes])
  fn = sum([matrix['fn'] for matrix in matrixes])

  prec = tp / max(tp + fp, SMALL_VALUE) 
  rec = tp / max(tp + fn, SMALL_VALUE)
  f1 = 2 * prec * rec / max(prec + rec, SMALL_VALUE)

  return {
    'precision': prec,
    'recall': rec,
    'f1': f1
  }

def print_metrics(
  grnd_bboxes: list[Tensor],
  grnd_labels: list[Tensor],
  pred_bboxes: list[Tensor],
  pred_labels: list[Tensor],
  pred_scores: list[Tensor]
):
  puzzle_metrics = calc_images_metrics_for_class(0, grnd_bboxes, grnd_labels, pred_bboxes, pred_labels)
  wordlist_metrics = calc_images_metrics_for_class(1, grnd_bboxes, grnd_labels, pred_bboxes, pred_labels)
  other_metrics = calc_images_metrics_for_class(2, grnd_bboxes, grnd_labels, pred_bboxes, pred_labels)
  
  iou = calc_images_avg_iou(grnd_bboxes, grnd_labels, pred_bboxes, pred_labels)
  map_and_ap = calc_map_and_class_ap(grnd_bboxes, grnd_labels, pred_bboxes, pred_labels, pred_scores)

  for name, metrics, ap in zip(['PUZZLE', 'WORDLIST', 'OTHER'], [puzzle_metrics, wordlist_metrics, other_metrics], map_and_ap['map_per_class']):
    print(f"{name} metrics | Precision: {metrics['precision']:.4} | Recall: {metrics['recall']:.4} | F1: {metrics['f1']:.4} | Average Precision: {ap.item():.4}")

  print(f"Average IOU: {iou:.4} | Mean Average Precision: {map_and_ap['map'].item():.4}")

def rects_overlap(a: list[float], b: list[float]) -> bool:
  return not (
    a[2] <= b[0] or  
    a[0] >= b[2] or  
    a[3] <= b[1] or  
    a[1] >= b[3]
  )