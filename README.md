# Evaluating Large-Vocabulary Detectors: The Devil is in the Details

Authors: Achal Dave, Piotr Dollar, Deva Ramanan, Alex Kirillov, Ross Girshick

This repository contains code to evaluate large-vocabulary detectors on the LVIS
dataset with our 'Fixed AP' and 'Pooled AP' metrics, as defined in our paper.

## Installation

1. Clone this repository.
1. Install locally:
    ```
    pip install -e .
    ```

## Fixed AP evaluation

Fixed AP evalutes the top K detections for each category (K=10,000 for LVIS), without
limiting the number of detections per image.
To evaluate a model's outputs using Fixed AP, use the `scripts/evaluate_ap_fixed.py`
script.

```
python scripts/evaluate_ap_fixed.py \
/path/to/lvis_v1_val.json \
/path/to/lvis_results.json \
/path/to/output_dir
```

### Ensuring K detections per class

Since the standard inference procedure for detectors will result in far fewer than
10,000 detections for most classes, Fixed AP will be artificially low.
We provide a modified detectron2 inference script which ensures that models output
10,000 detections for each class:

```
python lvdevil/infer_topk.py \
--config-file /path/to/detectron2/config.yaml \
MODEL.WEIGHTS /path/to/detectron2/model_final.pth
OUTPUT_DIR /path/to/output_dir/
```

This script has been tested with Mask R-CNN and Cascade R-CNN models.

## AP-Pool Evaluation

To evaluate using pooled AP, run the following command. Note that AP-pool similarly
uses a limit of 10,000 detections per class, with no limit on detections per image.

```
python scripts/evaluate_ap_pooled.py \
/path/to/lvis_v1_val.json \
/path/to/lvis_results.json \
/path/to/output_dir
```

## Calibration

To calibrate models, first run inference using `lvdevil/infer_topk.py` on LVIS train
and validation sets. Then, use one of the configs in `./configs/calib/` as follows:

```
python scripts/train_calibration.py \
./configs/calib/platt.yaml \
--output-dir /path/to/output_dir/ \
--opts \
TRAIN.RESULTS /path/to/train_results.json \
TEST.RESULTS /path/to/test_results.json \
TRAIN.ANNOTATIONS /path/to/lvis_v1_train.json \
TEST.ANNOTATIONS /path/to/lvis_v1_val.json
```

To calibrate on the test set (as an oracle analysis), use the following command

```
python scripts/train_calibration.py \
./configs/calib/platt.yaml \
--output-dir /path/to/output_dir/ \
--opts \
TRAIN.TRAIN_SET test \
TEST.RESULTS /path/to/test_results.json \
TEST.ANNOTATIONS /path/to/lvis_v1_val.json
```
