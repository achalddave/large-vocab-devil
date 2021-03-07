import copy
import datetime
import heapq
import itertools
import json
import logging
import os
import time
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from math import ceil
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg as get_d2_cfg, CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.data.catalog import MetadataCatalog
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import DatasetEvaluators, inference_context, print_csv_format
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from detectron2.structures import Instances
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import create_small_table, log_every_n_seconds
from fvcore.common.file_io import PathManager


logger = logging.getLogger("detectron2.infer_topk")


def get_infer_topk_cfg():
    cfg = get_d2_cfg()
    cfg.TEST.TOPK_CAT = CfgNode()
    cfg.TEST.TOPK_CAT.ENABLED = True
    cfg.TEST.TOPK_CAT.K = 10000
    cfg.TEST.TOPK_CAT.MIN_SCORE = 1.0e-7
    # Images used to estimate initial score threshold, with mask branch off.
    # Set to be greater than LVIS validation set, so we do a full pass with the
    # mask branch off.
    cfg.TEST.TOPK_CAT.NUM_ESTIMATE = 30000
    return cfg


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_infer_topk_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    args.eval_only = True
    default_setup(cfg, args)
    return cfg


def get_evaluator(cfg, dataset_name, output_folder=None):
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "lvis":
        return LVISEvaluatorMaxDets(
            dataset_name, cfg, True, output_folder, max_dets=[-1]
        )
    elif evaluator_type == "coco":
        return COCOEvaluatorMaxDets(
            dataset_name, cfg, True, output_folder, max_dets=[-1]
        )
    else:
        raise NotImplementedError


def _evaluate_predictions_on_lvis(
    lvis_gt, lvis_results, iou_type, max_dets=None, class_names=None
):
    """
    Copied from detectron2.evaluation.lvis_evaluation, with support for max_dets.

    Args:
        iou_type (str):
        kpt_oks_sigmas (list[float]):
        max_dets (None or int)
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    logger = logging.getLogger(__name__)

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    from lvis import LVISEval, LVISResults

    #####
    # <modified>
    if max_dets is None:
        max_dets = 300

    lvis_results_obj = LVISResults(lvis_gt, lvis_results, max_dets=max_dets)
    lvis_eval = LVISEval(lvis_gt, lvis_results_obj, iou_type)
    lvis_eval.params.max_dets = max_dets
    # </modified>
    #####
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    logger.info(
        f"Evaluation results for {iou_type}, max_dets {max_dets} \n"
        + create_small_table(results)
    )
    return results


def _evaluate_predictions_on_coco(
    coco_gt, coco_results, iou_type, max_dets=None, kpt_oks_sigmas=None
):
    """
    Evaluate the coco results using COCOEval API.

    Copied from detectron2.evaluation.coco_evaluation, with support for max_dets.

    Args:
        max_dets (int or None)
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    #####
    # <modified>
    if max_dets is None:
        max_dets = [1, 10, 100]

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.maxDets = [x if x > 0 else len(coco_results) for x in max_dets]
    # </modified>
    #####

    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

    if iou_type == "keypoints":
        num_keypoints = len(coco_results[0]["keypoints"]) // 3
        assert len(coco_eval.params.kpt_oks_sigmas) == num_keypoints, (
            "[COCOEvaluator] The length of cfg.TEST.KEYPOINT_OKS_SIGMAS (default: 17) "
            "must be equal to the number of keypoints. However the prediction has {} "
            "keypoints! For more information please refer to "
            "http://cocodataset.org/#keypoints-eval.".format(num_keypoints)
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


class LVISEvaluatorMaxDets(LVISEvaluator):
    """LVISEvaluator which allows modifying max dets."""

    def __init__(self, *args, max_dets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_dets = max_dets

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.

        Like super()._eval_predictions, but support modifying max dets.

        Args:
            predictions (list[dict]): list of outputs from the model
        """
        self._logger.info("Preparing results in the LVIS format ...")
        lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        #####
        # <modified>
        for d in self.max_dets:
            if len(self.max_dets) > 1:
                # Evaluation modifies results possibly, so make a copy.
                max_dets_results = copy.deepcopy(lvis_results)
            else:
                max_dets_results = lvis_results
            logger.info(f"Evaluating with max-dets={d}")
            self._results[f"maxdets={d}"] = OrderedDict()
            for task in sorted(tasks):
                res = _evaluate_predictions_on_lvis(
                    self._lvis_api,
                    max_dets_results,
                    task,
                    max_dets=d,
                    class_names=self._metadata.get("thing_classes"),
                )
                self._results[f"maxdets={d}"][task] = res
        # </modified>
        #####


class COCOEvaluatorMaxDets(COCOEvaluator):
    """COCOEvaluator which allows modifying max dets."""

    def __init__(self, *args, max_dets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_dets = max_dets

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.

        Like super()._eval_predictions, but support modifying max dets.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        #####
        # <modified>
        self._results = defaultdict(OrderedDict)
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    max_dets=self.max_dets,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            for d in self.max_dets:
                if task in ("bbox", "segm"):
                    coco_eval.stats = self._summarizeDets(coco_eval, d)
                else:
                    coco_eval.stats = self._summarizeKps(coco_eval, d)
                logger.info(f"Evaluating with max dets: {d}")
                res = self._derive_coco_results(
                    coco_eval, task, class_names=self._metadata.get("thing_classes")
                )
                self._results[f"maxdets={d}"][task] = res
        # </modified>
        #####

    def _summarize_coco(_, coco_eval, ap=1, iouThr=None, areaRng="all", maxDets=100):
        """_summarize function copied from the definition in COCOEval.summarize()."""
        p = coco_eval.params
        iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
        titleStr = "Average Precision" if ap == 1 else "Average Recall"
        typeStr = "(AP)" if ap == 1 else "(AR)"
        iouStr = (
            "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
            if iouThr is None
            else "{:0.2f}".format(iouThr)
        )

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval["precision"]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s

    def _summarizeDets(self, coco_eval, max_dets):
        """Like _summarizeDets in COCOEval.summarize, but for one maxDets setting."""
        stats = np.zeros((12,))
        _summarize = self._summarize_coco
        stats[0] = _summarize(coco_eval, 1, maxDets=max_dets)
        stats[1] = _summarize(coco_eval, 1, iouThr=0.5, maxDets=max_dets)
        stats[2] = _summarize(coco_eval, 1, iouThr=0.75, maxDets=max_dets)
        stats[3] = _summarize(coco_eval, 1, areaRng="small", maxDets=max_dets)
        stats[4] = _summarize(coco_eval, 1, areaRng="medium", maxDets=max_dets)
        stats[5] = _summarize(coco_eval, 1, areaRng="large", maxDets=max_dets)
        # Usually used for other max dets
        stats[6] = stats[7] = -1
        stats[8] = _summarize(coco_eval, 0, maxDets=max_dets)
        stats[9] = _summarize(coco_eval, 0, areaRng="small", maxDets=max_dets)
        stats[10] = _summarize(coco_eval, 0, areaRng="medium", maxDets=max_dets)
        stats[11] = _summarize(coco_eval, 0, areaRng="large", maxDets=max_dets)
        return stats

    def _summarizeKps(self, coco_eval, max_dets):
        """Like _summarizeKps in COCOEval.summarize, but for one maxDets setting."""
        stats = np.zeros((10,))
        _summarize = self._summarize_coco
        stats[0] = _summarize(coco_eval, 1, maxDets=max_dets)
        stats[1] = _summarize(coco_eval, 1, maxDets=max_dets, iouThr=0.5)
        stats[2] = _summarize(coco_eval, 1, maxDets=max_dets, iouThr=0.75)
        stats[3] = _summarize(coco_eval, 1, maxDets=max_dets, areaRng="medium")
        stats[4] = _summarize(coco_eval, 1, maxDets=max_dets, areaRng="large")
        stats[5] = _summarize(coco_eval, 0, maxDets=max_dets)
        stats[6] = _summarize(coco_eval, 0, maxDets=max_dets, iouThr=0.5)
        stats[7] = _summarize(coco_eval, 0, maxDets=max_dets, iouThr=0.75)
        stats[8] = _summarize(coco_eval, 0, maxDets=max_dets, areaRng="medium")
        stats[9] = _summarize(coco_eval, 0, maxDets=max_dets, areaRng="large")
        return stats


@contextmanager
def _turn_off_roi_heads(model, attrs):
    """
    Open a context where some heads in `model.roi_heads` are temporarily turned off.
    Args:
        attr (list[str]): the attribute in `model.roi_heads` which can be used
            to turn off a specific head, e.g., "mask_on", "keypoint_on".
    """
    roi_heads = model.roi_heads
    old = {}
    for attr in attrs:
        try:
            old[attr] = getattr(roi_heads, attr)
        except AttributeError:
            # The head may not be implemented in certain ROIHeads
            pass

    if len(old.keys()) == 0:
        yield
    else:
        for attr in old.keys():
            setattr(roi_heads, attr, False)
        yield
        for attr in old.keys():
            setattr(roi_heads, attr, old[attr])


@contextmanager
def _per_class_thresholded_inference_standard(model, score_thresholds):
    """
    Args:
        model (nn.Module): Detectron2 model.
        score_threshold (List[float]): Threshold for each class.
    """

    def _inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        # <modified>
        for image_scores in scores:
            image_scores[image_scores < score_thresholds] = -1
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh=0,
            nms_thresh=self.test_nms_thresh,
            topk_per_image=-1,
        )
        # </modified>

    box_predictor = model.roi_heads.box_predictor
    old_inference = box_predictor.inference
    box_predictor.inference = _inference.__get__(box_predictor)  # Get bound method
    yield
    box_predictor.inference = old_inference


@contextmanager
def _per_class_thresholded_inference_cascade(model, score_thresholds):
    """
    Args:
        model (nn.Module): Detectron2 model.
        score_threshold (List[float]): Threshold for each class.
    """

    def _forward_box(self, features, proposals, targets=None, extra_info=None):
        """
        CascadeROIHeads._forward_box modified to include per-class thresholds.

        Args:
            features, targets: the same as in
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are used to create the input
                # proposals of the next stage.
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes
                )
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                predictions, proposals
            )
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            # <modified>
            raise NotImplementedError
            # </modified>
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            # Use the boxes of the last head
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            # <modified>
            for image_scores in scores:
                image_scores[image_scores < score_thresholds] = -1
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                score_thresh=0,
                nms_thresh=predictor.test_nms_thresh,
                topk_per_image=-1,
            )
            return pred_instances
            # </modified>

    roi_heads = model.roi_heads
    _old_forward_box = roi_heads._forward_box
    roi_heads._forward_box = _forward_box.__get__(roi_heads)  # Get bound method
    yield
    roi_heads._forward_box = _old_forward_box


@contextmanager
def per_class_thresholded_inference(model, score_thresholds):
    if isinstance(model.roi_heads, CascadeROIHeads):
        fn = _per_class_thresholded_inference_cascade
    else:
        fn = _per_class_thresholded_inference_standard
    with fn(model, score_thresholds):
        yield


@contextmanager
def limit_mask_branch_proposals(model, max_proposals):
    """Modify an RCNN model to limit proposals processed by mask head at a time."""

    def _forward_with_given_boxes_limited(self, features, instances):
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        max_instances = max(len(x) for x in instances)
        if max_instances > max_proposals and (self.mask_on or self.keypoint_on):
            outputs = []
            num_ims = len(instances)
            from tqdm import tqdm

            for i in range(0, max_instances, max_proposals):
                chunk = [x[i : i + max_proposals] for x in instances]
                chunk = self._forward_mask(features, chunk)
                chunk = self._forward_keypoint(features, chunk)
                outputs.append(chunk)
            instances = [Instances.cat([x[j] for x in outputs]) for j in range(num_ims)]
        else:
            instances = self._forward_mask(features, instances)
            instances = self._forward_keypoint(features, instances)
        return instances

    roi_heads = model.roi_heads
    _old_fn = roi_heads.forward_with_given_boxes
    _new_fn = _forward_with_given_boxes_limited.__get__(roi_heads)  # Get bound method.
    roi_heads.forward_with_given_boxes = _new_fn
    yield
    roi_heads.forward_with_given_boxes = _old_fn


def inference_on_dataset(
    model, data_loader, evaluator, num_classes, topk, num_estimate, min_score
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.
        topk (int)
        num_estimate (int): Number of images to estimate initial score threshold.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger.info("Start inference on {} images".format(len(data_loader)))
    if isinstance(topk, int):
        logger.info(f"Collecting top-{topk} images.")
        topk = [topk] * num_classes
    else:
        logger.info(f"Collecting top-k images. Counts:\n{topk}")

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    # We keep track of scores from _this_ process (process_scores) and scores from
    # all processes (scores). Every iter, each process updates process_scores and its
    # local scores with the new scores from the model.
    # Every few iterations, all processes pass their process_scores to each other and
    # updates their own global scores.

    # Map category id to min-heap of top scores from this process.
    process_scores = defaultdict(list)
    # Map category id to min-heap of top scores from all processes.
    global_scores = defaultdict(list)
    init_thresholds = torch.full(
        (num_classes + 1,), fill_value=min_score, dtype=torch.float32
    ).to(model.device)
    init_threshold_path = Path(evaluator._output_dir) / "_thresholds_checkpoint.pth"
    if init_threshold_path.exists():
        logger.info("Loading thresholds from disk.")
        init_thresholds = torch.load(init_threshold_path).to(model.device)
    else:
        init_threshold_path.parent.mkdir(exist_ok=True, parents=True)

    # Trying to get exactly the top-k estimates can result in getting slightly fewer
    # than K estimates. This can happen due to subtle differences in the model's forward
    # pass in the first phase vs. the second phase. For example, in the first phase,
    # when we have low thresholds, D2 will use torchvision.ops.boxes.batched_nms for
    # batch NMS. In phase 2, D2 will use a slightly different, customized
    # implementation, which may occasionally result in fewer boxes.
    # To address this, we set thresholds to be a bit looser, targeting 10% more
    # predictions than requested.
    topk_loose = [int(ceil(k * 1.1)) for k in topk]

    def get_thresholds(scores, min_thresholds):
        thresholds = []
        for i in range(num_classes):
            if topk_loose[i] == 0:
                thresholds.append(float("inf"))
            elif len(scores[i]) < topk_loose[i]:
                thresholds.append(-1)
            else:
                thresholds.append(scores[i][0])
        # Add -1 for background
        thresholds = torch.FloatTensor(thresholds + [-1]).to(model.device)
        # Clamp at minimum thresholds
        return torch.max(thresholds, init_thresholds)

    def update_scores(scores, inputs, outputs):
        updated = set()
        for image, output in zip(inputs, outputs):
            if isinstance(output, dict):
                instances = output["instances"]
            else:
                instances = output
            curr_labels = instances.pred_classes.int().tolist()
            curr_scores = instances.scores.cpu().tolist()
            for label, score in zip(curr_labels, curr_scores):
                # label = label.int().item()
                # scores[label].append((image["image_id"], score.cpu().item()))
                if len(scores[label]) >= topk_loose[label]:
                    if score < scores[label][0]:
                        continue
                    else:
                        heapq.heappushpop(scores[label], score)
                else:
                    heapq.heappush(scores[label], score)
                updated.add(label)

    def gather_scores(process_scores):
        # List of scores per process
        scores_list = comm.all_gather(process_scores)
        gathered = defaultdict(list)
        labels = {x for scores in scores_list for x in scores.keys()}
        for label in labels:
            # Sort in descending order.
            sorted_generator = heapq.merge(
                *[sorted(x[label], reverse=True) for x in scores_list], reverse=True
            )
            top_k = itertools.islice(sorted_generator, topk_loose[label])
            top_k_ascending = list(reversed(list(top_k)))  # Return to ascending order
            heapq.heapify(top_k_ascending)
            gathered[label] = top_k_ascending
        return gathered

    with inference_context(model), torch.no_grad():
        #########
        # Phase 1: Compute initial, low score thresholds without mask branch.
        #########
        # First, get an estimate of score thresholds with the mask branch off.
        # Otherwise, in the initial few images, we will run the mask branch on a bunch
        # of useless proposals which makes everything slow.
        num_estimate = min(num_estimate, len(data_loader))
        for idx, inputs in enumerate(
            tqdm(
                data_loader,
                desc="Computing score thresholds",
                total=num_estimate,
                disable=comm.get_rank() != 0,
            )
        ):
            if idx > num_estimate:
                break
            # Gather scores from other processes periodically.
            # In early iterations, the thresholds are low, making inference slow and
            # gather relatively fast, so we gather more often.
            # Later, the thresholds are high enough that inference is fast and gathering
            # is slow, so we stop gathering.
            if (idx < 100 and idx % 10 == 0) or (idx % 500 == 0):
                global_scores = gather_scores(process_scores)

            thresholds = get_thresholds(global_scores, init_thresholds)
            if idx % 1000 == 0:  # Save thresholds for later runs
                torch.save(thresholds, init_threshold_path)

            with per_class_thresholded_inference(model, thresholds):
                with _turn_off_roi_heads(model, ["mask_on", "keypoint_on"]):
                    outputs = model.inference(inputs, do_postprocess=False)
            update_scores(global_scores, inputs, outputs)
            update_scores(process_scores, inputs, outputs)

            if (idx < 100 and idx % 10 == 0) or (idx % 100 == 0):
                logger.info(
                    "Threshold range (%s, %s); # collected: (%s, %s)",
                    thresholds[:-1].min(),
                    thresholds[:-1].max(),
                    min(len(x) for x in global_scores.values()),
                    max(len(x) for x in global_scores.values()),
                )

        del global_scores
        # Necessary to avoid timeout when gathering?
        comm.synchronize()

        # Map class to scores of predictions so far.
        init_scores = gather_scores(process_scores)
        # Minimum thresholds from the estimate stage
        init_thresholds = get_thresholds(init_scores, init_thresholds)
        # Clear scores from estimates; we will start tracking them again.
        scores = defaultdict(list)

        #########
        # Phase 2: Collect top-k predictions, with mask branch enabled.
        #########
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            thresholds = get_thresholds(scores, init_thresholds)
            with per_class_thresholded_inference(model, thresholds):
                with limit_mask_branch_proposals(model, max_proposals=300):
                    outputs = model(inputs)
            update_scores(scores, inputs, outputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                    name=logger.name,
                )

            # Clear unnecessary predictions every so often.
            if idx < 100 or ((idx + 1) % 10) == 0:
                by_cat = defaultdict(list)
                for pred in evaluator._predictions:
                    for ann in pred["instances"]:
                        by_cat[ann["category_id"]].append(ann)
                topk_preds = []
                for c, anns in by_cat.items():
                    topk_preds.extend(
                        sorted(anns, key=lambda a: a["score"], reverse=True)[: topk[c]]
                    )
                evaluator._predictions = [{"instances": topk_preds}]

    if evaluator._output_dir:
        PathManager.mkdirs(evaluator._output_dir)
        file_path = os.path.join(
            evaluator._output_dir, f"instances_predictions_rank{comm.get_rank()}.pth"
        )
        with PathManager.open(file_path, "wb") as f:
            torch.save(evaluator._predictions, f)

    # Necessary to avoid timeout when gathering?
    comm.synchronize()
    # Limit number of detections per category across workers.
    predictions = comm.gather(evaluator._predictions, dst=0)
    if comm.is_main_process():
        predictions = list(itertools.chain(*predictions))
        by_cat = defaultdict(list)
        for pred in predictions:
            for ann in pred["instances"]:
                by_cat[ann["category_id"]].append(ann)
        logger.info(f"Max per cat: {max([len(v) for v in by_cat.values()])}")
        logger.info(f"Min per cat: {min([len(v) for v in by_cat.values()])}")
        topk_preds = []
        for c, anns in by_cat.items():
            topk_preds.extend(
                sorted(anns, key=lambda a: a["score"], reverse=True)[: topk[c]]
            )
        evaluator._predictions = [{"instances": topk_preds}]
    else:
        evaluator._predictions = []

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def main(args):
    cfg = setup(args)
    assert cfg.TEST.TOPK_CAT.ENABLED
    topk = cfg.TEST.TOPK_CAT.K

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(
            model,
            data_loader,
            evaluator,
            num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            topk=topk,
            num_estimate=cfg.TEST.TOPK_CAT.NUM_ESTIMATE,
            min_score=cfg.TEST.TOPK_CAT.MIN_SCORE,
        )
        results[dataset_name] = results_i
        if comm.is_main_process():
            for max_dets, max_dets_results in results_i.items():
                logger.info(
                    f"Evaluation results for {dataset_name},max_dets={max_dets} in "
                    f"csv format:"
                )
                print_csv_format(max_dets_results)
    if len(results) == 1:
        results = list(results.values())[0]

    with open(Path(cfg.OUTPUT_DIR) / "metrics_infer_topk.json", "w") as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
