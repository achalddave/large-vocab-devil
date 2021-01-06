import copy
import datetime
import heapq
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from lvis import LVIS, LVISEval, LVISResults
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


class EvalWrapper:
    def __init__(
        self,
        groundtruth,
        results,
        dataset_type="lvis",
        iou_type="segm",
        max_dets=300,
        max_dets_per_class=-1,
        ious=None,
    ):
        if ious is None:
            ious = np.linspace(
                0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
            )
        self.dataset_type = dataset_type
        self.iou_type = iou_type
        self.max_dets = max_dets
        self.max_dets_per_class = max_dets_per_class
        self.ious = ious
        self.groundtruth_original = groundtruth
        self.results_original = results
        # Load lazily so we can pickle this object with low communication overhead.
        self._groundtruth, self._results = None, None

    def _lazy_load_groundtruth_results(self):
        self._groundtruth, self._results = self._load_gt_results(
            self.groundtruth_original, self.results_original
        )

    @property
    def groundtruth(self):
        if self._groundtruth is None:
            self._lazy_load_groundtruth_results()
        return self._groundtruth

    @property
    def results(self):
        if self._results is None:
            self._lazy_load_groundtruth_results()
        return self._results

    def is_coco(self):
        return self.dataset_type == "coco"

    def is_lvis(self):
        return self.dataset_type == "lvis"

    def compute_ap(self):
        eval_obj = self.construct_eval(use_cats=True)
        eval_obj.run()
        return eval_obj.results["AP"]

    def construct_pooled_eval(self, *, pools=None):
        groundtruth = self.groundtruth
        results = self.results
        if not self.is_lvis():
            raise NotImplementedError("Pooled eval currently only supported for LVIS.")
        eval_obj = LVISPooledEval(groundtruth, results, self.iou_type, pools=pools)
        eval_obj.params.max_dets = self.max_dets
        eval_obj.params.iou_thrs = self.ious
        eval_obj.params.area_rng = [eval_obj.params.area_rng[0]]
        eval_obj.params.area_rng_lbl = [eval_obj.params.area_rng_lbl[0]]
        return eval_obj

    def construct_eval(self, use_cats, groundtruth=None, results=None):
        if groundtruth is None:
            groundtruth = self.groundtruth
        if results is None:
            results = self.results
        if self.is_lvis():
            eval_obj = LVISEval(groundtruth, results, self.iou_type)
            eval_obj.params.max_dets = self.max_dets
            eval_obj.params.use_cats = use_cats
            eval_obj.params.iou_thrs = self.ious
            eval_obj.params.area_rng = [eval_obj.params.area_rng[0]]
            eval_obj.params.area_rng_lbl = [eval_obj.params.area_rng_lbl[0]]
        else:  # COCO
            if not use_cats:
                results = copy.deepcopy(results)
                for ann in results.dataset["annotations"]:
                    ann["category_id"] = 1
                results.createIndex()

            eval_obj = COCOEvalWrapper(groundtruth, results, self.iou_type)
            eval_obj.params.maxDets = [self.max_dets]
            eval_obj.params.useCats = use_cats
            eval_obj.params.iouThrs = self.ious
            eval_obj.params.areaRng = [eval_obj.params.areaRng[0]]
            eval_obj.params.areaRngLbl = [eval_obj.params.areaRngLbl[0]]
        return eval_obj

    def _load_gt_results(self, groundtruth, results):
        if isinstance(groundtruth, dict):
            fp = tempfile.NamedTemporaryFile("w")
            json.dump(groundtruth, fp)
            fp.seek(0)
            groundtruth = fp.name
        if self.max_dets_per_class >= 0:
            if isinstance(results, (str, Path)):
                with open(results, "r") as f:
                    results = json.load(f)
            results = limit_dets_per_class(results, self.max_dets_per_class)
        if self.is_lvis():
            if not isinstance(groundtruth, LVIS):
                groundtruth = LVIS(str(groundtruth))
            if isinstance(results, Path):
                results = str(results)
            results = LVISResults(groundtruth, results, self.max_dets)
        else:  # COCO
            if not isinstance(groundtruth, COCO):
                groundtruth = COCO(str(groundtruth))
            if isinstance(results, Path):
                results = str(results)
            if not isinstance(results, COCO):
                results = groundtruth.loadRes(results)
        return groundtruth, results


class COCOEvalWrapper(COCOeval):
    """Updates to COCOeval to make it act more like LVISEval."""
    def run(self):
        self.evaluate()
        self.accumulate()
        self.summarize()

    def summarize(self, verbose=False):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
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
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            if verbose:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = {}
            stats["AP"] = _summarize(1, maxDets=self.params.maxDets[-1])
            stats["AP50"] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[-1])
            stats["AP75"] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[-1])
            stats["APs"] = _summarize(
                1, areaRng="small", maxDets=self.params.maxDets[-1]
            )
            stats["APm"] = _summarize(
                1, areaRng="medium", maxDets=self.params.maxDets[-1]
            )
            stats["APl"] = _summarize(
                1, areaRng="large", maxDets=self.params.maxDets[-1]
            )
            for md in self.params.maxDets:
                stats[f"AR@{md}"] = _summarize(0, maxDets=self.params.maxDets[0])
            stats["ARs"] = _summarize(
                0, areaRng="small", maxDets=self.params.maxDets[-1]
            )
            stats["ARm"] = _summarize(
                0, areaRng="medium", maxDets=self.params.maxDets[-1]
            )
            stats["ARl"] = _summarize(
                0, areaRng="large", maxDets=self.params.maxDets[-1]
            )
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        self.stats = _summarizeDets()
        self.results = self.stats


class LVISPooledEval(LVISEval):
    """Like LVISEval, but compute AP pooled across category groups.

    To do this, we do the following:
        - Match predictions to groundtruth per category, as in standard evaluation.
            - evaluate() is not modified _at all_; neither are compute_iou() or
              evaluate_img().
        - When computing precision and recall, compute TP, FP, FN over all categories
          in a pool, instead of per category.
        - In summarize(), report AP per pool.

    This class modifies LVISEval as little as possible. See docstrings for each function
    to see what is modified.
    """

    def __init__(self, lvis_gt, lvis_dt, iou_type="segm", pools=None):
        super().__init__(lvis_gt, lvis_dt, iou_type)
        if pools is None:
            pools = ["all"]
        assert all(x in {"all", "r", "c", "f"} for x in pools)
        self.pools = pools

    def accumulate(self):
        """
        Like super().accumulate(), but compute statistics pooled across categories.

        All changes from the original code are marked within <modified></modified>
        blocks. At a high level the changes are the following:
            - Treat each group of categories (pool) as a category on its own.
            - Compute stats (TP, FP, FN) per pool instead of per category.
        """
        self.logger.info("Accumulating evaluation results.")

        if not self.eval_imgs:
            self.logger.warn("Please run evaluate first.")

        ####
        # <modified>
        # Map pool to list of category indices, which index into params.cat_ids.
        pool_cat_indices = {
            "all": list(range(len(self.params.cat_ids))),
            "r": self.freq_groups[self.params.img_count_lbl.index("r")],
            "c": self.freq_groups[self.params.img_count_lbl.index("c")],
            "f": self.freq_groups[self.params.img_count_lbl.index("f")],
        }
        assert self.params.use_cats
        # if self.params.use_cats:
        #     cat_ids = self.params.cat_ids
        # else:
        #     cat_ids = [-1]
        # </modified>
        ####

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        # <modified>
        # num_cats = len(cat_ids)
        num_pools = len(self.pools)
        # </modified>
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        # <modified> num_cats -> num_pools, cat_idx -> pool_idx
        # -1 for absent categories
        precision = -np.ones((num_thrs, num_recalls, num_pools, num_area_rngs))
        recall = -np.ones((num_thrs, num_pools, num_area_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for pool_idx in range(num_pools):
            dt_pointers[pool_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[pool_idx][area_idx] = {}
        # </modified>

        # <modified>cat_idx -> pool_idx
        for pool_idx in range(num_pools):
            # Nk = cat_idx * num_area_rngs * num_imgs
            # </modified>
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                ####
                # <modified>
                E = [
                    self.eval_imgs[c * num_area_rngs * num_imgs + Na + img_idx]
                    for img_idx in range(num_imgs)
                    for c in pool_cat_indices[self.pools[pool_idx]]
                ]
                # </modified>
                ####
                # Remove elements which are None
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                # Append all scores: shape (N,)
                dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

                gt_ig = np.concatenate([e["gt_ignore"] for e in E])
                # num gt anns to consider
                num_gt = np.count_nonzero(gt_ig == 0)

                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                # <modified>: cat_idx -> pool_idx
                dt_pointers[pool_idx][area_idx] = {
                    "dt_ids": dt_ids,
                    "tps": tps,
                    "fps": fps,
                }
                # </modified>

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    # <modified>: cat_idx -> pool_idx
                    if num_tp:
                        recall[iou_thr_idx, pool_idx, area_idx] = rc[-1]
                    else:
                        recall[iou_thr_idx, pool_idx, area_idx] = 0
                    # </modified>

                    # np.spacing(1) ~= eps
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = pr.tolist()

                    # Replace each precision value with the maximum precision
                    # value to the right of that recall level. This ensures
                    # that the  calculated AP value will be less suspectable
                    # to small variations in the ranking.
                    for i in range(num_tp - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    rec_thrs_insert_idx = np.searchsorted(
                        rc, self.params.rec_thrs, side="left"
                    )

                    pr_at_recall = [0.0] * num_recalls

                    try:
                        for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                            pr_at_recall[_idx] = pr[pr_idx]
                    except:
                        pass
                    # <modified>: cat_idx -> pool_idx
                    precision[iou_thr_idx, :, pool_idx, area_idx] = np.array(
                        pr_at_recall
                    )
                    # </modified>

        self.eval = {
            "params": self.params,
            # <modified> num_cats -> num_pools
            "counts": [num_thrs, num_recalls, num_pools, num_area_rngs],
            # </modified>
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
        }

    def _summarize_pool(self, summary_type, pool_idx, iou_thr=None, area_rng="all"):
        """Like super()._summarize, but only summarize for a specified pool.

        Modifications:
            - Report AP per category pool/group.
            - Remove support for freq_group_idx; this is subsumed by pool_idx.

        Args:
            pool_idx (int): Index into self.pools. This is the pool over which AP is
                reported.
        """
        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == "ap":
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, pool_idx, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, pool_idx, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Call _summarize_pool with each pool."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        # Loop in fixed order (instead of looping over self.pools), since results is an
        # OrderedDict.
        for p in ("all", "r", "c", "f"):
            if p in self.pools:
                self.results[f"AP-pooled-{p}"] = self._summarize_pool(
                    "ap", pool_idx=self.pools.index(p)
                )


def _limit_dets(results, topk, per="class"):
    assert per in ("class", "image")
    # This function may be used on _large_ results (e.g., 80GB), so we avoid creating
    # a mapping from category to all annotations of the category.
    # Instead, scores maps a class or image to a min-heap containing (score, index) of
    # size at most topk.
    top_scoring = defaultdict(list)
    output = {}
    for i, ann in enumerate(tqdm(results, mininterval=1.0)):
        key = ann["category_id"] if per == "class" else ann["image_id"]
        score = ann["score"]
        if len(top_scoring[key]) < topk:
            heapq.heappush(top_scoring[key], (score, i))
            output[i] = ann
        elif len(top_scoring[key]) > topk and score > top_scoring[key][0][0]:
            _, removed_idx = heapq.heappushpop(top_scoring[key], (score, i))
            output[i] = ann
            del output[removed_idx]
    return list(output.values())


def limit_dets_per_class(results, topk):
    return _limit_dets(results, topk, per="class")


def limit_dets_per_image(results, topk):
    return _limit_dets(results, topk, per="image")
