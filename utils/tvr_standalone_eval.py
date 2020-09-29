'''
TVR Evaluation functions modified from TVRetrieval implementation
(https://github.com/jayleicn/TVRetrieval)

Load prediction file and GT file to calculate TVR metrics:
- recall at top K (R@K), for a specified IoU, where K in [1, 5, 10, 100], IoU in [0.5, 0.7]
'''
import json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def pad_sequences_1d_np(sequences, dtype=np.float32):

    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    assert "numpy" in str(dtype), "dtype and input type does not match"
    padded_seqs = np.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = np.zeros((len(sequences), max(lengths)), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask


def compute_temporal_iou_batch(preds, gt):
    """ compute intersection-over-union along temporal axis
    This function is significantly faster than `compute_temporal_iou`,
    the result should be the same.
    Args:
        preds: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt: [st (float), ed (float)]
    Returns:
        iou (float): np.ndarray, (N, )

    References:
        for np.divide with zeros, see https://stackoverflow.com/a/37977222
    """
    intersection = np.maximum(0, np.minimum(preds[:, 1], gt[1]) - np.maximum(preds[:, 0], gt[0]))
    union = np.maximum(preds[:, 1], gt[1]) - np.minimum(preds[:, 0], gt[0])  # not the correct union though
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def get_rounded_percentage(float_number, n_floats=2):
    return round(float_number * 100, n_floats)


TASK_TYPES = OrderedDict([
    ("VCMR", "Video Corpus Moment Retrieval"),
    ("SVMR", "Single Video Moment Retrieval"),
    ("VR", "regular Video Retrieval")
])


def eval_by_task_type(moment_predictions, video2idx, ground_truth,
                      iou_thds=(0.5, 0.7), recall_topks=(1, 5, 10, 100),
                      task_type="SVMR", max_pred_per_query=100,
                      match_number=True, verbose=True, use_desc_type=True):
    """ a predicted triplet is positive only if:
    1) its vid_name matches the GT vid_name
    2) IoU between its timestamp and GT timestamp is higher than the given threshold

    moment_predictions w.r.t. different task_type:
        For each query, evaluated on top max_pred_per_query [vid_name, st, ed] triplets. (score entry ignored)
        VCMR: vid_name might be repeating.
        SVMR: vid_name is fixed to be the GT vid_name.
        VR: vid_name is not repeating, st and ed will not be used.

    Args:
        video2idx: {vid_name (str): index (int), ...}
        moment_predictions: list(dict), each dict is {
            "desc": str,
            "desc_id": int,
            "predictions": [vid_name_idx (int), st (float), ed (float), score (float)] * n_pred,
                sorted predictions, n_pred could be different for all dicts. For each prediction,
                only the first 3 elements [vid_name (str), st (float), ed (float),] are used,
                any other following elements are ignored. We leave score here for record.
        }
        ground_truth: list(dict), each dict is {
            "desc": str,
            "desc_id": int,
            "type": str, one of [v, t, vt]
            "vid_name": str
            "ts": [st (float), ed (float)], or list([st (float), ed (float)]), len == 4.
            ...
        }
        iou_thds: temporal IoU thresholds
        recall_topks: recall at different top k
        task_type: str, could be: ["VCMR", "SVMR", "VR"], see TASK_TYPES for definition.
        max_pred_per_query: int, only top max_pred_per_query predictions for each query are used.
        match_number: bool, must set to True if when do evaluation, False is only used for debug.
        verbose:
        use_desc_type: only TVR has desc type
    Returns:

    """
    assert task_type in TASK_TYPES, "task_type must be one of {}".format(list(TASK_TYPES.keys()))
    if verbose:
        print("Running evaluation with task_type {}, n results {}; n gt {}"
              .format(task_type, len(moment_predictions), len(ground_truth)))

    predictions_by_desc_id = {e["desc_id"]: e for e in moment_predictions}
    gt_by_desc_id = {e["desc_id"]: e for e in ground_truth}
    desc_type2idx = {"v": 0, "t": 1, "vt": 2}
    desc_types = []  # n_desc

    if match_number:
        assert set(gt_by_desc_id.keys()) == set(predictions_by_desc_id.keys()), \
            "desc_ids in predictions and ground_truth must match"
    # assert len(set([len(e["predictions"]) for e in predictions_by_desc_id.values()])) == 1, \
    #     "all queries must have the same number of predictions"

    pred_info_matrix_collection = []
    for k, gt_item in tqdm(gt_by_desc_id.items(), desc="Loop over moments", leave=False):
        if not match_number and k not in predictions_by_desc_id:
            continue
        pred_info_matrix = np.array(
            [e[:3] for e in predictions_by_desc_id[k]["predictions"]][:max_pred_per_query],
            dtype=np.float32)  # (n_pred, 3)
        if use_desc_type:
            desc_types.append(desc_type2idx[gt_item["type"]])
        vid_name_matched_pred = pred_info_matrix[:, 0] == video2idx[gt_item["vid_name"]]  # bool, (n_pred, )
        pred_info_matrix = np.concatenate([pred_info_matrix, vid_name_matched_pred[:, None]], axis=1)  # (n_pred, 4)

        # add 1 + len(iou_thds) columns, iou_scores, iou_corrects for each iou_thd.
        if "ts" in gt_item:
            iou_thd_corrects_columns = []
            if len(gt_item["ts"]) >= 4:  # didemo, fro all 3 splits, at least 4 ts for each, < 0.5% has more than 4.
                least_n_overlap = 2  # True if overlapped with at least least_n_overlap GT ts.
                iou_corrects_dict = defaultdict(list)
                for single_gt_ts in gt_item["ts"]:
                    single_gt_ts = np.array(single_gt_ts, dtype=np.float32)  # (2, )
                    # iou scores of the predictions that have wrong vid_name are set to 0.
                    iou_scores = compute_temporal_iou_batch(pred_info_matrix[:, 1:3], single_gt_ts) * vid_name_matched_pred
                    for iou_thd in iou_thds:
                        iou_corrects_dict[iou_thd].append(iou_scores >= iou_thd)
                for iou_thd in iou_thds:
                    iou_corrects = sum(iou_corrects_dict[iou_thd]) >= least_n_overlap  # bool, (n_pred, )
                    iou_thd_corrects_columns.append(iou_corrects[:, None])

            else:  # should be 2, len([st, ed]) == 2
                single_gt_ts = np.array(gt_item["ts"], dtype=np.float32)  # (2, )
                # iou scores of the predictions that have wrong vid_name are set to 0.
                iou_scores = compute_temporal_iou_batch(pred_info_matrix[:, 1:3], single_gt_ts) * vid_name_matched_pred

                for iou_thd in iou_thds:
                    iou_corrects = iou_scores >= iou_thd  # bool, (n_pred, )
                    iou_thd_corrects_columns.append(iou_corrects[:, None])

            pred_info_matrix = np.concatenate([pred_info_matrix, ] + iou_thd_corrects_columns, axis=1)  # (n_pred, 6)
        pred_info_matrix_collection.append(pred_info_matrix)

    # column header [vid_name_idx (int), st (float), ed (float), is_vid_name_match (bool),
    # iou_scores>=iou_thd0 (bool), iou_scores>=iou_thd1 (bool)]
    pred_info_matrix_collection = pad_sequences_1d_np(pred_info_matrix_collection)[0]  # (n_desc, n_pred, 6)
    if use_desc_type:
        desc_types = np.array(desc_types)  # (n_desc)

    # results wrapper
    metrics = OrderedDict()
    metrics_by_type = OrderedDict()

    iou_c_offset = 4  # iou_corrects column index starts here
    if task_type == "VCMR":
        for iou_idx, iou_thd in enumerate(iou_thds):
            iou_corrects = pred_info_matrix_collection[:, :, iou_c_offset + iou_idx].astype(np.bool)  # (n_desc, n_pred)
            # 1) there might be more than one positive clip, so use `>= 1`
            for k in recall_topks:
                metrics["{}-r{}".format(iou_thd, k)] = \
                    get_rounded_percentage(np.mean(np.sum(iou_corrects[:, :k], axis=1) >= 1))
        if use_desc_type:
            for desc_type in desc_type2idx:
                type_corrects = desc_types == desc_type2idx[desc_type]  # (n_desc)
                n_desc_in_type = np.sum(type_corrects)  # (n_desc)
                for iou_idx, iou_thd in enumerate(iou_thds):
                    # (n_desc, n_pred)
                    iou_corrects = pred_info_matrix_collection[:, :, iou_c_offset + iou_idx].astype(np.bool)
                    for k in recall_topks:
                        metrics_by_type["{}-{}-r{}".format(desc_type, iou_thd, k)] = get_rounded_percentage(
                            1.0 * np.sum(np.logical_and(np.sum(iou_corrects[:, :k], axis=1) >= 1, type_corrects))
                            / n_desc_in_type
                        )
    elif task_type == "SVMR":
        vid_name_matched = pred_info_matrix_collection[:, :, 3].astype(np.bool)  # (n_desc, n_pred)
        n_desc = len(vid_name_matched)
        for iou_idx, iou_thd in enumerate(iou_thds):
            iou_corrects = pred_info_matrix_collection[:, :, iou_c_offset + iou_idx].astype(np.bool)  # (n_desc, n_pred)
            # 1) there might be more than one positive clip, so use `>= 1`
            for k in recall_topks:
                metrics["{}-r{}".format(iou_thd, k)] = get_rounded_percentage(np.mean(
                    [np.sum(iou_corrects[idx][vid_name_matched[idx]][:k]) >= 1 for idx in range(n_desc)]
                ))
        if use_desc_type:
            for desc_type in desc_type2idx:
                type_corrects = desc_types == desc_type2idx[desc_type]  # (n_desc)
                n_desc_in_type = np.sum(type_corrects)  # (n_desc)
                for iou_idx, iou_thd in enumerate(iou_thds):
                    # (n_desc, n_pred)
                    iou_corrects = pred_info_matrix_collection[:, :, iou_c_offset + iou_idx].astype(np.bool)
                    # 1) there might be more than one positive clip, so use `>= 1`
                    for k in recall_topks:
                        metrics_by_type["{}-{}-r{}".format(desc_type, iou_thd, k)] = get_rounded_percentage(
                            1.0 * np.sum([np.sum(iou_corrects[idx][vid_name_matched[idx]][:k]) >= 1 and type_corrects[idx]
                                         for idx in range(n_desc)])
                            / n_desc_in_type)

    elif task_type == "VR":
        vid_name_matched = pred_info_matrix_collection[:, :, 3].astype(np.bool)  # (n_desc, n_pred)
        for k in recall_topks:
            metrics["r{}".format(k)] = \
                get_rounded_percentage(np.mean(np.sum(vid_name_matched[:, :k], axis=1) >= 1))
        if use_desc_type:
            for desc_type in desc_type2idx:
                type_corrects = desc_types == desc_type2idx[desc_type]  # (n_desc)
                n_desc_in_type = np.sum(type_corrects)  # (n_desc)
                for k in recall_topks:
                    metrics_by_type["{}-r{}".format(desc_type, k)] = get_rounded_percentage(
                        1.0 * np.sum(np.logical_and(np.sum(vid_name_matched[:, :k], axis=1) >= 1, type_corrects))
                        / n_desc_in_type)
    else:
        raise ValueError("task_type wrong.")
    if use_desc_type:
        metrics_by_type["desc_type_ratio"] = "v {} t {} vt {}"\
            .format(*[get_rounded_percentage(1.0 * np.sum(desc_types == desc_type2idx[k]) / len(desc_types))
                      for k in ["v", "t", "vt"]])
    return metrics, metrics_by_type


def eval_retrieval(
        submission, ground_truth, iou_thds=(0.5, 0.7), verbose=True,
        match_number=True, use_desc_type=True):
    video2idx = submission["video2idx"]
    submitted_task_types = [k for k in TASK_TYPES if k in submission]
    if verbose:
        print("Evaluating for task {}".format(submitted_task_types))
    eval_metrics = OrderedDict()
    metrics_raw_dict = {}
    for task_type in submitted_task_types:
        metrics, metrics_by_type = eval_by_task_type(
            submission[task_type], video2idx, ground_truth,
            iou_thds=iou_thds, recall_topks=(1, 5, 10, 100),
            task_type=task_type, max_pred_per_query=100,
            match_number=match_number, verbose=verbose, use_desc_type=use_desc_type)
        metrics_raw_dict[task_type] = metrics
        metrics_raw_dict[task_type+"_by_type"] = metrics_by_type

    for task_type in submitted_task_types:
        eval_metrics[task_type] = metrics_raw_dict[task_type]
    if use_desc_type:
        for task_type in submitted_task_types:
            eval_metrics[task_type+"_by_type"] = metrics_raw_dict[task_type+"_by_type"]
    return eval_metrics


def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="TVR Evaluation Script")
    parser.add_argument("--submission_path", type=str, help="path to generated prediction file")
    parser.add_argument("--gt_path", type=str, help="path to GT file")
    parser.add_argument("--save_path", type=str, help="path to save the results")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission = load_json(args.submission_path)
    gt = load_jsonl(args.gt_path)
    results = eval_retrieval(submission, gt, iou_thds=(0.5, 0.7), verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    with open(args.save_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    eval_main()
