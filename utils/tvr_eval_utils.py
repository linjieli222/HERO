'''
TVR Evaluation util functions copied from TVRetrieval implementation
(https://github.com/jayleicn/TVRetrieval)
'''
import numpy as np
from collections import defaultdict


"""
Non-Maximum Suppression for video proposals.
"""


def compute_temporal_iou(pred, gt):
    """ deprecated due to performance concerns
    compute intersection-over-union along temporal axis
    Args:
        pred: [st (float), ed (float)]
        gt: [st (float), ed (float)]
    Returns:
        iou (float):

    Ref: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    """
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = (
        max(pred[1], gt[1])
        - min(pred[0], gt[0]))  # not the correct union though
    if union == 0:
        return 0
    else:
        return 1.0 * intersection / union


def temporal_non_maximum_suppression(predictions, nms_threshold,
                                     max_after_nms=100):
    """
    Args:
        predictions:
            list(sublist), each sublist is
            [st (float), ed(float), score (float)],
            note larger scores are better and are preserved.
            For metrics that are better when smaller,
            please convert to its negative,
            e.g., convert distance to negative distance.
        nms_threshold: float in [0, 1]
        max_after_nms:
    Returns:
        predictions_after_nms:
        list(sublist),
        each sublist is [st (float), ed(float), score (float)]
    References:
        https://github.com/wzmsltw/BSN-boundary-sensitive-network/blob/7b101fc5978802aa3c95ba5779eb54151c6173c6/Post_processing.py#L42
    """
    if len(predictions) == 1:  # only has one prediction, no need for nms
        return predictions

    predictions = sorted(predictions, key=lambda x: x[2],
                         reverse=True)  # descending order

    tstart = [e[0] for e in predictions]
    tend = [e[1] for e in predictions]
    tscore = [e[2] for e in predictions]
    rstart = []
    rend = []
    rscore = []
    while len(tstart) > 1 and len(rscore) < max_after_nms:  # max 100 after nms
        idx = 1
        while idx < len(tstart):  # compare with every prediction in the list.
            if compute_temporal_iou(
                    [tstart[0], tend[0]],
                    [tstart[idx], tend[idx]]) > nms_threshold:
                # rm highly overlapped lower score entries.
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
            else:
                # move to next
                idx += 1
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    if (len(rscore) < max_after_nms
            and len(tstart) >= 1):  # add the last, possibly empty.
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    predictions_after_nms = [
        [st, ed, s] for s, st, ed in zip(rscore, rstart, rend)]
    return predictions_after_nms


def top_n_array_2d(array_2d, top_n):
    """
    Get topN indices and values of a 2d array,
    return a tuple of indices and their values,
    ranked by the value
    """
    row_indices, column_indices = np.unravel_index(
        np.argsort(array_2d, axis=None), array_2d.shape)
    row_indices = row_indices[::-1][:top_n]
    column_indices = column_indices[::-1][:top_n]
    sorted_values = array_2d[row_indices, column_indices]
    return np.stack([row_indices, column_indices, sorted_values],
                    axis=1)  # (N, 3)


def find_max_triples_from_upper_triangle_product(
        upper_product, top_n=5, prob_thd=None):
    """
    Find a list of (k1, k2) where k1 < k2
        with the maximum values of p1[k1] * p2[k2]
    Args:
        upper_product (torch.Tensor or np.ndarray): (N, L, L),
            the lower part becomes zeros, end_idx > start_idx
        top_n (int): return topN pairs with highest values
        prob_thd (float or None):
    Returns:
        batched_sorted_triple: N * [(st_idx, ed_idx, confidence), ...]
    """
    batched_sorted_triple = []
    for idx, e in enumerate(upper_product):
        sorted_triple = top_n_array_2d(e, top_n=top_n)
        if prob_thd is not None:
            sorted_triple = sorted_triple[sorted_triple[2] >= prob_thd]
        batched_sorted_triple.append(sorted_triple)
    return batched_sorted_triple


def filter_vcmr_by_nms(all_video_predictions, nms_threshold=0.6,
                       max_before_nms=1000, max_after_nms=100,
                       score_col_idx=3):
    """ Apply non-maximum suppression for all the predictions for each video.
    1) group predictions by video index
    2) apply nms individually for each video index group
    3) combine and sort the predictions
    Args:
        all_video_predictions: list(sublist),
            Each sublist is
            [video_idx (int), st (float), ed(float), score (float)]
            Note the scores are negative distances.
        nms_threshold: float
        max_before_nms: int
        max_after_nms: int
        score_col_idx: int
    Returns:

    """
    predictions_neg_by_video_group = defaultdict(list)
    for pred in all_video_predictions[:max_before_nms]:
        predictions_neg_by_video_group[pred[0]].append(
            pred[1:])  # [st (float), ed(float), score (float)]

    predictions_by_video_group_neg_after_nms = dict()
    for video_idx, grouped_preds in predictions_neg_by_video_group.items():
        predictions_by_video_group_neg_after_nms[video_idx] = \
            temporal_non_maximum_suppression(
                grouped_preds, nms_threshold=nms_threshold)

    predictions_after_nms = []
    for video_idx, grouped_preds in\
            predictions_by_video_group_neg_after_nms.items():
        for pred in grouped_preds:
            # [video_idx (int), st (float), ed(float), score (float)]
            pred = [video_idx] + pred
            predictions_after_nms.append(pred)

    # ranking happens across videos
    predictions_after_nms = sorted(
        predictions_after_nms,
        key=lambda x: x[score_col_idx],
        reverse=True)[:max_after_nms]  # descending order
    return predictions_after_nms


def get_submission_top_n(submission, top_n=100):
    def get_prediction_top_n(list_dict_predictions, top_n):
        top_n_res = []
        for e in list_dict_predictions:
            e["predictions"] = e["predictions"][:top_n]
            top_n_res.append(e)
        return top_n_res

    top_n_submission = dict(video2idx=submission["video2idx"], )
    for k in submission:
        if k != "video2idx":
            top_n_submission[k] = get_prediction_top_n(submission[k], top_n)
    return top_n_submission


def post_processing_vcmr_nms(
        vcmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    vcmr_res: list(dict), each dict is{
        "desc": str,
        "desc_id": int,
        "predictions": list(sublist), each sublist is
            [video_idx (int), st (float), ed(float), score (float)],
            video_idx could be different
    }
    """
    processed_vcmr_res = []
    for e in vcmr_res:
        e["predictions"] = filter_vcmr_by_nms(e["predictions"],
                                              nms_threshold=nms_thd,
                                              max_before_nms=max_before_nms,
                                              max_after_nms=max_after_nms)
        processed_vcmr_res.append(e)
    return processed_vcmr_res


def post_processing_svmr_nms(
        svmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    svmr_res: list(dict), each dict is
        {"desc": str,
         "desc_id": int,
         "predictions": list(sublist)  # each sublist is
            [video_idx (int), st (float), ed(float), score (float)],
            video_idx is the same.
         }
    """
    processed_svmr_res = []
    for e in svmr_res:
        # the predictions are sorted inside the nms func.
        _predictions = [d[1:] for d in e["predictions"][:max_before_nms]]
        _predictions = temporal_non_maximum_suppression(
            _predictions, nms_threshold=nms_thd)[:max_after_nms]
        _video_id = e["predictions"][0][0] # video_id is the same for all predictions
        e["predictions"] = [[_video_id, ] + d for d in _predictions]
        processed_svmr_res.append(e)
    return processed_svmr_res


def generate_min_max_length_mask(array_shape, min_l, max_l):
    """ The last two dimension denotes matrix of upper-triangle
        with upper-right corner masked, below is the case for 4x4.
    [[0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]

    Args:
        array_shape: np.shape??? The last two dimensions should be the same
        min_l: int, minimum length of predicted span
        max_l: int, maximum length of predicted span

    Returns:

    """
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    # (1, ..., 1, L, L)
    extra_length_mask_array = np.ones(mask_shape, dtype=np.float32)
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask  # with valid bit to be 1
