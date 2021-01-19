import numpy as np
import torch


def get_right_and_junk_index(query_label, gallery_labels, query_camera_label=None, gallery_camera_labels=None):
    same_label_index = np.argwhere(gallery_labels == query_label)
    if (query_camera_label is not None) and (gallery_camera_labels is not None):
        same_camera_label_index = np.argwhere(gallery_camera_labels == query_camera_label)
        # the index of mis-detected images, which contain the body parts.
        junk_index1 = np.argwhere(gallery_labels == -1)
        # find index that are both in query_index and camera_index
        # the index of the images, which are of the same identity in the same cameras.
        junk_index2 = np.intersect1d(same_label_index, same_camera_label_index)
        junk_index = np.append(junk_index2, junk_index1)

        # find index that in query_index but not in camera_index
        # which means the same lable but different camera
        right_index = np.setdiff1d(same_label_index, same_camera_label_index, assume_unique=True)
        return right_index, junk_index
    else:
        return same_label_index, None


def evaluate_with_index(sorted_similarity_index, right_result_index, junk_result_index=None):
    """calculate cmc curve and Average Precision for a single query with index
    :param sorted_similarity_index: index of all returned items. typically get with
        function `np.argsort(similarity)`
    :param right_result_index: index of right items. such as items in gallery
        that have the same id but different camera with query
    :param junk_result_index: index of junk items. such as items in gallery
        that have the same camera and id with query
    :return: single cmc, Average Precision
    """
    # initial a numpy array to store the AccK(like [0, 0, 0, 1, 1, ...,1]).
    cmc = np.zeros(len(sorted_similarity_index), dtype=np.int32)
    ap = 0.0

    if len(right_result_index) == 0:
        cmc[0] = -1
        return cmc, ap
    if junk_result_index is not None:
        # remove junk_index
        # all junk_result_index in sorted_similarity_index has been removed.
        # for example:
        # (sorted_similarity_index, junk_result_index)
        # ([3, 2, 0, 1, 4],         [0, 1])             -> [3, 2, 4]
        need_remove_mask = np.in1d(sorted_similarity_index, junk_result_index, invert=True)
        sorted_similarity_index = sorted_similarity_index[need_remove_mask]

    mask = np.in1d(sorted_similarity_index, right_result_index)
    right_index_location = np.argwhere(mask == True).flatten()

    # [0,0,0,...0, 1,1,1,...,1]
    #              |
    #  right answer first appearance
    cmc[right_index_location[0]:] = 1

    for i in range(len(right_result_index)):
        precision = float(i + 1) / (right_index_location[i] + 1)
        if right_index_location[i] != 0:
            # last rank precision, not last match precision
            old_precision = float(i) / (right_index_location[i])
        else:
            old_precision = 1.0
        ap = ap + (1.0 / len(right_result_index)) * (old_precision + precision) / 2

    return cmc, ap


def calculate_distance(query_feature, gallery_features, dist_type='cosine', norm=True):  # DONE: This is a similarity
    """calculate the distance between query and gallery
    :param gallery_features: the feature's list for gallery [M, feat_size]
    :param query_feature: the feature for query [feat_size]
    :return: similarity_distance, size = M
    """
    if norm:
        query_feature = torch.nn.functional.normalize(query_feature, p=2, dim=0)
        gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)
    if dist_type == 'cosine':
        return torch.mm(gallery_features, query_feature.view(-1, 1)).squeeze(1).cpu().numpy()
    else:
        raise NotImplementedError

def calculate_euclidean_dist(x, y):
    """
    calculate euclidean distance between query and gallery
    x: [M, feat_size]
    y: [N, feat_size]
    :return:
    """
    m, n = x.size(0), y.size(0)
    # y = y.view(-1, 1)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist.squeeze(0).cpu().numpy()


def _cosine_distance(a, b, norm=True):
    """
    Compute pair-wise cosine distance between points in `a` and `b`.
    """
    # return 1. - np.dot(a, b.T)
    if norm:
        a = torch.nn.functional.normalize(a, dim=1, p=2)
        b = torch.nn.functional.normalize(b, dim=1, p=2)
    return 1. - torch.mm(a, b.t())


def evaluate(query_features, query_labels,
             gallery_features, gallery_labels):
    """
    :param query_features: [N, feat_size], Union[np.ndarray, List[torch.FloatTensor]]
    :param query_labels: [N, 1]
    :param gallery_features: [M, feat_size],
    :param gallery_labels: [M, 1]
    :return:
    """
    total_cmc = np.zeros(len(gallery_labels), dtype=np.int32)
    total_average_precision = 0.0

    for i in range(len(query_labels)):  # for each query
        similarity_distance = calculate_distance(query_features[i], gallery_features)  # -> similarity
        # similarity_distance = calculate_euclidean_dist(query_features[i].view(1, -1), gallery_features)
        cmc, ap = evaluate_with_index(
            np.argsort(similarity_distance)[::-1],
            *get_right_and_junk_index(query_labels[i], gallery_labels)
        )  # for each class

        if cmc[0] == -1:
            continue
        total_cmc += cmc
        total_average_precision += ap

    return total_cmc.astype(np.float64) / len(query_labels), total_average_precision / len(query_labels)


# def main(opt):
#     result = scipy.io.loadmat('pytorch_result.mat')
#     if opt.GPU:
#         CMC, mAP = evaluate(torch.FloatTensor(result['query_f']).cuda(), result['query_label'][0], result['query_cam'][0],
#                             torch.FloatTensor(result['gallery_f']).cuda(), result['gallery_label'][0],
#                             result['gallery_cam'][0])
#     else:
#         CMC, mAP = evaluate(result['query_f'], result['query_label'][0], result['query_cam'][0],
#                             result['gallery_f'], result['gallery_label'][0], result['gallery_cam'][0])
#     print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], mAP))
if __name__ == '__main__':
    # Test calculate Euclidean distance
    query_feats = torch.tensor([[1, 1, 1]], dtype=torch.float)
    query_labels = np.array([1])
    gallery_feats = torch.tensor([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.6], [0.7, 0.7, 0.9]], dtype=torch.float)
    gallery_labels = np.array([0, 1, 2, 3])
    # print(_cosine_distance(query_feats, gallery_feats))
    # print(calculate_distance(query_feats, gallery_feats))

    cmc, mAP = evaluate(query_feats, query_labels, gallery_feats, gallery_labels)
    print(cmc, mAP)
    # query = torch.tensor([1, 1, 1], dtype=torch.float)  # [1, 3] 3: feat_size
    # gallery = torch.tensor([[1, 1, 1], [2, 2, 2], [0, 0, 0]], dtype=torch.float)
    # dist_mat = calculate_euclidean_dist(query.view(1, -1), gallery)  # [1, 3] [3, 3]
    # print(dist_mat)
    # dist_mat_2 = calculate_similarity_distance(query, gallery)
    # print(dist_mat_2)
