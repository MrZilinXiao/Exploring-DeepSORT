import torch
import numpy as np
from utils.general import Log


def _euclidean_dist(x, y):
    """
    Compute euclidean distance between features in x and y
    :param x: torch.FloatTensor shape: [M, feat_size]  query here
    :param y: torch.FloatTensor shape: [N, feat_size]  gallery here
    :return: distmat, shape: [M, N]
    """
    m, n = x.shape[0], y.shape[0]
    distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, x, y.t())
    return distmat.cpu().numpy()


def _cosine_dist(x, y, norm=True):
    """
    Compute cosine distance betweeen features in x and y
    :param x: torch.FloatTensor shape: [M, feat_size]  query here
    :param y: torch.FloatTensor shape: [N, feat_size]  gallery here
    :return: distmat, shape: [M, N]
    """
    if norm:
        x = torch.nn.functional.normalize(x, dim=1, p=2)
        y = torch.nn.functional.normalize(y, dim=1, p=2)
    return 1. - torch.mm(x, y.t()).cpu().numpy()


def evaluate(distmat, query_labels, gallery_labels, max_rank=50):
    num_query, num_gallery = distmat.shape
    if num_gallery < max_rank:
        Log.warn("Too small gallery size %d, smaller than max_rank %d" % (num_gallery, max_rank))
        max_rank = num_gallery
    indices = np.argsort(distmat, axis=1)  # get sorted distance index, asec order
    matches = (gallery_labels[indices] == query_labels[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_AP = []
    num_valid_query = 0.

    for query_id in range(num_query):
        # query_label = query_labels[query_id]

        # order = indices[query_id]
        # keep =
        # no need to remove gallery since we set cam_id to fixed
        ori_cmc = matches[query_id]
        if not np.any(ori_cmc):
            # no query identity appears in gallery identities
            continue
        cmc = ori_cmc.cumsum()
        cmc[cmc > 1] = 1  # threshold maximum
        all_cmc.append(cmc[:max_rank])  # add `max_rank` cmc

        num_valid_query += 1.

        num_rel = ori_cmc.sum()
        tmp_cmc = ori_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * ori_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_query > 0, "All query identities not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_query
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class CMC_mAP_calculator(object):
    def __init__(self, num_query, max_rank=50, L2_norm=True, metric='euclidean'):
        """
        :param num_query: Number of queries
        :param max_rank: Calculate how many ranks when computing CMC
        :param L2_norm: (bool) use L2 normalization on all features
        """
        self.num_query = num_query
        self.max_rank = max_rank
        self.L2_norm = L2_norm
        self.feats = []  # self.feats[:num_query] -> query_feats
        self.pids = []
        self.metric = metric
        if metric == 'euclidean':
            self._metric = _euclidean_dist
        elif metric == 'cosine':
            self._metric = _cosine_dist
        else:
            raise NotImplementedError
        Log.info(str(self))

    def reset(self):
        self.feats = []
        self.pids = []

    def update(self, feats, labels):
        """
        update to latest feats & labels
        where the first `num_query` feats & labels belong to
        :param feats: List[torch.FloatTensor]
        :param labels: np.array
        :return:
        """
        self.feats = feats
        self.pids = labels

    def compute(self):
        feat_size = self.feats[0].shape[0]
        feats = torch.cat(self.feats, dim=0).view(-1, feat_size)
        if self.L2_norm:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        query_feats = feats[: self.num_query]
        query_labels = self.pids[: self.num_query]
        gallery_feats = feats[self.num_query:]
        gallery_labels = self.pids[self.num_query:]
        distmat = self._metric(query_feats, gallery_feats)

        cmc, mAP = evaluate(distmat, query_labels, gallery_labels)
        return cmc, mAP

    def __repr__(self):
        _info = "Using %s metric ReID evaluator with L2_norm %s, num_query: %d, max_rank: %d" % \
                (self.metric, "enabled" if self.L2_norm else "disabled", self.num_query, self.max_rank)
        return _info


if __name__ == '__main__':
    # feats = [torch.tensor([1, 1, 1], dtype=torch.float), torch.tensor([0, 0, 0], dtype=torch.float),
    #          torch.tensor([1, 1, 1], dtype=torch.float), torch.tensor([0.5, 0.5, 0.6], dtype=torch.float),
    #          torch.tensor([0.7, 0.7, 0.9], dtype=torch.float)]
    feats = [torch.randn(100) for _ in range(100)]
    labels = np.array([np.random.randint(100) for _ in range(100)])
    # test num_query = 1
    # labels = np.array([1, 0, 1, 2, 3])
    evaluator = CMC_mAP_calculator(num_query=10, max_rank=100)
    evaluator.update(feats, labels)
    print(evaluator.compute())
    # right answer

    from utils.mymetric import evaluate

    print(evaluate(torch.cat(feats[:10]).view(-1, 100), np.array(labels[:10]), torch.cat(feats[10:]).view(-1, 100), np.array(labels[10:])))

    # query_feats = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float)
    # query_labels = np.array([1])
    # gallery_feats = torch.tensor([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]], dtype=torch.float)
    # gallery_labels = np.array([0, 1, 2, 3])
    # print(_cosine_dist(query_feats, gallery_feats))
    # print(_euclidean_dist(query_feats, gallery_feats))
    # use F.cosine_similarity to validate
    # print(1. - torch.nn.functional.cosine_similarity(query_feats, gallery_feats))
    # for i in range(gallery_feats.shape[0]):
    #     print(torch.nn.functional.cosine_similarity(query_feats[0], gallery_feats[i]))
