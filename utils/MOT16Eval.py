import configparser
from collections import OrderedDict
from pathlib import Path

import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
import glob
import motmetrics as mm
import pandas as pd

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.application_util import preprocessing as prep
from deep_sort.deep_sort.detection import Detection
from torchvision import transforms
from utils.general import Log


class MOTEvaluator(object):
    """
    Evaluator for MOT16, which takes in
     `Detection` and output track.txt for py-motmetrics to calculate MOTA
    """
    data_split = {
        "train": ("train", (
            "MOT16-02", "MOT16-04", "MOT16-05", "MOT16-11", "MOT16-13",
        )),
        "val": ("train", (
            "MOT16-09", "MOT16-10",
        )),
        "trainval": ("train", (
            "MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10",
            "MOT16-11", "MOT16-13",
        )),
        "test": ("test", (
            "MOT16-01", "MOT16-03", "MOT16-06", "MOT16-07", "MOT16-08",
            "MOT16-12", "MOT16-14",
        )),
    }

    def __init__(self, model, eval_type='val', MOT_path='./MOT16',
                 output_path='./predictions/indi0001',
                 min_confidence=0.3, nms_threshold=0.8, nn_budget=30,
                 max_cosine_distance=0.2, visualize='', trans=None, dataset='MOT16', device='cuda:0', show_progress=True):
        self.show_progress = show_progress
        self.device = device
        self.model = model
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.eval_type = eval_type
        self.data_path = MOT_path
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.visualize = visualize
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.tracker = None
        self.transforms = trans
        if self.transforms is None:  # default DeepSORT transforms
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.target_prefix, self.target_dirs = self.data_split[self.eval_type]  # test, ('MOT16-XX', 'MOT16-XX')
        self.target_paths = [os.path.join(self.data_path, self.target_prefix, _dir) for _dir in
                             self.target_dirs]  # ['./MOT16/train/MOT16-01']

    def process(self):
        self.model.eval()
        for target_path in self.target_paths:  # for each sequence
            seqname = os.path.basename(target_path)
            if self.visualize and self.visualize != 'window':
                fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
                seqinfo = self.get_seqinfo(target_path)
                writer = cv2.VideoWriter(os.path.join(self.output_path, seqname + '.mp4'), fourcc,
                                         int(seqinfo['frameRate']), (int(seqinfo['imWidth']), int(seqinfo['imHeight'])))
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
            self.tracker = Tracker(metric)  # init a tracker when handling a new sequence
            det_path = os.path.join(target_path, 'det', 'det.txt')
            assert os.path.exists(det_path)
            gt_dict = self.get_dict(det_path, self.min_confidence)

            img_path = os.path.join(target_path, 'img1')
            result_str = []
            _iterator = tqdm(sorted(os.listdir(img_path))) if self.show_progress else sorted(os.listdir(img_path))
            for img in _iterator:  # for each frame
                file_path = os.path.join(img_path, img)
                frame_id = int(os.path.splitext(img)[0])
                # frame = Image.open(file_path).convert("RGB")
                frame = cv2.imread(file_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections, out_scores = self.get_gt(frame_id, gt_dict)
                if detections is None:
                    continue
                detections = np.array(detections)
                out_scores = np.array(out_scores)
                tracker, detections_class = self.run_deep_sort(frame, out_scores, detections)
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                    id_num = str(track.track_id)  # Get the ID for the particular track.
                    if self.visualize:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (255, 255, 255), 2)  # white -> tracking bbox
                        cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

                    result_str.append("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" %
                                      (frame_id, int(id_num), bbox[0], bbox[1], bbox[2] - bbox[0],
                                       bbox[3] - bbox[1]))
                if self.visualize:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    for det in detections_class:
                        bbox = det.to_tlbr()
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (255, 255, 0), 2)  # blue -> detection bbox
                    if self.visualize == 'window':
                        cv2.imshow('frame', frame)
                        cv2.waitKey(1)
                    else:
                        writer.write(frame)

            with open(os.path.join(self.output_path, seqname + '.txt'), 'w') as f:
                f.write('\n'.join(result_str))
            if self.visualize and self.visualize != 'window':
                writer.release()
            cv2.destroyAllWindows()

    def get_model_metric(self, data_format='mot16', save_path=None):
        def compare_dataframes(gts, ts):
            """Builds accumulator for each sequence."""
            accs = []
            names = []
            for k, tsacc in ts.items():
                if k in gts:
                    Log.info('Comparing %s...' % k)
                    accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
                    names.append(k)
                else:
                    Log.warn('No ground truth for %s, skipping.' % k)

            return accs, names

        """
        submit model's track.txt to py-motmetrics
        :return:
        """
        gtfiles = [os.path.join(_path, 'gt', 'gt.txt') for _path in self.target_paths]
        testfiles = [_f for _f in glob.glob(os.path.join(self.output_path, '*.txt'))]
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=data_format, min_confidence=1)) for f in gtfiles])
        ts = OrderedDict(
            [(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=data_format)) for f in testfiles])
        mh = mm.metrics.create()
        accs, names = compare_dataframes(gt, ts)
        metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                   'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
        Log.info('Running metrics...')
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
        if save_path is not None:
            writer = pd.ExcelWriter(save_path)
            summary.to_excel(writer)
            writer.save()
        return summary

    def pre_process(self, frame, detections):
        crops = []
        for d in detections:
            for i in range(len(d)):
                if d[i] < 0:
                    d[i] = 0
            img_h, img_w, img_ch = frame.shape

            xmin, ymin, w, h = d

            if xmin > img_w:
                xmin = img_w

            if ymin > img_h:
                ymin = img_h

            xmax = xmin + w
            ymax = ymin + h

            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))
            try:
                crop = frame[ymin:ymax, xmin:xmax, :]
                crop = self.transforms(crop)
                crops.append(crop)
            except Exception as e:
                print(e)
        crops = torch.stack(crops)
        return crops

    def run_deep_sort(self, frame, out_scores, out_boxes):
        if out_boxes == []:
            self.tracker.predict()
            print('No detections')
            trackers = self.tracker.tracks
            return trackers
        detections = np.array(out_boxes)
        processed_crops = self.pre_process(frame, detections).to(self.device)
        # processed_crops = self.gaussian_mask * processed_crops
        with torch.no_grad():
            features = self.model.forward_once(processed_crops)
        features = features.detach().cpu().numpy()

        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        dets = [Detection(bbox, score, feature) \
                for bbox, score, feature in \
                zip(detections, out_scores, features)]

        outboxes = np.array([d.tlwh for d in dets])

        outscores = np.array([d.confidence for d in dets])
        indices = prep.non_max_suppression(outboxes, self.nms_threshold, outscores)

        dets = [dets[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker, dets

    @staticmethod
    def get_dict(file_path, min_conf):
        with open(file_path) as f:
            d = f.readlines()
        d = list(map(lambda x: x.strip(), d))
        last_frame = int(d[-1].split(',')[0])
        gt_dict = {x: [] for x in range(last_frame + 1)}
        for i in range(len(d)):
            a = list(d[i].split(','))
            a = list(map(float, a[:-1]))
            coords = a[2:6]
            confidence = a[6]
            if confidence < min_conf:
                continue
            a[0] = int(a[0])
            gt_dict[a[0]].append({'coords': coords, 'conf': confidence})
        return gt_dict

    @staticmethod
    def get_gt(frame_id, gt_dict):
        if frame_id not in gt_dict.keys() or gt_dict[frame_id] == []:
            return None, None
        frame_info = gt_dict[frame_id]

        detections = []
        out_scores = []
        for i in range(len(frame_info)):
            coords = frame_info[i]['coords']
            x1, y1, w, h = coords
            x2 = x1 + w
            y2 = y1 + h
            detections.append([x1, y1, w, h])
            out_scores.append(frame_info[i]['conf'])
        return detections, out_scores

    @staticmethod
    def get_seqinfo(path):
        config = configparser.ConfigParser()
        config.optionxform = str
        assert os.path.exists(os.path.join(path, "seqinfo.ini"))
        config.read(os.path.join(path, "seqinfo.ini"))
        return dict(config["Sequence"])


if __name__ == '__main__':
    pass