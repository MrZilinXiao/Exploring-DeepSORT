import argparse
import traceback

from utils.MOT16Eval import MOTEvaluator
from utils.general import Log, GPUManager, ExpLogger, choose_device, Timer
from models.multiexp import ModelBuilder
from typing import List, Dict, Union, Tuple
from utils.dataset import MOTSplitDataset, RandomIdentitySampler
from utils.mymetric import evaluate
from utils.reid_metric import CMC_mAP_calculator
from utils.parallel import MyPool
from torch.utils.data import DataLoader
from tricks.warmup import WarmupStepLR
from tricks.loss import build_loss
from tensorboardX import SummaryWriter
from multiprocessing import Manager
import os
import glob
import yaml
import torch
import time
import numpy as np
from tqdm import tqdm
import copy
from torchvision import transforms


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default='./experiments', type=str)
    parser.add_argument('--gpu', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--mot_eval_interval', type=int, default=5)
    return parser.parse_args()


class ExperimentsLoader(object):
    def __init__(self, exp_dir: str):
        if os.path.isfile(exp_dir):  # single GPU
            self.exp_config = [(os.path.basename(exp_dir).split('.')[0], yaml.load(open(exp_dir), Loader=yaml.Loader))]
            return
        yml_list: List[str] = glob.glob(os.path.join(exp_dir, '*.yml'))
        self.exp_config: List[Tuple[str, Dict[str, Union[str, tuple, bool]]]] = [
            (os.path.basename(_path).split('.')[0], yaml.load(open(_path), Loader=yaml.Loader)) for
            _path in yml_list]  # [(exp_name, yml_path), ...]

    def __repr__(self):
        _info = "Load all %d experiments: " % len(self.exp_config)
        _info += '\n'.join([_tuple[0] for _tuple in self.exp_config])
        return _info


class ExperimentTrainer(object):
    def __init__(self, exp_name: str, exp_dict: dict, device, print_interval=20):
        self.exp_name = exp_name
        self.logger = ExpLogger(self.exp_name, file=True)
        self.exp_dict = exp_dict
        self.device: torch.device = device
        self.input_size: Tuple[int, int] = exp_dict['input_size']
        self.print_interval = print_interval
        data_type: str = exp_dict['dataset']
        batch_size: int = exp_dict['general']['batch_size']
        num_workers: int = exp_dict['general']['num_workers']

        self.max_epoch: int = exp_dict['general']['max_epoch']  # init optimizer & its scheduler
        self._init_dataset(data_type, batch_size, num_workers)  # init Dataset & DataLoader
        self.model = ModelBuilder(exp_dict, num_classes=len(self.dataset.class_to_idx))
        self.model.to(device)
        self._init_loss(exp_dict)  # init loss
        self._init_optim(exp_dict)  # init optimizer after init loss
        self.tot_iter = 0

        self.writer = SummaryWriter(comment=self.exp_name)
        self.evaluator = CMC_mAP_calculator(num_query=len(self.dataset.reid_set.query_list), L2_norm=True,
                                            metric='cosine')

    def _init_loss(self, exp_dict):
        self.criterion, self.center_criterion = build_loss(exp_dict, num_classes=len(self.dataset.class_to_idx),
                                                           in_feat_size=self.model.in_feat_size, device=self.device)
        if self.center_criterion is not None:
            self.center_weight: float = exp_dict['loss']['center_weight']

    def _init_dataset(self, data_type, batch_size, num_workers):
        self.dataset = MOTSplitDataset(self.exp_dict, data_type)
        train_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'drop_last': True
        }
        if self.exp_dict['sampler'] == 'triplet':
            train_kwargs['sampler'] = RandomIdentitySampler(data_list=[(_path, label, 1) for (_path, label)
                                                                       in zip(self.dataset.train_set.path_list,
                                                                              self.dataset.train_set.label_list)],
                                                            batch_size=batch_size, num_instances=4)
        else:
            train_kwargs['shuffle'] = True
        self.train_loader = DataLoader(self.dataset.train_set, **train_kwargs)
        # self.val_loader = DataLoader(self.dataset.val_set, batch_size=batch_size,
        #                              shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn)
        self.reid_loader = DataLoader(self.dataset.reid_set, batch_size=batch_size, shuffle=False, drop_last=False)

    def _init_optim(self, exp_dict):
        optim_type: str = exp_dict['general']['optim']
        base_lr, base_weight_decay = float(exp_dict['general']['lr']), float(exp_dict['general']['weight_decay'])
        center_lr = exp_dict['general']['center_lr']
        bias_lr_factor, bias_weight_decay = exp_dict['general']['bias_lr_factor'], exp_dict['general'][
            'bias_weight_decay']
        optim_kwargs = {'lr': base_lr, 'weight_decay': base_weight_decay}
        if optim_type == 'SGD':
            optim_kwargs['momentum'] = exp_dict['general']['momentum']
        else:
            if exp_dict['general']['momentum'] > 0:
                self.logger.warning("Optimizer type is not SGD but a valid momentum is given!")
        params = []
        for k, v in self.model.named_parameters():
            if not v.requires_grad:  # skip those frozen params
                continue
            lr = base_lr
            weight_decay = base_weight_decay
            if 'bias' in k:
                lr = lr * bias_lr_factor
                weight_decay = bias_weight_decay
            params += [{'params': [v], 'lr': lr, 'weight_decay': weight_decay}]
        self.optimizer = getattr(torch.optim, exp_dict['general']['optim'])(params, **optim_kwargs)
        self.center_optimizer = None

        if self.center_criterion is not None:
            self.center_optimizer = torch.optim.SGD(self.center_criterion.parameters(), lr=center_lr)
        self.lr_scheduler = None
        if exp_dict['warmup']['enable']:
            self.lr_scheduler = WarmupStepLR(self.optimizer, milestones=exp_dict['warmup']['steps'],
                                             base_factor=eval(exp_dict['warmup']['factor']),
                                             warmup_method=exp_dict['warmup']['method'],
                                             warmup_epoch=exp_dict['warmup']['max_epoch'])

    def print_details(self):
        _str = "Trainer initialized on {} with following details:\n{}".format(self.device, yaml.dump(self.exp_dict))
        self.logger.info(_str)

    def train(self, epoch):
        self.model.train()
        iter_train_loss = 0.0  # iter training loss, will be reset after training on `interval` mini-batch
        iter_center_loss = 0.0
        train_loss = 0.0  # total training loss, useful for calculating epoch train loss
        center_loss = 0.0
        correct, total = 0, 0  # correct / total sample count
        st_time = time.time()
        current_lr = self.exp_dict['general']['lr']
        if self.lr_scheduler is not None:
            current_lr = self.lr_scheduler.get_lr()[0]
        _log = "[%s][epoch %d/%d]Train #%d with lr %.8f:" % (self.exp_name, epoch, self.max_epoch, epoch, current_lr)
        if self.lr_scheduler is not None:
            _log += " (warmup enable)"
        self.logger.info(_log)
        for _iter, batch in enumerate(self.train_loader):
            self.tot_iter += 1
            # type: imgs: torch.Tensor; labels: torch.Tensor
            self.optimizer.zero_grad()
            if self.center_optimizer is not None:
                self.center_optimizer.zero_grad()
            imgs, labels = batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            score, feat = self.model(imgs)
            loss = self.criterion(score, feat, labels)  # Done
            loss.backward()
            self.optimizer.step()
            if self.center_criterion is not None:
                c_loss = self.center_criterion(feat, labels)
                center_loss += c_loss.item()
                iter_center_loss += c_loss.item()
                for param in self.center_criterion.parameters():
                    param.grad.data *= (1. / self.center_weight)
                self.center_optimizer.step()

            # evaluation in each iter
            correct += score.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
            # don't evaluate classification mAP in training...
            iter_train_loss += loss.item()
            train_loss += loss.item()
            if (_iter + 1) % self.print_interval == 0:
                self.writer.add_scalar('train_acc@top1', correct / total, self.tot_iter)
                self.writer.add_scalar('train_loss', iter_train_loss / self.print_interval, self.tot_iter)
                _log = "[{}][epoch {}/{}][progress:{:.1f}%]time:{:.2f}s Correct:{}/{} Acc:{:.3f}% Loss:{:.5f}".format(
                    self.exp_name, epoch, self.max_epoch,
                    100. * (_iter + 1) / len(self.train_loader), time.time() - st_time,
                    correct, total,
                    100. * correct / total, iter_train_loss / self.print_interval,
                )
                if iter_center_loss != 0.0:
                    _log += " Center Loss:{:.5f}".format(iter_center_loss / self.print_interval)
                self.logger.info(_log)
                iter_train_loss = 0.0
                iter_center_loss = 0.0
                st_time = time.time()
        _log = "[{}][epoch {}/{}]Train done with loss: {:.5f}".format(self.exp_name, epoch, self.max_epoch,
                                                                      train_loss / len(self.train_loader))
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.center_criterion is not None:
            _log += ", center loss: {:5f}".format(center_loss / len(self.train_loader))
        self.logger.info(_log)

    def eval(self, epoch):
        self.logger.info("[%s][epoch %d/%d]ReID Eval #%d: " % (self.exp_name, epoch, self.max_epoch, epoch))
        self.model.eval()
        query_feat = []
        query_labels = []
        gallery_feat = []
        gallery_labels = []
        with torch.no_grad():
            _iterator = tqdm(self.reid_loader) if len(
                options.gpu) == 1 else self.reid_loader  # disable tqdm when using multiple GPU
            for batch in _iterator:
                imgs, labels, is_query = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                is_query = is_query.numpy()
                feat = self.model.forward(imgs, add_logits=False)
                query_idx = np.where(is_query == True)[0].tolist()

                # for i in query_idx:
                #     query_feat.append(feat[i])
                #     query_labels.append(labels[i].tolist())
                # for j in list(set(range(len(labels))) - set(query_idx)):
                #     gallery_feat.append(feat[j].tolist())
                #     gallery_labels.append(labels[j].tolist())
                for i in query_idx:  # TODO: Optimize via constructing proper dataloader
                    query_feat.append(feat[i])
                    query_labels.append(labels[i].tolist())
                for j in list(set(range(len(labels))) - set(query_idx)):
                    gallery_feat.append(feat[j])
                    gallery_labels.append(labels[j].tolist())

            feats = query_feat + gallery_feat  # first num_query -> query_feats
            labels = query_labels + gallery_labels
            assert len(query_feat) == self.evaluator.num_query
            self.evaluator.update(feats, np.array(labels))
            with Timer(verbose=True):  # TODO: this evaluator yields better result? But evaluation is too slow...
                CMC, mAP = self.evaluator.compute()

            if CMC[0] > self.best_r1[0]:
                self.best_r1 = (CMC[0], epoch)
            if mAP > self.best_mAP[0]:
                self.best_mAP = (mAP, epoch)

            feat_size = query_feat[0].shape[0]
            with Timer(verbose=True):
                CMC_2, mAP_2 = evaluate(query_feat, np.array(query_labels),
                                        torch.cat(gallery_feat).view(-1, feat_size).to(self.device),
                                        np.array(gallery_labels))

            self.logger.info('[%s][epoch %d/%d]ReID Test Result: Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (
                self.exp_name, epoch, self.max_epoch, CMC[0], CMC[4], CMC[9], mAP))
            self.logger.info('[%s][epoch %d/%d]ReID Test Result: Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (
                self.exp_name, epoch, self.max_epoch, CMC_2[0], CMC_2[4], CMC_2[9], mAP_2))
            for r in (1, 5, 10):
                self.writer.add_scalar('reid_cmc_rank%d' % r, CMC[r - 1], epoch)
            self.writer.add_scalar('reid_mAP', mAP, epoch)

    def eval_mot(self, epoch):
        self.logger.info('MOT Test:')
        mot_transforms: List = copy.deepcopy(self.dataset.reid_set.transforms.transforms)
        mot_transforms.insert(0, transforms.ToPILImage())  # insert a ToPILImage()
        trans = transforms.Compose(mot_transforms)
        evaluator = MOTEvaluator(model=self.model, MOT_path='./MOT16-new',
                                 output_path='./predictions-new',
                                 visualize='', eval_type='val',
                                 max_cosine_distance=0.2, trans=trans, device=self.device,
                                 show_progress=(len(options.gpu) == 1))
        evaluator.process()  # run DeepSORT
        summary = evaluator.get_model_metric(save_path='%s-%d.xls' % (self.exp_name, epoch))
        self.logger.info('[%s][epoch %d/%d]Overall MOTA: %.3f, FP: %d, FN: %d, IDsw: %d' % (
            self.exp_name, epoch, self.max_epoch, summary['mota'][-1],
            summary['num_false_positives'][-1],
            summary['num_misses'][-1],
            summary['num_switches'][-1]))
        self.writer.add_scalar('MOTA', summary['mota'][-1], epoch)
        self.writer.add_scalar('mot_FP', summary['num_false_positives'][-1], epoch)
        self.writer.add_scalar('mot_FN', summary['num_misses'][-1], epoch)
        self.writer.add_scalar('mot_idsw', summary['num_switches'][-1], epoch)

    def start(self):
        self.best_r1, self.best_mAP = (0., -1), (0., -1)
        self.print_details()
        self.logger.info("[{}] Eval before training...".format(self.exp_name))
        self.eval(0)
        self.eval_mot(0)
        for epoch in range(1, self.max_epoch + 1):
            self.train(epoch)
            self.eval(epoch)
            if epoch % options.mot_eval_interval == 0:
                self.eval_mot(epoch)
        self.logger.info("[{}] Train done with best R1 {} & mAP {}"
                         .format(self.exp_name, self.best_r1, self.best_mAP))


def submit(exp_name, exp_dict, q=None, gpu_device=None):
    if gpu_device is None and q is None:
        Log.warn("!!You should not see this unless one of submitted GPUs is taken!! No available GPU!")
        return
    if q is not None:
        gpu_id = q.get()
        gpu_device = choose_device(gpu_id)
    try:
        trainer = ExperimentTrainer(exp_name, exp_dict, gpu_device)
        trainer.start()
    except Exception as e:
        traceback.print_exc()
        Log.warn(str(e))
    if q is not None:
        q.put(gpu_id)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    options = get_options()
    manager = GPUManager()
    exp_loader = ExperimentsLoader(exp_dir=options.exp_dir)
    gpu_list = list(map(int, options.gpu))
    Log.info(str(exp_loader))  # exp_loader overall statistics
    time.sleep(3)

    if len(gpu_list) > 1:
        Log.info('Using Multiple GPUs, distribute jobs via multiprocess.Pool...')
        pool = MyPool(processes=len(gpu_list))
        m = Manager()
        gpu_queue = m.Queue(maxsize=len(gpu_list))
        for gpu_id in gpu_list:
            # type: gpu_id: int
            gpu_queue.put(gpu_id)

        for (exp_name, exp_dict) in exp_loader.exp_config:  # for each experiment
            pool.apply_async(submit, args=(exp_name, exp_dict, gpu_queue,))  # submit to a free GPU
            time.sleep(1)  # wait until `python` shown in `nvidia-smi`
        pool.close()
        pool.join()
    else:
        for (exp_name, exp_dict) in exp_loader.exp_config:  # for each experiment
            submit(exp_name, exp_dict, gpu_device='cuda:' + str(gpu_list[0]))  # submit to a free GPU
