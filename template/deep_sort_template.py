import os
import warnings
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.functional import pad
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader
from torch import optim
import math
from utils.dataset import CropTrainDataset, CropTestDataset, MOTSplitDataset
import torchvision.datasets as datasets
from utils.general import Log, ExpLogger
from utils.metric import mAPMetric
from utils.MOT16Eval import MOTEvaluator
from utils.reid_metric import CMC_mAP_calculator
from torchvision import transforms


class _ConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode='zeros'):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv2d_with_padding(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_with_padding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    @staticmethod
    def conv2d_same_padding(input, weight, bias=None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
        # padding not relevant, just to be compatible with conv2d
        input_rows = input.size(2)
        filter_rows = weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        padding_cols = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        cols_odd = (padding_rows % 2 != 0)

        if rows_odd or cols_odd:
            input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, weight, bias, stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=dilation, groups=groups)

    def forward(self, input):
        return Conv2d_with_padding.conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                                       self.padding, self.dilation, self.groups)


class ExploringNet(nn.Module):
    feature_dim = 128

    def __init__(self, num_classes=1000, reid=True):
        super(ExploringNet, self).__init__()
        self.reid = reid
        self.num_classes = num_classes
        #generated_init
        # ConvPart, Linear should be generated via CNN-GA approach
        self.gap = nn.AdaptiveAvgPool2d(1)
        if self.reid:
            self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, input, add_logits=True):
        #generated_forward
        # generate: out = self.gap(self.net(input)) -> (N, last_out_channel, 1, 1)
        # generate: out = out.view(out.shape[0], -1) -> (N, last_out_channel)
        # generate: out = self.linear(out) -> (N, feat_size)
        out = F.normalize(out, p=2, dim=1)
        if self.reid and add_logits:
            logits = self.classifier(out)
            return out, logits
        else:
            return out

    def forward_once(self, input):
        return self.forward(input, add_logits=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.same_shape = in_channels == out_channels
        if not self.same_shape:
            stride = 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x, inplace=True)


class TrainModel(object):
    stats_iter_frequency = 10
    test_epoch_frequency = 1
    mot_frequency = 5
    not_changing_ratio = 0.01
    not_changing_limit = 2
    lr_decay_checkpoints = [5, 10, 15]
    lr_decay_step = 5e-1

    def __init__(self, img_path='./MOT16Cropped', gpu_id: str = None, batch_size: int = 50, num_workers: int = 12,
                 max_epoch: int = 30, exp_config='./experiments/cnn_ga.yml'):
        # train_dataset = datasets.ImageFolder(root=img_path)
        # X, Y = zip(*train_dataset.imgs)
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=233, stratify=Y)
        self.gpu_id = gpu_id
        self.device = torch.device('cuda:0')  # fixed on cuda:0
        self.max_epoch = max_epoch
        exp_dict = yaml.load(open(exp_config), Loader=yaml.Loader)
        self.dataset = MOTSplitDataset(exp_dict, 'MOT16')
        self.train_dataloader = DataLoader(self.dataset.train_set, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, drop_last=True)
        self.reid_dataloader = DataLoader(self.dataset.reid_set, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers, drop_last=False)
        # self.train_dataset = CropTrainDataset(X_train, Y_train)
        # self.test_dataset = CropTestDataset(X_test, Y_test)
        # self.train_dataloader = DataLoader(self.train_dataset,
        #                                    batch_size=batch_size,
        #                                    shuffle=True, num_workers=num_workers,
        #                                    drop_last=True)
        # self.test_dataloader = DataLoader(self.test_dataset,
        #                                   batch_size=batch_size,
        #                                   shuffle=False, num_workers=num_workers,
        #                                   drop_last=False)
        # self.mAP_validator = mAPMetric()
        # self.model = CosineMetricNet(num_classes=len(train_dataset.class_to_idx)).cuda()
        self.model = ExploringNet(num_classes=len(self.dataset.class_to_idx)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.module_name = str(os.path.basename(__file__)).split('.')[0]  # like `indi0001`
        self.writer = SummaryWriter(comment=self.module_name)  # use module name as log_dir
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.tot_iter = 0
        # self.best_metric = 0.0
        # self.map_validator = mAPMetric()
        self.metric_list = []
        self.not_changing_cnt = 0
        self.not_changing_solver = lambda _metric: math.fabs(_metric - self.metric_list[-1]) / self.metric_list[
            -1] < self.not_changing_ratio

        self.evaluator = CMC_mAP_calculator(num_query=len(self.dataset.reid_set.query_list), L2_norm=True,
                                            metric='cosine')
        self.logger = ExpLogger(self.module_name, file=True)
        Log.info('Individual {} init successfully! Start training...'.format(self.module_name))

    def train(self, epoch: int):
        self.model.train()
        iter_train_loss = 0.0  # iter training loss, will be reset after training on `interval` mini-batch
        # train_loss = 0.0  # total training loss, useful for calculating epoch train loss
        correct, total = 0, 0  # correct / total sample count
        st_time = time.time()
        for j, batched_data in enumerate(self.train_dataloader):
            img, img_labels = batched_data
            img, img_labels = Variable(img).to(self.device), Variable(img_labels).to(self.device)
            _, logits = self.model(img)  # feature: [N, 128]   logits: [N, num_classes]

            loss = self.criterion(logits, img_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.tot_iter += 1
            # accumulating
            iter_train_loss += loss.item()
            correct += logits.max(dim=1)[1].eq(img_labels).sum().item()
            total += img_labels.size(0)
            if (j + 1) % self.stats_iter_frequency == 0:
                end_time = time.time()
                self.logger.info(
                    "[{} on GPU {}][epoch {}][progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                        self.module_name, str(self.gpu_id), str(epoch),
                        100. * (j + 1) / len(self.train_dataloader), end_time - st_time,
                        iter_train_loss / self.stats_iter_frequency, correct, total,
                        100. * correct / total
                    ))

                self.writer.add_scalar('training_cosine_loss', iter_train_loss / self.stats_iter_frequency,
                                       self.tot_iter)
                self.writer.add_scalar('train_acc', 100. * correct / total, self.tot_iter)
                st_time = time.time()
                iter_train_loss = 0.0

    def eval(self, epoch: int):
        self.logger.info("[%s][epoch %d/%d]ReID Eval #%d: " % (self.module_name, epoch, self.max_epoch, epoch))
        self.model.eval()
        query_feat = []
        query_labels = []
        gallery_feat = []
        gallery_labels = []
        with torch.no_grad():
            for batch in self.reid_dataloader:
                imgs, labels, is_query = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                is_query = is_query.numpy()
                feat = self.model.forward(imgs, add_logits=False)
                query_idx = np.where(is_query == True)[0].tolist()
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
            CMC, mAP = self.evaluator.compute()
            if CMC[0] > self.best_r1[0]:
                self.best_r1 = (CMC[0], epoch)
            if mAP > self.best_mAP[0]:
                self.best_mAP = (mAP, epoch)

            self.logger.info('[%s][epoch %d/%d]ReID Test Result: Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (
                self.module_name, epoch, self.max_epoch, CMC[0], CMC[4], CMC[9], mAP))
            for r in (1, 5, 10):
                self.writer.add_scalar('reid_cmc_rank%d' % r, CMC[r - 1], epoch)
            self.writer.add_scalar('reid_mAP', mAP, epoch)
        return mAP

    def eval_mot(self, epoch):
        self.logger.info('MOT Test:')
        mot_transforms: list = copy.deepcopy(self.dataset.reid_set.transforms.transforms)
        mot_transforms.insert(0, transforms.ToPILImage())  # insert a ToPILImage()
        trans = transforms.Compose(mot_transforms)
        evaluator = MOTEvaluator(model=self.model, MOT_path='./MOT16-new',
                                 output_path='./predictions-new',
                                 visualize='', eval_type='val',
                                 max_cosine_distance=0.2, trans=trans, device=self.device,
                                 show_progress=False)
        evaluator.process()
        summary = evaluator.get_model_metric(save_path='%s-%d.xls' % (self.module_name, epoch))
        self.logger.info('[%s][epoch %d/%d]Overall MOTA: %.3f, FP: %d, FN: %d, IDsw: %d' % (
            self.module_name, epoch, self.max_epoch, summary['mota'][-1],
            summary['num_false_positives'][-1],
            summary['num_misses'][-1],
            summary['num_switches'][-1]))
        self.writer.add_scalar('MOTA', summary['mota'][-1], epoch)
        self.writer.add_scalar('mot_FP', summary['num_false_positives'][-1], epoch)
        self.writer.add_scalar('mot_FN', summary['num_misses'][-1], epoch)
        self.writer.add_scalar('mot_idsw', summary['num_switches'][-1], epoch)

    def lr_decay(self):
        for params in self.optimizer.param_groups:
            params['lr'] *= self.lr_decay_step
            lr = params['lr']
            Log.info("Learning rate adjusted to {}".format(lr))

    def start(self):
        self.best_r1, self.best_mAP = (0., -1), (0., -1)

        for epoch in range(1, self.max_epoch + 1):
            if epoch in self.lr_decay_checkpoints:
                self.lr_decay()
            self.train(epoch)
            if epoch % self.test_epoch_frequency == 0:
                mAP = self.eval(epoch)
                if len(self.metric_list) == 0:
                    self.logger.info(
                        '[{} on GPU {}] first evaluation completed... adding mAP...'.format(self.module_name,
                                                                                            str(self.gpu_id)))
                    self.metric_list.append(mAP)
                # elif self.not_changing_solver(mAP):
                #     self.not_changing_cnt += 1
                #     if self.not_changing_cnt >= self.not_changing_limit:
                #         Log.warn(
                #             '[{} on GPU {}] mAP not changing...Ending...'.format(self.module_name, str(self.gpu_id)))
                #         break
                #     else:
                #         Log.info('[{} on GPU {}] mAP not changing, recording it...'.format(self.module_name,
                #                                                                            str(self.gpu_id)))
                # else:
                #     self.not_changing_cnt = 0
            if epoch % self.mot_frequency == 0:
                self.eval_mot(epoch)

        return self.best_mAP


class RunModel(object):
    def unittest(self, gpu_id, file_path: str, max_epoch: int, batch_size: int, num_workers: int):
        import random
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # a = torch.randn(100, 100).cuda()  # intimidate training
        # time.sleep(10)
        individual_name = str(os.path.basename(__file__)).split('.')[0]
        best_metric = random.random()
        Log.info('Evaluation for %s finished with final mAP of %.4f' % (individual_name, best_metric))
        with open(file_path, 'a+') as f:
            f.write('%s=%.5f\n' % (individual_name, best_metric))
        # train = TrainModel(gpu_id='0')
        # input = torch.randn((10, 3, 128, 64)).to(torch.device('cuda:0'))
        # assert type(train.model.forward_once(input)) == torch.Tensor
        # best_metric = -1.0
        # try:
        #     best_metric = train.start()
        # finally:
        #     pass

    def do_work(self, gpu_id, file_path: str, max_epoch: int, batch_size: int, num_workers: int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        individual_name = str(os.path.basename(__file__)).split('.')[0]
        best_metric = -1.0
        retry_times = 0
        try:
            while retry_times < 3:
                try:
                    train = TrainModel(gpu_id=str(gpu_id), batch_size=batch_size, num_workers=num_workers,
                                       max_epoch=max_epoch)
                    best_metric, best_epoch = train.start()
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        retry_times += 1
                        batch_size = batch_size // 2
                        Log.warn("[%s on GPU %s]Out of memory detected! Trying to lower batch_size to %d and retrain..." % (individual_name, gpu_id, batch_size))
                    else:
                        raise e
                finally:
                    if 'train' in locals():
                        for p in train.model.parameters():
                            if p.grad is not None:
                                del p.grad
                        del train
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(5)
        except Exception as e:
            Log.warn('%s[on GPU %s], pid: %d with ' % (individual_name, gpu_id, os.getpid()) + str(e))
        finally:
            Log.info('Evaluation for %s finished with final mAP of %.4f' % (individual_name, best_metric))
            with open(file_path, 'a+') as f:
                f.write('%s=%.5f\n' % (individual_name, best_metric))


if __name__ == '__main__':  # unit_test for network_builder
    _obj = RunModel()
    _obj.do_work('0', './after_test.txt', max_epoch=30, batch_size=100, num_workers=4)
