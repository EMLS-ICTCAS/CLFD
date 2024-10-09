# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import torchvision.transforms as transforms

class ErACE_clfd(ContinualModel):
    NAME = 'er_ace_clfd'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Experience Replay with asymmetric cross-entropy.')
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size * 4)
        self.dropout_warmup = args.n_epochs * 0.4
        self.n_epochs = args.n_epochs
        self.fre_features = None
        self.cal_fre_features = False
        self.fre_sim = torch.zeros(self.N_CLASSES, self.N_CLASSES).to(self.device)
        self.dropout_factor = torch.zeros(self.N_CLASSES)
        for i in range(0,self._cpt):
            self.dropout_factor[i] = 2
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.fre_transform = None
        self.first_task_num = 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.cal_fre_features:
            self.get_fre_feature(not_aug_inputs, labels)
        if self.current_task == 0:
            self.first_task_num += inputs.shape[0]

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs, labels)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.current_task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.current_task > 0:
            # sample from buffer
            if self.fre_transform == None:
                buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size,
                                                                 device=self.device)
            else:
                buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size,transform=self.fre_transform,
                                                              device=self.device)
                mask = (buf_inputs == 0)
                buf_inputs[mask] = self.fill_tensor[mask]
            loss_re = self.loss(self.net.construct(buf_inputs), buf_labels)

        loss += loss_re

        loss.backward()
        self.opt.step()
        inputs = torch.stack([self.transform(ee) for ee in not_aug_inputs]).to(self.device)
        fre_feature, _ = self.net.feature_extractor(inputs)

        self.buffer.add_data(examples=fre_feature.detach(),
                             labels=labels)

        return loss.item()

    def get_fre_feature(self, not_aug_inputs, labels):
        if self.fre_features is None:
            self.fre_features = torch.zeros(self.N_CLASSES,
                                            int(not_aug_inputs.shape[1] * not_aug_inputs.shape[2] * not_aug_inputs.shape[3] / 4)).to(
                self.device)
        inputs = torch.stack([self.transform(ee) for ee in not_aug_inputs]).to(self.device)
        _, fre_feature = self.net.feature_extractor(inputs)
        for i in range(self.current_task*self._cpt , self.current_task*self._cpt + self._cpt):
            mask = labels == i
            if mask.any():
                sum_value = fre_feature[mask].sum(dim = 0)
                self.fre_features[i] += sum_value.flatten()
        if self.fre_transform == None:
            pic = torch.zeros_like(not_aug_inputs)
            pic = torch.stack([self.transform(ee) for ee in pic]).to(self.device)
            self.fill_tensor, _ = self.net.feature_extractor(pic)
            self.fre_transform = transforms.Compose(
                 [transforms.RandomCrop(int(not_aug_inputs.shape[2]/2), padding=2),
                 transforms.RandomHorizontalFlip()])

    def get_fre_smi(self):
        for i in range(self.current_task * self._cpt, self.current_task * self._cpt + self._cpt):
            for j in range(0, self.current_task * self._cpt):
                self.fre_sim[i][j] = torch.cosine_similarity(self.fre_features[i], self.fre_features[j], dim=0)

    def end_task(self, dataset) -> None:
        print("mem:",torch.cuda.max_memory_allocated())
        if self.current_task == 0:
            self.net.freeze_layers()
            self.buffer.num_seen_examples = self.first_task_num


    def begin_epoch(self, epoch) -> None:
        if self.current_task == 0:
            if epoch == 1:
                for i in range(self.current_task * self._cpt, self.current_task * self._cpt + self._cpt):
                    self.net.classwise_select_probs[i] = self.net.select_probs
            elif epoch == self.n_epochs:
                self.cal_fre_features = True
                self.net.freeze_layers()
                self.buffer.empty()
        elif epoch == 1:
            self.net.select_probs[:] = self.net.dropout_st
            self.cal_fre_features = True
            for i in range(self.current_task * self._cpt, self.current_task * self._cpt + self._cpt):
                self.net.classwise_select_probs[i] = self.net.select_probs
        else:
            self.cal_fre_features = False

    def end_epoch(self, epoch) -> None:
        if self.current_task == 0:
            if epoch == self.n_epochs:
               self.get_fre_smi()
        elif epoch == 1:
            self.get_fre_smi()
            for i in range(self.current_task * self._cpt, self.current_task * self._cpt + self._cpt):
                min_index = self.fre_sim[i][self.fre_sim[i].nonzero()].argmin(dim = 0)
                max_index = self.fre_sim[i].argmax(dim=0)
                activation_min = self.net.classwise_select_counts[min_index]
                max_act = torch.max(activation_min)
                min_factor = ((self.fre_sim[i].sum(dim = 0)+torch.count_nonzero(self.fre_sim[i], dim=0))/\
                             ((self.fre_sim[i][min_index]+1)*torch.count_nonzero(self.fre_sim[i], dim=0))).cpu()
                min_pro = torch.exp(-activation_min * min_factor / (max_act + 1e-16))
                activation_max = self.net.classwise_select_counts[max_index]
                max_act = torch.max(activation_max)
                max_factor = (((self.fre_sim[i][max_index]+1) * torch.count_nonzero(self.fre_sim[i], dim=0))/\
                              (self.fre_sim[i].sum(dim=0)+torch.count_nonzero(self.fre_sim[i], dim=0))).cpu()
                max_pro = 1 - torch.exp(-activation_max * max_factor / (max_act + 1e-16))
                self.net.classwise_select_probs[i] =  min_pro/2 + max_pro/2
                self.dropout_factor[i] = 2


        if epoch > self.dropout_warmup:
            activation_counts = self.net.classwise_select_counts
            max_act = torch.max(activation_counts, dim=1)[0]
            self.net.classwise_select_probs = 1 - torch.exp(-activation_counts * self.dropout_factor.unsqueeze(1) / (max_act[:, None] + 1e-16))