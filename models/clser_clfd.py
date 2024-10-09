import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
import os
import torchvision.transforms as transforms



# =============================================================================
# Mean-ER
# =============================================================================
class CLSER_clfd(ContinualModel):
    NAME = 'clser_clfd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
        # add_management_args(parser)
        # add_experiment_args(parser)
        add_rehearsal_args(parser)

        # Consistency Regularization Weight
        parser.add_argument('--reg_weight', type=float, default=0.1)

        # Stable Model parameters
        parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
        parser.add_argument('--stable_model_alpha', type=float, default=0.999)

        # Plastic Model Parameters
        parser.add_argument('--plastic_model_update_freq', type=float, default=0.90)
        parser.add_argument('--plastic_model_alpha', type=float, default=0.999)

        return parser

    def __init__(self, backbone, loss, args, transform):
        super(CLSER_clfd, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size * 4)
        self.dropout_warmup = args.n_epochs * 0.4
        self.n_epochs = args.n_epochs
        self.fre_features = None
        self.cal_fre_features = False
        self.fre_sim = torch.zeros(self.N_CLASSES, self.N_CLASSES).to(self.device)
        self.dropout_factor = torch.zeros(self.N_CLASSES)
        for i in range(0, self._cpt):
            self.dropout_factor[i] = 2
        self.fre_transform = None
        self.first_task_num = 0
        # Initialize plastic and stable model
        self.plastic_model = deepcopy(self.net).to(self.device)
        self.stable_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = args.plastic_model_update_freq
        self.plastic_model_alpha = args.plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.global_step = 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.cal_fre_features:
            self.get_fre_feature(not_aug_inputs, labels)
        if self.current_task == 0:
            self.first_task_num += inputs.shape[0]

        self.opt.zero_grad()
        outputs = self.net(inputs, labels)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            if self.fre_transform == None:
                buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size,
                                                                 device=self.device)
            else:
                buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size,transform=self.fre_transform,
                                                              device=self.device)
                mask = (buf_inputs == 0)
                buf_inputs[mask] = self.fill_tensor[mask]
            buf_outputs = self.net.construct(buf_inputs)
            ce_loss = self.loss(buf_outputs, buf_labels)
            loss += ce_loss

            stable_model_logits = self.stable_model.construct(buf_inputs)
            plastic_model_logits = self.plastic_model.construct(buf_inputs)

            stable_model_prob = F.softmax(stable_model_logits, 1)
            plastic_model_prob = F.softmax(plastic_model_logits, 1)

            label_mask = F.one_hot(buf_labels, num_classes=stable_model_logits.shape[-1]) > 0
            sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
            sel_idx = sel_idx.unsqueeze(1)

            ema_logits = torch.where(
                sel_idx,
                stable_model_logits,
                plastic_model_logits,
            )

            l_cons = torch.mean(self.consistency_loss(self.net.construct(buf_inputs), ema_logits.detach()))
            l_reg = self.args.reg_weight * l_cons
            loss += l_reg

        loss.backward()
        self.opt.step()
        inputs = torch.stack([self.transform(ee) for ee in not_aug_inputs]).to(self.device)
        fre_feature, _ = self.net.feature_extractor(inputs)

        self.buffer.add_data(examples=fre_feature.detach(),
                             labels=labels)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.plastic_model_update_freq:
            self.update_plastic_model_variables()

        if torch.rand(1) < self.stable_model_update_freq:
            self.update_stable_model_variables()

        return loss.item()

    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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

