# -*- coding: utf-8 -*-
import numpy as np
import torch as t
from torch.nn import Module
from torchnet import meter
from core import Config, Visualizer
from datasets import CloudDataLoader

def conv_to_label(batch_prob):
    length = batch_prob.numel()
    new_prob = t.FloatTensor(length).zero_()
    for i in range(length):
        if batch_prob[i] <= 0.0:
            new_prob[i] = 0.0
        elif batch_prob[i] <= 0.1:
            new_prob[i] = 1.0
        elif batch_prob[i] <= 0.25:
            new_prob[i] = 2.0
        elif batch_prob[i] <= 0.75:
            new_prob[i] = 3.0
        else:
            new_prob[i] = 4.0
    return new_prob


def sub_conv_pare(batch_sub_prob):
    label2score = [0.0, 0.1, 0.25, 0.75, 1.0, 0.0]
    sub_prob = batch_sub_prob.long()
    sub_len = sub_prob.numel()
    pare_prob = t.LongTensor(int(sub_len/8)).zero_()
    label_sum = 0
    count_F = 0
    j = 0
    for i in range(sub_len):
        if (i+1) % 8 != 0:
            if sub_prob[i] == 5:
                count_F += 1
            label_sum += label2score[sub_prob[i]] / 8
        else:
            pare_prob[j] = count_label(label_sum, count_F)
            j += 1
            label_sum = 0
            count_F = 0
    new_pare_prob = pare_prob.type_as(batch_sub_prob)
    return new_pare_prob


def count_label(number, count_F):
    if count_F == 8:
        return 5
    elif number == 0.0:
        return 0
    elif number <= 0.1:
        return 1
    elif number <= 0.25:
        return 2
    elif number <= 0.75:
        return 3
    else:
        return 4


def evaluate(confusion_matrix):
    cm = confusion_matrix.value()
    accuracy = cm.trace().astype(np.float) / cm.sum()
    macro_precision = 0
    macro_recall = 0
    for label in range(6):
        if int(cm[:, label].sum()) != 0:
            precision = cm[label, label].astype(np.float) / cm[:, label].sum()
            macro_precision += precision
        if int(cm[label].sum()) != 0:
            recall = cm[label, label].astype(np.float) / cm[label].sum()
            macro_recall += recall
    macro_precision /= 6
    macro_recall /= 6
    if macro_precision + macro_recall != 0:
        f1_score = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    else:
        f1_score = 0.0
    return accuracy, f1_score, cm


def validate(model, val_data, config, vis):
    # type: (Module,CloudDataLoader,Config,Visualizer)->(any,any,any,any,dict,dict)
    with t.no_grad():
        sub_confusion_matrix = meter.ConfusionMeter(config.num_classes)
        pare_confusion_matrix = meter.ConfusionMeter(config.num_classes)
        pare_bo_sub_confusion_matrix = meter.ConfusionMeter(config.num_classes)
        # validate
        for i, input in enumerate(val_data):
            model.eval()
            # input data
            batch_sub_img, batch_sub_label, batch_parent_img, batch_pare_label = input
            if config.loss_type == 'cross_entropy':
                batch_sub_label = batch_sub_label.long()
                batch_pare_label = batch_pare_label.long()
            elif config.loss_type == 'mseloss':
                batch_sub_label = batch_sub_label.float()
                batch_pare_label = batch_pare_label.float()

            if config.use_gpu:
                with t.cuda.device(0):
                    batch_sub_img = batch_sub_img.cuda()
                    batch_sub_label = batch_sub_label.cuda()
                    batch_parent_img = batch_parent_img.cuda()
                    batch_pare_label = batch_pare_label.cuda()
            batch_sub_label = batch_sub_label.view(-1)
            batch_pare_label = batch_pare_label.view(-1)
            # forward
            batch_sub_prob, batch_pare_prob = model(batch_sub_img, batch_parent_img)

            # confusion matrix statistic
            if config.loss_type == 'cross_entropy':
                batch_sub_prob = t.argmax(batch_sub_prob, dim=-1)
                batch_pare_prob = t.argmax(batch_pare_prob, dim=-1)
            elif config.loss_type == 'mseloss':
                batch_sub_prob = conv_to_label(batch_sub_prob)
                batch_pare_prob = conv_to_label(batch_pare_prob)
            sub_confusion_matrix.add(batch_sub_prob, batch_sub_label)
            pare_confusion_matrix.add(batch_pare_prob, batch_pare_label)

            batch_pare_bo_sub_prob = sub_conv_pare(batch_sub_prob)
            pare_bo_sub_confusion_matrix.add(batch_pare_bo_sub_prob, batch_pare_label)
            # print process
            if i % config.ckpt_freq == 0 or i >= len(val_data) - 1:
                cm_value = pare_confusion_matrix.value()
                msg = "[Validation]process:{}/{},scene confusion matrix:\n{}\n".format(i, len(val_data) - 1, cm_value)
                vis.log_process(i, len(val_data) - 1, msg, 'val_log', append=True)

        sub_acc, sub_f1_score, sub_cm = evaluate(sub_confusion_matrix)
        pare_acc, pare_f1_score, pare_cm = evaluate(pare_confusion_matrix)
        pare_bo_sub_acc, pare_bo_sub_f1_score, pare_bo_sub_cm = evaluate(pare_bo_sub_confusion_matrix)

    return sub_acc, sub_f1_score, sub_cm, pare_acc, pare_f1_score, pare_cm, pare_bo_sub_acc, pare_bo_sub_f1_score, pare_bo_sub_cm
