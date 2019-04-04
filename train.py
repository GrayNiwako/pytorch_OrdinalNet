# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import torch as t
from torch.nn import Module
from torchnet import meter
from core import *
from validate import validate as val
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


def train(model, train_data, val_data, config, vis):
    # type: (Module,CloudDataLoader,CloudDataLoader,Config,Visualizer)->None

    # init loss and optim
    if config.loss_type == 'cross_entropy':
        criterion1, criterion2 = t.nn.CrossEntropyLoss(), t.nn.CrossEntropyLoss()
    elif config.loss_type == 'mseloss':
        criterion1, criterion2 = t.nn.MSELoss(), t.nn.MSELoss()
    else:
        print('Invalid value: config.loss_type')

    if config.optimizer == 'sgd':
        optimizer = t.optim.SGD(model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = t.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        print('Invalid value: config.optimizer')

    scheduler = t.optim.lr_scheduler.StepLR(optimizer, 1, config.lr_decay)
    # try to resume
    last_epoch = resume_checkpoint(config, model, optimizer)
    assert last_epoch + 1 < config.max_epoch, \
        "previous training has reached epoch {}, please increase the max_epoch in {}". \
            format(last_epoch + 1, type(config))
    if last_epoch == -1:  # start a new train proc
        vis.save(config.vis_env_path + 'last')
        vis.clear()
    # init meter statistics
    loss_meter = meter.AverageValueMeter()
    loss1_meter = meter.AverageValueMeter()
    loss2_meter = meter.AverageValueMeter()
    confusion_matrix1 = meter.ConfusionMeter(config.num_classes)
    confusion_matrix2 = meter.ConfusionMeter(config.num_classes)
    last_accuracy = 0
    for epoch in range(last_epoch + 1, config.max_epoch):
        epoch_start = time.time()
        loss_mean = None
        train_acc1 = 0
        train_acc2 = 0
        scheduler.step(epoch)
        loss_meter.reset()
        loss1_meter.reset()
        loss2_meter.reset()
        confusion_matrix1.reset()
        confusion_matrix2.reset()
        model.train()
        for i, input in enumerate(train_data):
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
                    criterion1 = criterion1.cuda()
                    criterion2 = criterion2.cuda()
            batch_sub_img.requires_grad_(True)
            batch_parent_img.requires_grad_(True)
            batch_sub_label.requires_grad_(False)
            batch_pare_label.requires_grad_(False)
            batch_sub_label = batch_sub_label.view(-1)
            batch_pare_label = batch_pare_label.view(-1)
            # forward
            batch_sub_prob, batch_pare_prob = model(batch_sub_img, batch_parent_img)
            c1 = criterion1(batch_sub_prob, batch_sub_label)
            c2 = criterion2(batch_pare_prob, batch_pare_label)
            loss, loss1, loss2 = 2*c1*c2/(c1+c2), c1, c2
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistic
            loss_meter.add(loss.data.cpu())
            loss1_meter.add(loss1.data.cpu())
            loss2_meter.add(loss2.data.cpu())

            if config.loss_type == 'cross_entropy':
                batch_sub_prob = t.argmax(batch_sub_prob, dim=-1)
                batch_pare_prob = t.argmax(batch_pare_prob, dim=-1)
            elif config.loss_type == 'mseloss':
                batch_sub_prob = conv_to_label(batch_sub_prob)
                batch_pare_prob = conv_to_label(batch_pare_prob)
            confusion_matrix1.add(batch_sub_prob, batch_sub_label)
            confusion_matrix2.add(batch_pare_prob, batch_pare_label)

            # print process
            if i % config.ckpt_freq == 0 or i >= len(train_data) - 1:
                step = epoch * len(train_data) + i
                loss_mean = loss_meter.value()[0]
                loss1_mean = loss1_meter.value()[0]
                loss2_mean = loss2_meter.value()[0]
                cm1_value = confusion_matrix1.value()
                num_correct1 = cm1_value.trace().astype(np.float)
                train_acc1 = num_correct1 / cm1_value.sum()
                cm2_value = confusion_matrix2.value()
                num_correct2 = cm2_value.trace().astype(np.float)
                train_acc2 = num_correct2 / cm2_value.sum()
                vis.plot(loss1_mean, step, 'sub_loss', 'Loss Curve', ['sub_loss', 'pare_loss', 'loss_sum'])
                vis.plot(loss2_mean, step, 'pare_loss', 'Loss Curve')
                vis.plot(loss_mean, step, 'loss_sum', 'Loss Curve')
                vis.plot(train_acc1, step, 'sub_acc', 'Training Accuracy', ['sub_acc', 'pare_acc'])
                vis.plot(train_acc2, step, 'pare_acc', 'Training Accuracy')
                lr = optimizer.param_groups[0]['lr']
                msg = "epoch:{},iteration:{}/{},loss:{},sub_loss:{},pare_loss:{},sub_train_acc:{},pare_train_acc:{},lr:{}\nsub_confusion_matrix:\n{}\npare_confusion_matrix:\n{}".format(
                    epoch, i, len(train_data) - 1, loss_mean, loss1_mean, loss2_mean,
                    train_acc1, train_acc2, lr, confusion_matrix1.value(), confusion_matrix2.value())
                vis.log_process(i, len(train_data) - 1, msg, 'train_log')

        # validate after each epoch
        sub_acc, sub_f1_score, sub_cm, pare_acc, pare_f1_score, pare_cm, pare_bo_sub_acc, pare_bo_sub_f1_score, pare_bo_sub_cm = val(model, val_data, config, vis)
        vis.plot(sub_acc, epoch, 'sub_acc', 'Validation Accuracy', ['sub_acc', 'pare_acc', 'p_b_s_acc'])
        vis.plot(pare_acc, epoch, 'pare_acc', 'Validation Accuracy')
        vis.plot(pare_bo_sub_acc, epoch, 'p_b_s_acc', 'Validation Accuracy')
        vis.plot(sub_f1_score, epoch, 'sub_f1score', 'Validation Macro F1-Score', ['sub_f1score', 'pare_f1score', 'p_b_s_f1score'])
        vis.plot(pare_f1_score, epoch, 'pare_f1score', 'Validation Macro F1-Score')
        vis.plot(pare_bo_sub_f1_score, epoch, 'p_b_s_f1score', 'Validation Macro F1-Score')
        # save checkpoint
        if pare_acc > last_accuracy:
            msg += '\nbest validation result after epoch {}, loss:{}, sub_train_acc: {}, pare_train_acc: {}\n'.format(epoch, loss_mean, train_acc1, train_acc2)
            msg += 'sub-image validation accuracy:{}\n'.format(sub_acc)
            msg += 'parent-image validation accuracy:{}\n'.format(pare_acc)
            msg += 'parent-image base on sub-image validation accuracy:{}\n'.format(pare_bo_sub_acc)
            msg += 'sub-image validation macro f1-score:{}\n'.format(sub_f1_score)
            msg += 'parent-image validation macro f1-score:{}\n'.format(pare_f1_score)
            msg += 'parent-image base on sub-image validation macro f1-score:{}\n'.format(pare_bo_sub_f1_score)
            msg += 'validation sub confusion matrix:\n{}\n'.format(sub_cm)
            msg += 'validation pare confusion matrix:\n{}\n'.format(pare_cm)
            msg += 'validation pare-base-on-sub confusion matrix:\n{}\n'.format(pare_bo_sub_cm)
            vis.log(msg, 'best_val_result', log_file=config.val_result, append=False)
            print("save best validation result into " + config.val_result)
        last_accuracy = pare_acc
        make_checkpoint(config, epoch, epoch_start, loss_mean, train_acc1, sub_acc, model, optimizer)


def main(*args, **kwargs):
    config = Config('train', **kwargs)
    print(config)
    train_data = CloudDataLoader('train', config)
    val_data = CloudDataLoader('validation', config)
    model = get_model(config)
    vis = Visualizer(config)
    print("Prepare to train model...")
    train(model, train_data, val_data, config, vis)
    # save core
    print("Training Finish! Saving model...")
    try:
        t.save(model.state_dict(), config.weight_save_path)
        os.remove(config.temp_optim_path)
        os.remove(config.temp_weight_path)
        print("Model saved into " + config.weight_save_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to save model because {}, check temp weight file in {}".format(e, config.temp_weight_path))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    args = parse.parse_args()
    main(args)
