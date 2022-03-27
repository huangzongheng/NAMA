# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import datetime
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP, R1_mAP_qg
from utils import cum_matmul
from utils.visualize import vis_AP, visualize_ranked_results, vis_norm_uc
from data.transforms.transforms import random_affine_img
from einops import reduce, rearrange
from .inference import create_supervised_evaluator

global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, opt_level='O1', affine='pad', cls_weight=None,
                              sampler=None, n_imgs=4):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    # if device:
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)
    # model.to(device)

    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if 'cpu' in device:
        model = model.cpu()
        # model = nn.DataParallel(model)
    elif 'cuda' in device:
        device = 'cuda'
        model = model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, enabled=True)
        if torch.cuda.device_count() > 1:
            # model = nn.DataParallel(model)
            print('use muti gpu')
    # model = nn.DataParallel(model)

    if hasattr(sampler, 'bank'):
        bank = sampler.bank
    else:
        bank = None

    if hasattr(model, 'module'):
        base = model.module.base
        # affine_predictor = nn.DataParallel(model.module.base.affine_predictor)
        print('use data parallel')
    else:
        base = model.base

    def _update(engine, batch):
        # if not hasattr(_update, 'loss_fn'):
        #     _update.loss_fn = loss_fn
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target

        out = model(img)
        score = out[0]
        # score, feat = model(img)
        # if bank is not None:
        #     cur_center = reduce(feat, '(n i) c -> n c', 'max', i=n_imgs).detach().cpu()
        #     bank[target[::n_imgs]] += 0.2 * (cur_center - bank[target[::n_imgs]])     # ema
            # bank[target[::n_imgs]] = feat[::n_imgs].detach().cpu()

        # score, feat = model(img)        # forward

        loss = _update.loss_fn(*out, target=target)
        # loss = _update.loss_fn(score, feat, target, cls_weight)
        if isinstance(loss, tuple):
            loss, loss_t = loss
            loss_t = loss_t.item()
        else:
            loss_t = 0

        if 'cpu' in device:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item(), loss_t

    _update.loss_fn = loss_fn
    return Engine(_update)

# def create_supervised_evaluator(model, metrics,
#                                 device=None):
#     """
#     Factory function for creating an evaluator for supervised models
#
#     Args:
#         model (`torch.nn.Module`): the model to train
#         metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
#         device (str, optional): device type specification (default: None).
#             Applies to both model and batches.
#     Returns:
#         Engine: an evaluator engine with supervised inference function
#     """
#     if device:
#         # if torch.cuda.device_count() > 1:
#         #     model = nn.DataParallel(model)
#         model.to(device)
#
#     def _inference(engine, batch):
#         model.eval()
#         with torch.no_grad():
#             data, pids, camids = batch
#             data = data.to(device) if torch.cuda.device_count() >= 1 else data
#             feat = model(data)
#             if isinstance(feat, torch.Tensor):
#                 feat = feat.cpu()
#             elif isinstance(feat, (tuple, list)):
#                 feat = tuple(x.cpu() for x in feat)
#             return feat, pids, camids
#
#     engine = Engine(_inference)
#
#     for name, metric in metrics.items():
#         metric.attach(engine, name)
#
#     return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    cls_weight = [1/(len(v)**0.25) for v in train_loader.sampler.index_dic.values()] # (k, len(v)**0.25)
    cls_weight = torch.tensor(cls_weight) / min(cls_weight)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device, opt_level=cfg.SOLVER.OPT_LEVEL,
                                        affine=cfg.INPUT.VE_MODE, cls_weight=cls_weight,
                                        sampler=train_loader.sampler, n_imgs=cfg.DATALOADER.NUM_INSTANCE)
    def gst(engine, last_event_name):
        return engine.state.epoch
    metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, return_AP=True)}
    evaluator = create_supervised_evaluator(model, metrics=metrics,
                                            device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, None, n_saved=1, require_empty=False, global_step_transform=gst)
    # checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
    # checkpointer.add

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'avg_loss_t')

    # trainer.state.epoch = start_epoch
    # import pdb
    # pdb.set_trace()

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    # @trainer.on(Events.EPOCH_STARTED)
    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    # 动态调整triplet loss权重
    # @trainer.on(Events.EPOCH_STARTED)
    # def adjust_tri_weight(engine):
    #     engine._process_function.loss_fn.weight_t \
    #         = cfg.SOLVER.TRI_LOSS_WEIGHT * min(1, cfg.SOLVER.TW_MIN + (engine.state.epoch / cfg.SOLVER.STEPS[0]) ** 2)
    #     print('Tri Weight: {:.3f}'.format(engine._process_function.loss_fn.weight_t))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            eta_seconds = timer.value() * (len(train_loader) - ITER + 1 +
                                           (engine.state.max_epochs - (engine.state.epoch)) * len(train_loader))

            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_t: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e} eta: {}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss_t'],
                                engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0], str(datetime.timedelta(seconds=int(eta_seconds)))))

    # 计算一个epoch的平均triplet loss
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def compute_epoch_loss(engine):
    #
    #     if engine.state.epoch < 10:
    #         engine._process_function.loss_fn.weight_t = 0.01
    #     elif engine.state.epoch <= 20:
    #         engine._process_function.loss_fn.weight_t = 0.01 + (cfg.SOLVER.TRI_LOSS_WEIGHT - 0.01) * (engine.state.epoch - 10) / 10

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        global ITER
        ITER = 0
        logger.info('Epoch {}/{} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, engine.state.max_epochs, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        if hasattr(loss_fn.triplet, 'avg_m') and hasattr(loss_fn.triplet, 'avg_l'):
            info = "Avg margin: {:.3f}/({:.3f} {:.3f}), Avg norm: {:.3f}".format(
                loss_fn.triplet.avg_m, loss_fn.triplet.min_m, loss_fn.triplet.max_m, loss_fn.triplet.avg_l
            )
            if hasattr(loss_fn.triplet, 'avg_lw'):
                info += "/{:.3f}".format(loss_fn.triplet.avg_lw)
            logger.info(info)
        if hasattr(model, 'uc_k'):
            logger.info(" uc_k: {}".format(model.uc_k))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            logger.info('Eval complete. Time: {}'
                        .format(str(datetime.timedelta(seconds=int(timer.value())))))
            cmc, mAP, APs = evaluator.state.metrics['r1_mAP']
            logger.info('Validation Results')
            logger.info("mAP: {:.2%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
            vis_AP(APs, cfg.OUTPUT_DIR)
            vis_norm_uc(metrics['r1_mAP'].norms, metrics['r1_mAP'].uc, save_dir=cfg.OUTPUT_DIR)

            torch.save({'norm': metrics['r1_mAP'].norms, 'uc':metrics['r1_mAP'].uc, 'data':val_loader.dataset.dataset,
                        'AP': metrics['r1_mAP'].AP},
                       os.path.join(cfg.OUTPUT_DIR, 'norm.pt'))

            # if cfg.TEST.VISRANK > 0:
            #     visualize_ranked_results(metrics['r1_mAP'].dist_mat, val_loader)
        if engine.state.epoch >= epochs and cfg.TEST.VISRANK > 0:
            visualize_ranked_results(metrics['r1_mAP'].dist_mat, val_loader.dataset.dataset, save_dir=os.path.join(cfg.OUTPUT_DIR, 'visrank'))
            # visualize_ranked_results(metrics['r1_mAP'].dist_mat, val_loader, norms=metrics['r1_mAP'].norms)
        timer.reset()

    torch.cuda.empty_cache()
    trainer.run(train_loader, max_epochs=epochs)
    # while trainer.state.epoch < epochs:
    #     try:
    #         trainer.run(train_loader, max_epochs=epochs)
    #     except RuntimeError as e:
    #         trainer.state.epoch = max(0, trainer.state.epoch - 1)
    #         logger.error(e.__repr__())
    if epochs == 0:
        log_validation_results(trainer)

