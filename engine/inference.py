# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
import os
from ignite.engine import Engine, Events

from utils.reid_metric import R1_mAP, R1_mAP_reranking
from utils.visualize import visualize_ranked_results, vis_AP, vis_norm_uc


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)

            if isinstance(feat, torch.Tensor):
                feat = feat.cpu()
            elif isinstance(feat, (tuple, list)):
                feat = tuple(x.cpu() for x in feat)
                feat = torch.cat([feat[0], feat[1][:, None]], dim=-1)
            # feat = torch.rand(len(pids), 16)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, return_AP=True,
                                  norm_k=cfg.TEST.NORM_K)}
        evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}
        evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    # import pdb
    # pdb.set_trace()
    cmc, mAP, APs = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))

    vis_AP(APs, cfg.OUTPUT_DIR)
    vis_norm_uc(metrics['r1_mAP'].norms, metrics['r1_mAP'].uc, save_dir=cfg.OUTPUT_DIR)

    if cfg.TEST.VISRANK > 0:
        visualize_ranked_results(
            list(metrics.values())[0].dist_mat,
            val_loader.dataset.dataset,
            'image',
            width=cfg.INPUT.SIZE_TEST[1],
            height=cfg.INPUT.SIZE_TEST[0],
            save_dir=os.path.join(cfg.OUTPUT_DIR, 'visrank_'+cfg.DATASETS.NAMES[1]),
            topk=cfg.TEST.VISRANK,
            norms=(None if cfg.TEST.NORM_K is None else list(metrics.values())[0].norms),
            AP=APs
        )
