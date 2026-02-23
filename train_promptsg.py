import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from datasets.make_dataloader_promptsg import make_dataloader
from loss.make_loss_promptsg import make_loss
from model.make_model_promptsg import make_model
from solver.lr_scheduler import WarmupMultiStepLR
from solver.make_optimizer_promptsg import make_optimizer
from utils.logger import setup_logger
from processor.processor_promptsg import do_train


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PromptSG Training')
    parser.add_argument('--config_file', default='configs/person/vit_promptsg.yml', type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()

    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger('promptsg', output_dir, if_train=True)
    logger.info('Saving model in the path :{}'.format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != '':
        logger.info('Loaded configuration file {}'.format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info('\n' + cf.read())
    logger.info('Running with config:\n{}'.format(cfg))


    train_loader, val_loader, query_loader, gallery_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
    
    # Create directory for dataloader logging
    dataloader_log_dir = os.path.join(output_dir, 'make_dataloader_train_logging')
    if not os.path.exists(dataloader_log_dir):
        os.makedirs(dataloader_log_dir)
    
    # Setup logger for dataloader info
    dataloader_logger = setup_logger('dataloader_info', dataloader_log_dir, if_train=True)
    dataloader_logger.info('=== DataLoader Info ===')
    dataloader_logger.info(f'Train loader length: {len(train_loader)} batches')
    dataloader_logger.info(f'Val loader length: {len(val_loader)} batches')
    dataloader_logger.info(f'Num query: {num_query}')
    dataloader_logger.info(f'Num classes: {num_classes}')
    dataloader_logger.info(f'Camera num: {cam_num}')
    dataloader_logger.info(f'View num: {view_num}')
    dataloader_logger.info('======================')
    
    model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)

    loss_fn = make_loss(cfg, num_classes=num_classes)

    optimizer = make_optimizer(cfg, model)

    # Get warmup settings from config (with defaults)
    warmup_epochs = getattr(cfg.SOLVER.PROMPTSG, 'WARMUP_EPOCHS', 0)
    warmup_lr_init = getattr(cfg.SOLVER.PROMPTSG, 'WARMUP_LR_INIT', 0.0)

    if warmup_epochs > 0 and warmup_lr_init > 0:
        warmup_iters = warmup_epochs
        warmup_factor = warmup_lr_init / cfg.SOLVER.PROMPTSG.BASE_LR_NEW
        warmup_factor = min(1.0, max(0.0, warmup_factor))
    else:
        warmup_iters = 0
        warmup_factor = 1.0

    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.PROMPTSG.STEPS,
        cfg.SOLVER.PROMPTSG.GAMMA,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method='linear',
    )

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        query_loader,
        gallery_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        args.local_rank,
    )

    # After training completes, run gradient check and plotting
    import subprocess
    import glob

    print("\n" + "="*50)
    print("TRAINING COMPLETED - RUNNING POST-TRAINING TASKS")
    print("="*50)

    # 1. Check gradients
    print("Running gradient check...")
    try:
        result = subprocess.run(['python', 'check_grad.py'], capture_output=True, text=True)
        print("Gradient check output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Failed to run gradient check: {e}")

    # 2. Generate plots
    print("\nRunning learning curve plotting...")
    try:
        # Find the latest log file in output_dir
        log_files = glob.glob(os.path.join(output_dir, '**', '*.log'), recursive=True) + \
                   glob.glob(os.path.join(output_dir, '**', '*metrics*.txt'), recursive=True)
        
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            print(f"Found log file: {latest_log}")
            
            result = subprocess.run([
                'python', 'plot_learning_curves.py', 
                '--log_file', latest_log, 
                '--output_dir', os.path.join(output_dir, 'plots')
            ], capture_output=True, text=True)
            
            print("Plotting output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        else:
            print("No log files found for plotting")
    except Exception as e:
        print(f"Failed to run plotting: {e}")

    print("\n" + "="*50)
    print("ALL TASKS COMPLETED")
    print("="*50)
