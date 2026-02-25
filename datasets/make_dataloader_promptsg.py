import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

# ✅ đổi import: load ra 2 map (by_img, by_pid)
from utils.caption import load_caption_maps

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occ_duke import OCC_DukeMTMCreID


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
}

def train_collate_fn(batch):
    if len(batch[0]) == 6:
        imgs, pids, camids, viewids, img_paths, caps = zip(*batch)
        caps = [c if c is not None else "" for c in caps]  # <= thêm dòng này
        return (
            torch.stack(imgs, dim=0),
            torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
            list(img_paths),
            list(caps),
        )
    else:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        return (
            torch.stack(imgs, dim=0),
            torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
            list(img_paths),
        )

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    method = getattr(cfg.MODEL, "METHOD", "promptsg")

    caption_by_img, caption_by_pid = None, None
    if method == "clip_scgi":
        cap_file = getattr(cfg.DATASETS, "CAPTION_FILE", "")
        if cap_file:
            cap_file = os.path.join(cfg.DATASETS.ROOT_DIR, cap_file)
        caption_by_img, caption_by_pid = load_caption_maps(cap_file)  # (by_img, by_pid)

    train_set = ImageDataset(
        dataset.train,
        train_transforms,
        caption_by_img=caption_by_img,
        caption_by_pid=caption_by_pid,
    )
    train_set_normal = ImageDataset(dataset.train, val_transforms)  # không cần caption


    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    ims_per_batch = cfg.SOLVER.PROMPTSG.IMS_PER_BATCH

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            mini_batch_size = ims_per_batch // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, ims_per_batch, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=ims_per_batch,
                sampler=RandomIdentitySampler(dataset.train, ims_per_batch, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.PROMPTSG.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        raise ValueError(f"unsupported sampler! expected softmax or triplet but got {cfg.DATALOADER.SAMPLER}")

    # Val/query/gallery: không cần caption
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )

    query_set = ImageDataset(dataset.query, val_transforms)
    query_loader = DataLoader(
        query_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )

    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )

    return train_loader, val_loader, query_loader, gallery_loader, len(dataset.query), num_classes, cam_num, view_num, len(caption_by_img) if caption_by_img else 0, len(caption_by_pid) if caption_by_pid else 0