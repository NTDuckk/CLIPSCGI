from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

# bases.py
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, caption_by_img=None, caption_by_pid=None, caption_dict=None):
        self.dataset = dataset
        self.transform = transform
        self.caption_by_img = caption_by_img
        self.caption_by_pid = caption_by_pid
        self.caption_dict = caption_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        fname = osp.basename(img_path)

        caption = None
        caption_source = None
        # ưu tiên caption_dict nếu có (để tương thích code cũ)
        if self.caption_dict is not None:
            caption = self.caption_dict.get(fname) or self.caption_dict.get(img_path)
            if caption is not None:
                caption_source = "caption_dict"

        if caption is None and self.caption_by_img is not None:
            caption = self.caption_by_img.get(fname) or self.caption_by_img.get(img_path)
            if caption is not None:
                caption_source = "caption_by_img"

        if caption is None and self.caption_by_pid is not None:
            caption = self.caption_by_pid.get(pid) or self.caption_by_pid.get(str(pid))
            if caption is not None:
                caption_source = "caption_by_pid"

        has_caption_source = (self.caption_dict is not None) or (self.caption_by_img is not None) or (self.caption_by_pid is not None)

        if has_caption_source:
            if caption is None:
                caption = ""   # đảm bảo luôn là str
            return img, pid, camid, trackid, img_path, caption
        else:
            return img, pid, camid, trackid, img_path