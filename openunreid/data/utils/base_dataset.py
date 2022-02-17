# Modified from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/dataset.py  # noqa
# to support unsupervised features

import copy
import os.path as osp
import tarfile
import zipfile
from torch.utils.data import Dataset
from mmcv.runner import master_only
from ...utils.file_utils import download_url, mkdir_if_missing
from ..utils.data_utils import read_image

class CustomDataset(Dataset):
    """An abstract class representing a Dataset. All other image datasets should subclass it.
    Args:
        data (list): contains tuples of (img_path(s), pid, camid).
        mode (str): 'train', 'val', 'trainval', 'query' or 'gallery'.
        transform: transform function.
        verbose (bool): show information.
    """

    def __init__(self, data, mode, transform=None, verbose=True, sort=True, pseudo_labels=None, **kwargs):
        self.data = data
        self.transform = transform
        self.mode = mode
        self.verbose = verbose
        self.num_pids, self.num_cams = self.parse_data(self.data)

        if sort:
            self.data = sorted(self.data)
        
        # "all_data" stores the original data list
        # "data" stores the pseudo-labeled data list
        self.all_data = copy.deepcopy(self.data)

        if pseudo_labels is not None:
            self.renew_labels(pseudo_labels)

        if self.verbose:
            print(self.__repr__()) 

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        msg = (
            f"-----------------------------------------------------\n"
            f"         dataset        | # ids | # items | # cameras\n"
            f"-----------------------------------------------------\n"
            f"{self.__class__.__name__ } - {self.mode if self.mode != None else 'Nomode'}"
            f"    | {self.num_pids} | {self.__len__()} | {self.num_cams}\n"
        )
        return msg

    def __add__(self, other):
        """
        work for combining query and gallery into the test data loader
        """
        return CustomDataset(
            self.data + other.data,
            self.mode + "+" + other.mode,
            pseudo_labels=None,
            transform=self.transform,
            verbose=False,
            sort=False,
        )

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def _get_single_item(self, index):
        r"""``_get_single_item`` returns an image given index.
        It will return (``img``, ``img_path``, ``pid``, ``camid``, ``index``)
        where ``img`` has shape (channel, height, width). As a result,
        data in each batch has shape (batch_size, channel, height, width).
        """
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return {
            "img": img,
            "path": img_path,
            "id": pid,
            "cid": camid,
            "ind": index,
        }

    def renew_labels(self, pseudo_labels):
        assert isinstance(pseudo_labels, list)
        assert len(pseudo_labels) == len(
            self.all_data
        ), "the number of pseudo labels should be the same as that of data"

        data = []
        for label, (img_path, _, camid) in zip(pseudo_labels, self.all_data):
            if label != -1:
                data.append((img_path, label, camid))
        self.data = data
        self.num_pids, self.num_cams = self.parse_data(self.data)

    @master_only
    def download(self, dataset_dir, dataset_url) -> None:
        """Downloads and extracts dataset.
        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                f"{self.__class__.__name__} dataset needs to be manually "
                f"prepared, please download this dataset "
                f"under the folder of {dataset_dir}"
            )
        
        print(f"Creating directory {dataset_dir}")
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))
        
        print(f"Downloading {self.__class__.__name__} dataset to {dataset_dir}")
        download_url(dataset_url, fpath)
        
        print(f"Extracting {fpath}")
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except Exception:
            zip_ref = zipfile.ZipFile(fpath, "r")
            zip_ref.extractall(dataset_dir)
            zip_ref.close()
        print(f"{self.__class__.__name__} dataset is ready")
