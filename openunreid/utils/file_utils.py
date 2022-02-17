import errno
import json
import os
import os.path as osp
import sys
import time

import requests


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_json(fpath):
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def download_url(url, dst):
    """Downloads file from a url to a destination.
    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    mkdir_if_missing(osp.dirname(dst))
    from six.moves import urllib

    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
            % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dst, _reporthook)
        sys.stdout.write("\n")
    except Exception:
        raise RuntimeError(
            "Please download the dataset manually from {} " "to {}".format(url, dst)
        )

