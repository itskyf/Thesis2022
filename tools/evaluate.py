import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Union

import torch
import torch.backends.cudnn
from eval_utils import eval_utils
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader

import pcdet.datasets
import pcdet.models
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file


def main():
    args, conf = parse_config()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")  # TODO

    batch_size = (
        conf.OPTIMIZATION.BATCH_SIZE_PER_GPU if args.batch_size is None else args.batch_size
    )
    class_names = conf.CLASS_NAMES
    val_set, val_loader = build_dataloader(
        conf.DATA_CONFIG, batch_size, args.num_workers, class_names
    )

    ckpt_path: Path = args.ckpt_path
    if ckpt_path.suffix == ".pkl" and ckpt_path.stem.startswith("result"):
        with ckpt_path.open("rb") as ret_file:
            eval_ret = pickle.load(ret_file)
        print_result(class_names, eval_ret)
    # ckpt in out_dir/ckpt/ckpt.pt -> eval_dir = out_dir/eval
    eval_dir = ckpt_path.parent.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_fn = getattr(pcdet.models, conf.MODEL.NAME)
    model: nn.Module = model_fn(conf.MODEL, len(class_names))
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path)["model_state"])

    print("Batch size:", batch_size)
    thresh_list = cfg.MODEL.post_process_cfg.thresh_list

    det_annos, recall_dict = eval_utils.eval_one_epoch(model, val_set, val_loader, thresh_list)
    rcnn_recall = [recall_dict[f"recall_rcnn_{thresh}"] for thresh in thresh_list]
    print(tabulate([rcnn_recall], headers=["IoU3D recall", *thresh_list]))

    det_annos_path = eval_dir / f"det_{ckpt_path.stem}.pkl"
    with det_annos_path.open("wb") as det_file:
        pickle.dump(det_annos, det_file)
    print("Prediction is saved to", det_annos_path)

    print("Evaluating...")
    eval_ret = val_set.evaluation(det_annos)
    eval_ret_path = eval_dir / f"result_{ckpt_path.stem}.pkl"
    with eval_ret_path.open("wb") as ret_file:
        pickle.dump(eval_ret, ret_file)
    print("Evaluation result is saved to", eval_ret_path)

    print_result(class_names, eval_ret)


def build_dataloader(data_cfg, batch_size: int, num_workers: int, class_names: List[str]):
    val_set_fn = getattr(pcdet.datasets, data_cfg.DATASET)
    val_set = val_set_fn(data_cfg, class_names, training=False)
    val_loader = DataLoader(
        val_set,
        batch_size,
        collate_fn=val_set.collate_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
    return val_set, val_loader


def print_result(class_names: list[str], eval_ret):
    ret_table: list[list[Union[float, str]]] = [["Easy"], ["Moderate"], ["Hard"]]
    for row in ret_table:
        for name in class_names:
            row.append(eval_ret[f"{name}_3d/{row[0]}_R40"])
    print(tabulate(ret_table, headers=["Kitti R40", *class_names]))


def create_logger(log_path: Path, rank: int):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.addHandler(console)
    if log_path is not None:
        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--set", dest="set_cfgs", nargs=argparse.REMAINDER, help="set extra config keys if needed"
    )
    parser.add_argument(
        "--eval_all", action="store_true", default=False, help="whether to evaluate all checkpoints"
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_path, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


if __name__ == "__main__":
    main()
