import argparse
import os
import shutil
from glob import glob

import torch
import torch.utils.tensorboard
import yaml
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm

from agdiff import __file__ as agdiff_root
from agdiff.models.epsnet import get_model
from agdiff.utils.common import get_optimizer, get_scheduler
from agdiff.utils.datasets import ConformationDataset
from agdiff.utils.misc import *
from agdiff.utils.transforms import *

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    agdiff_root_dir = os.path.dirname(agdiff_root)
    models_dir = os.path.join(agdiff_root_dir, "models")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_iter", type=int, default=None)
    parser.add_argument(
        "--logdir", type=str, default=os.path.join(project_root_dir, "logs")
    )
    args = parser.parse_args()

    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, "*.yml"))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[
        : os.path.basename(config_path).rfind(".")
    ]
    seed_all(config.train.seed)

    # Logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag="resume")
        os.symlink(
            os.path.realpath(resume_from),
            os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))),
        )
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        # you may want to uncomment this line if you'd like to have a copy of the models folder in your output dir
        # shutil.copytree( models_dir, os.path.join(log_dir, 'models'))

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger("train", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    logger.info("Loading datasets...")
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms)
    val_set = ConformationDataset(config.dataset.val, transform=transforms)
    train_iterator = inf_iterator(
        DataLoader(
            train_set,
            config.train.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
    )
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)

    # Model
    logger.info("Building model...")
    model = get_model(config.model).to(args.device)

    logger.info("Building model OK...")

    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    # optimizer_global = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=5e-3)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)
    start_iter = 1

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(
            os.path.join(resume_from, "checkpoints"), it=args.resume_iter
        )
        logger.info("Resuming from: %s" % ckpt_path)
        logger.info("Iteration: %d" % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer_global.load_state_dict(ckpt["optimizer_global"])
        optimizer_local.load_state_dict(ckpt["optimizer_local"])
        scheduler_global.load_state_dict(ckpt["scheduler_global"])
        scheduler_local.load_state_dict(ckpt["scheduler_local"])

    def train(it):
        model.train()
        optimizer_global.zero_grad()
        optimizer_local.zero_grad()
        batch = next(train_iterator).to(args.device)
        loss, loss_global, loss_local = model.get_loss(
            atom_type=batch.atom_type,
            pos=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph,
            num_graphs=batch.num_graphs,
            anneal_power=config.train.anneal_power,
            return_unreduced_loss=True,
        )

        loss = loss.mean()
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer_global.step()
        optimizer_local.step()

        logger.info(
            "[Train] Iter %05d | Loss %.2f | Loss(Global) %.2f | Loss(Local) %.2f | Grad %.2f | LR(Global) %.6f | LR(Local) %.6f"
            % (
                it,
                loss.item(),
                loss_global.mean().item(),
                loss_local.mean().item(),
                orig_grad_norm,
                optimizer_global.param_groups[0]["lr"],
                optimizer_local.param_groups[0]["lr"],
            )
        )
        writer.add_scalar("train/loss", loss, it)
        writer.add_scalar("train/loss_global", loss_global.mean(), it)
        writer.add_scalar("train/loss_local", loss_local.mean(), it)
        writer.add_scalar("train/lr_global", optimizer_global.param_groups[0]["lr"], it)
        writer.add_scalar("train/lr_local", optimizer_local.param_groups[0]["lr"], it)
        writer.add_scalar("train/grad_norm", orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                batch = batch.to(args.device)
                loss, loss_global, loss_local = model.get_loss(
                    atom_type=batch.atom_type,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_nodes_per_graph=batch.num_nodes_per_graph,
                    num_graphs=batch.num_graphs,
                    anneal_power=config.train.anneal_power,
                    return_unreduced_loss=True,
                )
                sum_loss += loss.sum().item()
                sum_n += loss.size(0)
                sum_loss_global += loss_global.sum().item()
                sum_n_global += loss_global.size(0)
                sum_loss_local += loss_local.sum().item()
                sum_n_local += loss_local.size(0)
        avg_loss = sum_loss / sum_n
        avg_loss_global = sum_loss_global / sum_n_global
        avg_loss_local = sum_loss_local / sum_n_local

        if config.train.scheduler.type == "plateau":
            scheduler_global.step(avg_loss_global)
            scheduler_local.step(avg_loss_local)
        else:
            scheduler_global.step()
            scheduler_local.step()

        logger.info(
            "[Validate] Iter %05d | Loss %.6f | Loss(Global) %.6f | Loss(Local) %.6f"
            % (
                it,
                avg_loss,
                avg_loss_global,
                avg_loss_local,
            )
        )
        writer.add_scalar("val/loss", avg_loss, it)
        writer.add_scalar("val/loss_global", avg_loss_global, it)
        writer.add_scalar("val/loss_local", avg_loss_local, it)
        writer.flush()
        return avg_loss

    best_val_loss = float("inf")  # Initialize the best validation loss

    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, "%d.pt" % it)
                torch.save(
                    {
                        "config": config,
                        "model": model.state_dict(),
                        "optimizer_global": optimizer_global.state_dict(),
                        "scheduler_global": scheduler_global.state_dict(),
                        "optimizer_local": optimizer_local.state_dict(),
                        "scheduler_local": scheduler_local.state_dict(),
                        "iteration": it,
                        "avg_val_loss": avg_val_loss,
                    },
                    ckpt_path,
                )

                # Check if the current validation loss is the best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss  # Update the best validation loss
                    best_model_dir = os.path.join(log_dir, "best_model")
                    if not os.path.exists(best_model_dir):
                        os.makedirs(best_model_dir)
                    # Save the best model
                    torch.save(
                        {
                            "config": config,
                            "model": model.state_dict(),
                            "optimizer_global": optimizer_global.state_dict(),
                            "scheduler_global": scheduler_global.state_dict(),
                            "optimizer_local": optimizer_local.state_dict(),
                            "scheduler_local": scheduler_local.state_dict(),
                            "iteration": it,
                            "avg_val_loss": avg_val_loss,
                        },
                        os.path.join(best_model_dir, "best_model.pt"),
                    )
                    logger.info(f"New best model saved with loss {avg_val_loss}")

                    print(
                        f'best_model_path: {os.path.join(best_model_dir , "best_model.pt")}'
                    )

    except KeyboardInterrupt:
        logger.info("Terminating...")
