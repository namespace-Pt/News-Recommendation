import os
import torch
import random
import logging
import argparse
import transformers
import smtplib
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.util import Sequential_Sampler, load_pickle, save_pickle, download_plm
from utils.dataset import *

logger = logging.getLogger("Manager")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s (%(name)s) %(message)s")
# prevent warning of transformers
transformers.logging.set_verbosity_error()
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)



class Manager():
    """
    the class to handle dataloader preperation, model training/evaluation
    """
    def __init__(self, config=None, notebook=False):
        """
        set hyper parameters

        Args:
            config: some extra configuration
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--scale", dest="scale", help="data scale", type=str, choices=["demo", "small", "large", "whole"], default="large")
        parser.add_argument("-e", "--epoch", dest="epochs", help="epochs to train the model", type=int, default=10)
        parser.add_argument("-m", "--mode", dest="mode", help="choose mode", default="train")
        parser.add_argument("-d", "--device", dest="device", help="gpu index, -1 for cpu", type=int, default=0)
        parser.add_argument("-bs", "--batch-size", dest="batch_size", help="batch size in training", type=int, default=32)
        parser.add_argument("-bse", "--batch-size-eval", dest="batch_size_eval", help="batch size in encoding", type=int, default=200)
        # parser.add_argument("-dl", "--dataloaders", dest="dataloaders", help="training dataloaders", nargs="+", action="extend", choices=["train", "dev", "news", "behaviors"], default=["train", "dev", "news"])

        parser.add_argument("-ck","--checkpoint", dest="checkpoint", help="load the model from checkpoint before training/evaluating", type=str, default="none")
        parser.add_argument("-vs","--validate-step", dest="validate_step", help="evaluate and save the model every step", type=str, default="0")
        parser.add_argument("-hst","--hold-step", dest="hold_step", help="don't evaluate until reaching hold step", type=str, default="0")
        parser.add_argument("-sav","--save-at-validate", dest="save_at_validate", help="save the model every time of validating", action="store_true", default=False)
        parser.add_argument("-vb","--verbose", dest="verbose", help="variant's name", type=str, default=None)
        parser.add_argument("--metrics", dest="metrics", help="metrics for evaluating the model", nargs="+", action="extend", default=["auc", "mean_mrr", "ndcg@5", "ndcg@10"])

        parser.add_argument("-hs", "--his_size", dest="his_size",help="history size", type=int, default=50)
        parser.add_argument("-is", "--impr_size", dest="impr_size", help="impression size for evaluating", type=int, default=20)
        parser.add_argument("-nn", "--negative-num", dest="negative_num", help="number of negatives", type=int, default=4)
        parser.add_argument("-dp", "--dropout-p", dest="dropout_p", help="dropout probability", type=float, default=0.1)
        parser.add_argument("-lr", "--learning-rate", dest="learning_rate", help="learning rate", type=float, default=1e-5)
        parser.add_argument("-sch", "--scheduler", dest="scheduler", help="choose schedule scheme for optimizer", choices=["linear","none"], default="none")
        parser.add_argument("--warmup", dest="warmup", help="warmup steps of scheduler", type=float, default=0.1)

        parser.add_argument("-pth", "--preprocess-threads", dest="preprocess_threads", help="thread number in preprocessing", type=int, default=32)
        parser.add_argument("-dr", "--data-root", dest="data_root", default="../../../Data")
        parser.add_argument("-cr", "--cache-root", dest="cache_root", default="data/cache")

        parser.add_argument("-tl", "--title-length", dest="title_length", type=int, default=32)
        parser.add_argument("-al", "--abs-length", dest="abs_length", type=int, default=64)
        parser.add_argument("-mtl", "--max-title-length", dest="max_title_length", type=int, default=64)
        parser.add_argument("-mal", "--max-abs-length", dest="max_abs_length", type=int, default=128)

        parser.add_argument("-ef", "--enable-fields", dest="enable_fields", help="text fields to model", nargs="+", action="extend", choices=["title", "abs"], default=[])
        parser.add_argument("-eg", "--enable-gate", dest="enable_gate", help="way to gate tokens", type=str, choices=["weight", "none", "bm25", "first", "keybert", "random"], default="weight")

        parser.add_argument("-ne", "--news-encoder", dest="newsEncoder", default="cnn")
        parser.add_argument("-ue", "--user-encoder", dest="userEncoder", default="rnn")
        parser.add_argument("-wt", "--weighter", dest="weighter", default="cnn")

        parser.add_argument("-hd", "--hidden-dim", dest="hidden_dim", type=int, default=768)
        parser.add_argument("-ged", "--gate-embedding-dim", dest="gate_embedding_dim", type=int, default=300)
        parser.add_argument("-ghd", "--gate-hidden-dim", dest="gate_hidden_dim", type=int, default=300)
        parser.add_argument("-hn", "--head-num", dest="head_num", help="attention head number of tranformer model", type=int, default=12)

        parser.add_argument("-k", dest="k", help="gate number", type=int, default=4)

        parser.add_argument("-plm", dest="plm", help="short name of pre-trained language models", type=str, default="bert")

        parser.add_argument("--seed", dest="seed", default=3407, type=int)
        parser.add_argument("-ws", "--world-size", dest="world_size", help="gpu number", type=int, default=1)
        parser.add_argument("-br", "--base-rank", dest="base_rank", help="base device index", type=int, default=0)

        parser.add_argument("--debug", dest="debug", help="debug mode", action="store_true", default=False)

        if not notebook:
            if config:
                # different default settings per model
                parser.set_defaults(**config)
            args = vars(parser.parse_args())
        else:
            args = dict(vars(config))

        if args['device'] == -1:
            args['device'] = "cpu"
        # used for checking
        if args["debug"]:
            args["hold_step"] = "0"
            args["validate_step"] = "2"
        # default to load best checkpoint
        if args["mode"] != "train":
            if args["checkpoint"] == "none":
                args["checkpoint"] = "best"
        sequence_length = 0
        if "title" in args["enable_fields"]:
            sequence_length += args["title_length"]
        if "abs" in args["enable_fields"]:
            sequence_length += args["abs_length"]
        if sequence_length == 0:
            raise ValueError("Include at least one field!")
        else:
            args["sequence_length"] = sequence_length

        if args["enable_gate"] in ["first", "bm25", "keybert", "random"]:
            args["weighter"] = "first"

        if args['seed'] is not None:
            seed = args['seed']
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        for k,v in args.items():
            if not k.startswith("__"):
                setattr(self, k, v)

        plm_map_dimension = {
            "bert": 768,
            "deberta": 768,
            "unilm": 768
        }
        plm_special_token_id_map = {
            "bert":{
                "[PAD]": 0,
                "[CLS]": 101,
                "[SEP]": 102,
                "punctuations": {},
            },
            "deberta":{
                "[PAD]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "punctuations": {},
            },
            "unilm":{
                "[PAD]": 0,
                "[CLS]": 101,
                "[SEP]": 102,
                "punctuations": {},
            },
        }
        vocab_size_map = {
            "bert": 30522
        }
        dataloader_map = {
            "train": ["train", "dev", "news"],
            "dev": ["dev", "news"],
            "test": ["test", "news"],
            "inspect": ["news"]
        }
        news_cache_dir_map = {
            "none": "original",
            "weight": "original",
            "first": "original",
            "bm25": "bm25",
            "keybert": "keybert",
            "random": "random"
        }
        news_file_map = {
            "none": "news.tsv",
            "weight": "news.tsv",
            "first": "news.tsv",
            "bm25": "bm25.tsv",
            "keybert": "keybert.tsv",
            "random": "random.tsv"
        }

        self.plm_dir = os.path.join(self.data_root, "PLM", self.plm)
        self.plm_dim = plm_map_dimension[self.plm]
        self.special_token_ids = plm_special_token_id_map[self.plm]
        self.vocab_size = vocab_size_map[self.plm]
        self.news_nums = {
            "MINDdemo_train": 51282,
            "MINDdemo_dev": 42416,
            "MINDsmall_train": 51282,
            "MINDsmall_dev": 42416,
            "MINDlarge_train": 101527,
            "MINDlarge_dev": 72023,
            "MINDlarge_test": 120961,
        }
        self.dataloaders = dataloader_map[self.mode]
        self.news_cache_dir = news_cache_dir_map[self.enable_gate]
        self.news_file = news_file_map[self.enable_gate]

        self.distributed = self.world_size > 1
        self.exclude_hparams = set(["news_nums", "vocab_size_map", "metrics", "plm_dim", "plm_dir", "data_root", "cache_root", "distributed", "exclude_hparams", "rank", "epochs", "mode", "debug", "special_token_ids", "validate_step", "hold_step", "exclude_hparams", "device", "save_at_validate", "preprocess_threads", "base_rank", "max_title_length", "max_abs_length"])

        logger.info("Hyper Parameters are:\n{}\n".format({k:v for k,v in args.items() if k[0:2] != "__" and k not in self.exclude_hparams}))


    def setup(self, rank):
        """
        set up distributed training and fix seeds
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

        if self.world_size > 1:
            os.environ["NCCL_DEBUG"] = "WARN"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(12355 + self.base_rank)

            # initialize the process group
            # set timeout to inf to prevent timeout error
            dist.init_process_group("nccl", rank=rank, world_size=self.world_size, timeout=timedelta(0, 1000000))

            # manager.rank will be invoked in creating DistributedSampler
            self.rank = rank
            # manager.device will be invoked in the model
            self.device = rank + self.base_rank

        else:
            # one-gpu
            self.rank = 0

        if self.device != "cpu":
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            # essential to make all_gather_object work properly
            torch.cuda.set_device(self.device)


    def prepare(self):
        """
        prepare dataloader for training/evaluating

        Returns:
            loaders: list of dataloaders
                train: default/triple
                (passage)
                (query)
                (rerank)
        """
        if self.rank == 0:
            # download plm once
            if os.path.exists(self.plm_dir):
                pass
            else:
                logger.info("downloading PLMs...")
                download_plm(self.plm, self.plm_dir)
        if self.distributed:
            dist.barrier(device_ids=[self.device])

        loaders = {}

        # training dataloaders
        if "train" in self.dataloaders:
            dataset_train = MIND_Train(self)
            if self.distributed:
                sampler_train = DistributedSampler(dataset_train, num_replicas=self.world_size, rank=self.rank, seed=self.seed)
                shuffle = False
            else:
                sampler_train = None
                shuffle = True
            loaders["train"] = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler_train)

        if "dev" in self.dataloaders:
            dataset_dev = MIND_Dev(self)
            sampler_dev = Sequential_Sampler(len(dataset_dev), num_replicas=self.world_size, rank=self.rank)
            loaders["dev"] = DataLoader(dataset_dev, batch_size=self.batch_size_eval, sampler=sampler_dev, drop_last=False)

        if "test" in self.dataloaders:
            dataset_test = MIND_Test(self)
            sampler_test = Sequential_Sampler(len(dataset_test), num_replicas=self.world_size, rank=self.rank)
            loaders["test"] = DataLoader(dataset_test, batch_size=self.batch_size_eval, sampler=sampler_test, drop_last=False)

        if "news" in self.dataloaders:
            dataset_news = MIND_News(self)
            # no sampler
            loaders["news"] = DataLoader(dataset_news, batch_size=self.batch_size_eval, drop_last=False)

        return loaders


    def save(self, model, step, best=False):
        """
            shortcut for saving the model and optimizer
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        if best:
            save_path = f"data/ckpts/{model.name}/{self.scale}/best.model"
        else:
            save_path = f"data/ckpts/{model.name}/{self.scale}/{step}.model"

        logger.info("saving model at {}...".format(save_path))
        model_dict = model.state_dict()

        save_dict = {}
        save_dict["manager"] = {k:v for k,v in vars(self).items() if k[:2] != "__" and k not in self.exclude_hparams}
        save_dict["model"] = model_dict

        torch.save(save_dict, save_path)


    def load(self, model):
        """
            shortcut for loading model and optimizer parameters

        Args:
            model: nn.Module
            checkpoint: the checkpoint step to load, if checkpoint==0, default to load the best model, if
                it doesn't exist, do not load
            strict: whether to enable strict loading
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        checkpoint = self.checkpoint
        if checkpoint == "none":
            return
        elif os.path.isfile(checkpoint):
            save_path = checkpoint
        elif checkpoint == "best":
            save_path = f"data/ckpts/{model.name}/{self.scale}/best.model"
        else:
            save_path = f"data/ckpts/{model.name}/{self.scale}/{checkpoint}.model"

        if not os.path.exists(save_path):
            if self.rank == 0:
                logger.warning(f"Checkpoint {save_path} Not Found, Not Loading Any Checkpoints!")
            return

        if self.rank == 0:
            logger.info("loading model from {}...".format(save_path))

        state_dict = torch.load(save_path, map_location=torch.device(model.device))

        if self.rank == 0:
            current_manager = vars(self)
            for k,v in state_dict["manager"].items():
                try:
                    if v != current_manager[k] and k not in {"dataloaders", "checkpoint"}:
                        logger.info(f"manager settings {k} of the checkpoint is {v}, while it's {current_manager[k]} in current setting!")
                except KeyError:
                    logger.info(f"manager settings {k} not found!")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict["model"], strict=False)
        if self.rank == 0:
            if len(missing_keys):
                logger.warning(f"Missing Keys: {missing_keys}")
            if len(unexpected_keys):
                logger.warning(f"Unexpected Keys: {unexpected_keys}")


    def _log(self, model_name, metrics):
        """
            wrap logging
        """
        with open("performance.log", "a+") as f:
            d = {}
            for k, v in vars(self).items():
                if k not in self.exclude_hparams:
                    d[k] = v

            line = "{} : {}\n{}\n\n".format(model_name, str(d), str(metrics))
            f.write(line)

            try:
                from data.email import email,password
                subject = f"[PR] {model_name}"
                email_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                email_server.login(email, password)
                message = "Subject: {}\n\n{}".format(subject, line)
                email_server.sendmail(email, email, message)
                email_server.close()
            except:
                logger.info("error in connecting SMTP")


    def _train(self, model, loaders, validate_step, hold_step, optimizer, scheduler, save_at_validate=False):
        total_steps = 1
        loader_train = loaders["train"]

        best_res = {"main": -1.}

        if self.rank == 0:
            logger.info("training {}...".format(model.module.name if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.name))

        for epoch in range(self.epochs):
            epoch_loss = 0
            if self.distributed:
                try:
                    loader_train.sampler.set_epoch(epoch)
                except AttributeError:
                    if self.rank == 0:
                        logger.warning(f"{type(loader_train)} has no attribute 'sampler', make sure you're using Triple training dataloader")
            tqdm_ = tqdm(loader_train, ncols=120)

            for step, x in enumerate(tqdm_, 1):
                optimizer.zero_grad(set_to_none=True)
                loss = model(x)
                epoch_loss += float(loss)
                loss.backward()

                optimizer.step()
                if scheduler:
                    scheduler.step()

                if step % 5 == 0:
                    tqdm_.set_description("epoch: {:d}, step: {:d}, loss: {:.4f}".format(epoch + 1, step, epoch_loss / step))

                if total_steps > hold_step and total_steps % validate_step == 0:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        result = model.module.dev(self, loaders, load=False)
                    else:
                        result = model.dev(self, loaders, load=False)
                    # only the result of master node is useful
                    if self.rank == 0:
                        result["step"] = total_steps
                        if save_at_validate:
                            self.save(model, total_steps)

                        # save the best model checkpoint
                        if result["main"] >= best_res["main"]:
                            best_res = result
                            self.save(model, total_steps, best=True)
                            self._log(model.module.name if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.name, result)

                    # prevent SIGABRT
                    if self.distributed:
                        dist.barrier(device_ids=[self.device])
                    # continue training
                    model.train()

                total_steps += 1

        return best_res


    def train(self, model, loaders):
        """
        train the model
        """
        model.train()
        if self.rank == 0:
            # in case the folder does not exists, create one
            os.makedirs(f"data/ckpts/{model.module.name if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.name}/{self.scale}", exist_ok=True)

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            optimizer, scheduler = model.module.get_optimizer(self, len(loaders["train"]))
        else:
            optimizer, scheduler = model.get_optimizer(self, len(loaders["train"]))

        self.load(model)

        if self.validate_step[-1] == "e":
            # validate at the end of several epochs
            validate_step = round(len(loaders["train"]) * float(self.validate_step[:-1]))
        elif self.validate_step == "0":
            # validate at the end of every epoch
            validate_step = len(loaders["train"])
        else:
            # validate at certain steps
            validate_step = int(self.validate_step)
        if self.hold_step[-1] == "e":
            hold_step = int(len(loaders["train"]) * float(self.hold_step[:-1]))
        else:
            hold_step = int(self.hold_step)

        result = self._train(model, loaders, validate_step, hold_step, optimizer, scheduler=scheduler, save_at_validate=self.save_at_validate)

        if self.rank in [-1,0]:
            logger.info("Best result: {}".format(result))
            self._log(model.module.name if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.name, result)
