import torch
import torch.nn as nn
import torch.optim as optim

from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import re
import os
import datetime
import logging
import subprocess

import numpy as np
import scipy.stats as ss
from tqdm import tqdm

from utils.utils import cal_metric, parameter

logger = logging.getLogger(__name__)

hparam_list = ["name", "scale","his_size", "k","threshold","learning_rate"]
param_list = []


class BaseModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.scale = hparams['scale']
        # self.attrs = hparams['attrs']
        # FIXME, checkpoint is not clear
        # self.checkpoint = hparams['checkpoint']


        self.cdd_size = (hparams['npratio'] +
                         1) if hparams['npratio'] > 0 else 1
        self.batch_size = hparams['batch_size']
        self.his_size = hparams['his_size']
        self.device = hparams['device']
        self.learning_rate = hparams['learning_rate']

        self.name = 'default'

    def save(self, epoch, step, optimizers=[]):
        """
            shortcut for saving the model and optimizer
        """
        # parse checkpoint
        # if "checkpoint" in hparams:
        #     save_path = "data/model_params/{}/{}_epoch{}_step{}_ck{}_[hs={},topk={}].model".format(
        #         hparams["name"], hparams["scale"], epoch, step, hparams["checkpoint"], hparams["his_size"], hparams["k"], ",".join(hparams["attrs"]))
        # else:

        save_path = "data/model_params/{}/{}_epoch{}_step{}_[hs={},topk={}].model".format(
            self.name, self.scale, epoch, step, self.his_size, self.k)

        state_dict = self.state_dict()

        if re.search("pipeline", self.name):
            state_dict = {k: v for k, v in state_dict.items() if k not in [
                "encoder.news_repr.weight", "encoder.news_embedding.weight"]}

        save_dict = {}
        save_dict["model"] = state_dict

        if len(optimizers) > 1:
            save_dict["optimizer"] = optimizers[0].state_dict()
            save_dict["optimizer_embedding"] = optimizers[1].state_dict()
        else:
            save_dict["optimizer"] = optimizers[0].state_dict()

        # if schedulers:
        #     if len(schedulers) > 1:
        #         save_dict["scheduler"] = schedulers[0].state_dict()
        #         save_dict["scheduler_embedding"] = schedulers[1].state_dict()
        #     else:
        #         save_dict["scheduler"] = schedulers[0].state_dict()

        torch.save(save_dict, save_path)
        logger.info("saved model of step {}, epoch {} at {}".format(
            step, epoch, save_path))


    def load(self, epoch, step, optimizers=None):
        """
            shortcut for loading model and optimizer parameters
        """

        save_path = "data/model_params/{}/{}_epoch{}_step{}_[hs={},topk={}].model".format(
            self.name, self.scale, epoch, step, self.his_size, self.k)

        state_dict = torch.load(save_path, map_location=self.device)
        if re.search("pipeline",self.name):
            logger.info("loading in pipeline")
            self.load_state_dict(state_dict["model"], strict=False)
        else:
            self.load_state_dict(state_dict["model"])

        if optimizers:
            optimizers[0].load_state_dict(state_dict["optimizer"])
            if len(optimizers) > 1:
                optimizers[1].load_state_dict(state_dict["optimizer_embedding"])

        # if schedulers:
        #     schedulers[0].load_state_dict(state_dict["scheduler"])
        #     if len(schedulers) > 1:
        #         schedulers[1].load_state_dict(state_dict["scheduler_embedding"])

        logger.info("Loading model from {}...".format(save_path))


    def _log(self, res, hparams):
        """
            wrap logging
        """
        logger.info("evaluation results:{}".format(res))
        with open("performance.log", "a+") as f:
            d = {}
            for k, v in hparams.items():
                if k in hparam_list:
                    d[k] = v
            for name, param in self.named_parameters():
                if name in param_list:
                    d[name] = tuple(param.shape)

            f.write(str(d)+"\n")
            f.write(str(res) + "\n")
            f.write("\n")


    def _get_loss(self):
        """
            get loss function for model
        """
        if self.cdd_size > 1:
            loss = nn.NLLLoss()
        else:
            loss = nn.BCELoss()

        return loss


    def _get_optim(self, hparams, loader_train):
        """
            get optimizer/scheduler
        """

        learning_rate = hparams['learning_rate']

        if hparams["spadam"]:
            optimizer_param = optim.Adam(
                parameter(self, ["encoder.embedding.weight","encoder.user_embedding.weight"], exclude=True), lr=learning_rate)
            optimizer_embedding = optim.SparseAdam(
                list(parameter(self, ["encoder.embedding.weight","encoder.user_embedding.weight"])), lr=learning_rate)

            optimizers = (optimizer_param, optimizer_embedding)

            if hparams["schedule"] == "linear":
                scheduler_param =get_linear_schedule_with_warmup(optimizer_param, num_warmup_steps=0, num_training_steps=len(loader_train) * hparams["epochs"])
                scheduler_embedding =get_linear_schedule_with_warmup(optimizer_embedding, num_warmup_steps=0, num_training_steps=len(loader_train) * hparams["epochs"])
                schedulers = (scheduler_param, scheduler_embedding)
            else:
                schedulers = []

        else:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            optimizers = (optimizer,)
            if hparams["schedule"] == "linear":
                scheduler =get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(loader_train) * hparams["epochs"])
                schedulers = (scheduler,)
            else:
                schedulers = []

        # FIXME
        # if "checkpoint" in hparams:
        #     logger.info("loading checkpoint...")
        #     ck = hparams["checkpoint"].split(",")
        #     # modify the epoch so the model can be properly saved
        #     self.load(self, ck[0], ck[1], optimizers)

        return optimizers, schedulers


    def _get_label(self, x):
        """
            parse labels to label indexes, used in NLLoss
        """
        if self.cdd_size > 1:
            index = torch.arange(0, self.cdd_size, device=self.device).expand(
                self.batch_size, -1)
            label = x["labels"] == 1
            label = index[label]
        else:
            label = x["labels"]

        return label


    @torch.no_grad()
    def _run_eval(self, dataloader, smoothing=1):
        """ making prediction and gather results into groups according to impression_id

        Args:
            dataloader(torch.utils.data.DataLoader): provide data

        Returns:
            all_impr_ids: impression ids after group
            all_labels: labels after group.
            all_preds: preds after group.
        """
        preds = []
        labels = []
        imp_indexes = []

        for batch_data_input in tqdm(dataloader, smoothing=smoothing):
            pred = self(batch_data_input).squeeze(dim=-1).tolist()
            preds.extend(pred)
            label = batch_data_input["labels"].squeeze(dim=-1).tolist()
            labels.extend(label)
            imp_indexes.extend(batch_data_input["impression_index"])

        all_keys = list(set(imp_indexes))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(labels, preds, imp_indexes):
            group_labels[k].append(l)
            group_preds[k].append(p)

        all_impr_ids = group_labels.keys()
        all_labels = list(group_labels.values())
        all_preds = list(group_preds.values())

        return all_impr_ids, all_labels, all_preds


    def evaluate(self, hparams, dataloader, load=False, log=True):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            self(nn.Module)
            hparams(dict)
            dataloader(torch.utils.data.DataLoader)
            loading(bool): whether to load model
            log(bool): whether to log

        Returns:
            res(dict): A dictionary contains evaluation metrics.
        """
        # multiple evaluation steps
        if len(hparams["save_step"]) > 1:
            for step in hparams["save_step"][1:]:
                command = re.sub(",".join([str(i) for i in hparams["save_step"]]), str(
                    step), hparams["command"])
                subprocess.Popen(command, shell=True)

        cdd_size = self.cdd_size
        self.cdd_size = 1
        self.eval()

        if load:
            self.load(hparams, hparams["epochs"], hparams["save_step"][0])

        logger.info("evaluating...")

        imp_indexes, labels, preds = self._run_eval(dataloader)

        res = cal_metric(labels, preds, hparams["metrics"].split(","))

        if log:
            res["epoch"] = hparams["epochs"]
            res["step"] = hparams["save_step"][0]

            self._log(res, hparams)

        self.train()
        self.cdd_size = cdd_size

        return res

    def _run_train(self, dataloader, optimizers, loss_func, epochs, schedulers=None, writer=None, interval=100, save_step=None):
        """ train model and print loss meanwhile
        Args:
            dataloader(torch.utils.data.DataLoader): provide data
            optimizer(list of torch.nn.optim): optimizer for training
            loss_func(torch.nn.Loss): loss function for training
            epochs(int): training epochs
            schedulers
            writer(torch.utils.tensorboard.SummaryWriter): tensorboard writer
            interval(int): within each epoch, the interval of training steps to display loss
            save_step(int): how many steps to save the model
        Returns:
            model: trained model
        """
        total_loss = 0
        total_steps = 0

        for epoch in range(epochs):
            epoch_loss = 0
            tqdm_ = tqdm(dataloader)
            for step, x in enumerate(tqdm_):

                for optimizer in optimizers:
                    optimizer.zero_grad()

                pred = self(x)
                label = self._get_label(x)

                loss = loss_func(pred, label)

                epoch_loss += loss
                total_loss += loss

                loss.backward()

                for optimizer in optimizers:
                    optimizer.step()

                if schedulers:
                    for scheduler in schedulers:
                        scheduler.step()

                if step % interval == 0:

                    tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                    if writer:
                        for name, param in self.named_parameters():
                            writer.add_histogram(name, param, step)

                        writer.add_scalar("data_loss",
                                        total_loss/total_steps)

                if save_step:
                    if step % save_step == 0 and step > 0:
                        self.save(epoch+1, step, optimizers)

                total_steps += 1

            if writer:
                writer.add_scalar("epoch_loss", epoch_loss, epoch)

            self.save(epoch+1, 0, optimizers)


    def fit(self, hparams, loaders, tb=False):
        """ wrap training process

        Args:
            model(torch.nn.Module): the model to be trained
            loaders(list): list of torch.utils.data.DataLoader
            hparams(dict): hyper paramaters
            en: shell parameter
        """
        self.train()
        writer = None

        if tb:
            writer = SummaryWriter("data/tb/{}/{}/{}/".format(
                hparams["name"], hparams["scale"], datetime.now().strftime("%Y%m%d-%H")))

        # in case the folder does not exists, create one
        save_derectory = "data/model_params/{}".format(hparams["name"])
        if not os.path.exists(save_derectory):
            os.mkdir(save_derectory)

        logger.info("training...")
        loss_func = self._get_loss()
        optimizers, schedulers = self._get_optim(hparams, loaders[0])

        self._run_train(loaders[0], optimizers, loss_func, hparams['epochs'], schedulers=schedulers,
                        writer=writer, interval=hparams["interval"], save_step=hparams["save_step"][0])


    def _run_tune(self, loaders, optimizers, loss_func, hparams, schedulers=[], writer=None, interval=100, save_step=None):
        """ train model and evaluate on validation set once every save_step
        Args:
            dataloader(torch.utils.data.DataLoader): provide data
            optimizer(list of torch.nn.optim): optimizer for training
            loss_func(torch.nn.Loss): loss function for training
            hparams
            schedulers
            writer(torch.utils.tensorboard.SummaryWriter): tensorboard writer
            interval(int): within each epoch, the interval of training steps to display loss
            save_step(int): how many steps to save the model
        Returns:
            model: trained model
        """
        total_loss = 0
        total_steps = 0

        best_res = {"auc":0}

        for epoch in range(hparams['epochs']):
            epoch_loss = 0
            tqdm_ = tqdm(loaders[0], smoothing=0)
            for step, x in enumerate(tqdm_):

                for optimizer in optimizers:
                    optimizer.zero_grad()

                pred = self(x)
                label = self._get_label(x)

                loss = loss_func(pred, label)

                epoch_loss += loss
                total_loss += loss

                loss.backward()

                for optimizer in optimizers:
                    optimizer.step()

                if schedulers:
                    for scheduler in schedulers:
                        scheduler.step()

                if step % interval == 0:

                    tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                    if writer:
                        for name, param in self.named_parameters():
                            writer.add_histogram(name, param, step)

                        writer.add_scalar("data_loss",
                                        total_loss/total_steps)

                if step % save_step == 0 and step > 0 and epoch > 1:
                    print("\n")
                    with torch.no_grad():
                        result = self.evaluate(hparams, loaders[1], log=False)
                        result["epoch"] = epoch+1
                        result["step"] = step

                        logger.info("current result of {} is {}".format(self.name, result))
                        if result["auc"] > best_res["auc"]:
                            best_res = result
                            logger.info("best result till now is {}".format(best_res))
                            self.save(epoch+1, step, optimizers)
                            self._log(result, hparams)

                        elif result["auc"] - best_res["auc"] < -0.05:
                            logger.info("model is overfitting, the result is {}, force shutdown".format(result))
                            return best_res

                total_steps += 1

            if writer:
                writer.add_scalar("epoch_loss", epoch_loss, epoch)

        return best_res


    def tune(self, hparams, loaders, tb=False):
        """ train and evaluate sequentially

        Args:
            model(torch.nn.Module): the model to be trained
            loaders(list): list of torch.utils.data.DataLoader
            hparams(dict): hyper paramaters
            en: shell parameter
        """

        self.train()
        writer = None

        if tb:
            writer = SummaryWriter("data/tb/{}/{}/{}/".format(
                self.name, self.scale, datetime.now().strftime("%Y%m%d-%H")))

        # in case the folder does not exists, create one
        save_derectory = "data/model_params/{}".format(self.name)
        if not os.path.exists(save_derectory):
            os.mkdir(save_derectory)

        logger.info("training...")
        loss_func = self._get_loss()
        optimizers, schedulers = self._get_optim(hparams, loaders[0])

        res = self._run_tune(loaders, optimizers, loss_func, hparams, schedulers=schedulers,
                        writer=writer, interval=hparams["interval"], save_step=int(len(loaders[0])/hparams["val_freq"])-1)

        self._log(res, hparams)

    @torch.no_grad()
    def test(self, hparams, loader_test):
        """ test the model on test dataset of MINDlarge

        Args:
            model
            hparams
            loader_test: DataLoader of MINDlarge_test
        """
        self.load(hparams["epochs"], hparams["save_step"][0])

        logger.info("testing...")
        self.cdd_size = 1
        self.eval()

        with open("performance.log", "a+") as f:
            d = {}
            for k, v in self.named_buffers():
                if k in hparam_list:
                    d[k] = v
            for name, param in self.named_parameters():
                if name in param_list:
                    d[name] = tuple(param.shape)

            f.write(str(d)+"\n")
            f.write("\n")
            f.write("\n")

        save_path = "data/results/prediction={}_{}_epoch{}_step{}_[hs={},topk={}].txt".format(
            self.name, self.scale, hparams["epochs"], hparams["save_step"][0], self.his_size, self.k)
        with open(save_path, "w") as f:
            preds = []
            imp_indexes = []
            for x in tqdm(loader_test, smoothing=0):
                preds.extend(self.forward(x).tolist())
                imp_indexes.extend(x["impression_index"])

            all_keys = list(set(imp_indexes))
            all_keys.sort()
            group_preds = {k: [] for k in all_keys}

            for i, p in zip(imp_indexes, preds):
                group_preds[i].append(p)

            for k, v in group_preds.items():
                array = np.asarray(v)
                rank_list = ss.rankdata(1 - array, method="ordinal")
                line = str(k) + " [" + ",".join([str(i)
                                                for i in rank_list]) + "]" + "\n"
                f.write(line)

        logger.info("written to prediction at {}!".format(save_path))

    @torch.no_grad()
    def encode(self, loader):
        """
            Encode news of hparams["scale"] in each mode, currently force to encode dev dataset
        """

        # very important
        self.eval()
        news_num_dict = {
            "demo": {
                "train": 51282,
                "dev": 42416
            },
            "small": {
                "train": 51282,
                "dev": 42416
            },
            "large": {
                "train": 101527,
                "dev": 72023,
                "test": 120961
            }
        }

        scale = loader.dataset.scale
        mode = loader.dataset.mode
        news_num = news_num_dict[scale][mode]

        encoder = self.encoder

        news_reprs = torch.zeros((news_num + 1, encoder.hidden_dim))
        news_embeddings = torch.zeros(
            (news_num + 1, encoder.signal_length, encoder.level, encoder.hidden_dim))

        start_pos = 0
        for x in tqdm(loader):
            if x['candidate_title'].shape[0] != self.batch_size:
                self.batch_size = x['candidate_title'].shape[0]

            news = x['candidate_title'].long()
            news_embedding, news_repr = self.encoder(
                news,
                news_id=x['cdd_id'].long(),
                attn_mask=x['candidate_title_pad'])

            news_reprs[start_pos:start_pos+self.batch_size] = news_repr
            news_embeddings[start_pos:start_pos+self.batch_size] = news_embedding

            start_pos += self.batch_size

        torch.save(news_reprs, "data/tensors/news_repr_{}_{}-[{}].tensor".format(
            scale, mode, self.name))
        torch.save(news_embeddings, "data/tensors/news_embedding_{}_{}-[{}].tensor".format(
            scale, mode, self.name))
        del news_reprs
        del news_embeddings
        logging.info("encoded news saved in data/tensors/news_**_{}_{}-[{}].tensor".format(scale, mode, self.name))