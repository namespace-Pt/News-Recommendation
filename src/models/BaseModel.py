import os
import math
import torch
import logging
import subprocess
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import scipy.stats as ss
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from utils.util import pack_results, compute_metrics



class BaseModel(nn.Module):
    def __init__(self, manager, name):
        super().__init__()

        self.his_size = manager.his_size
        self.sequence_length = manager.sequence_length
        self.hidden_dim = manager.hidden_dim
        self.device = manager.device
        self.rank = manager.rank
        self.world_size = manager.world_size

        # set all enable_xxx as attributes
        for k,v in vars(manager).items():
            if k.startswith("enable"):
                setattr(self, k, v)
        self.negative_num = manager.negative_num

        if name is None:
            name = type(self).__name__
        if manager.verbose is not None:
            self.name = "-".join([name, manager.verbose])
        else:
            self.name = name

        self.crossEntropy = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(self.name)


    def get_optimizer(self, manager, dataloader_length):
        optimizer = optim.Adam(self.parameters(), lr=manager.learning_rate)

        scheduler = None
        if manager.scheduler == "linear":
            total_steps = dataloader_length * manager.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = round(manager.warmup * total_steps),
                                            num_training_steps = total_steps)

        return optimizer, scheduler


    def _gather_tensors_variable_shape(self, local_tensor):
        """
        gather tensors from all gpus

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            all_tensors: concatenation of local_tensor in each process
        """
        all_tensors = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_tensors, local_tensor)
        all_tensors[self.rank] = local_tensor
        return torch.cat(all_tensors, dim=0)


    def _compute_gate(self, token_id, attn_mask, gate_mask, token_weight):
        """ gating by the weight of each token

        Returns:
            gated_token_ids: [B, K]
            gated_attn_masks: [B, K]
            gated_token_weight: [B, K]
        """
        if gate_mask is not None:
            keep_k_modifier = self.keep_k_modifier * (gate_mask.sum(dim=-1, keepdim=True) < self.k)
            pad_pos = ~((gate_mask + keep_k_modifier).bool())   # B, L
            token_weight = token_weight.masked_fill(pad_pos, -float('inf'))

            gated_token_weight, gated_token_idx = token_weight.topk(self.k)
            gated_token_weight = torch.softmax(gated_token_weight, dim=-1)
            gated_token_id = token_id.gather(dim=-1, index=gated_token_idx)
            gated_attn_mask = attn_mask.gather(dim=-1, index=gated_token_idx)

        # heuristic gate
        else:
            if token_id.dim() == 2:
                gated_token_id = token_id[:, 1: self.k + 1]
                gated_attn_mask = attn_mask[:, 1: self.k + 1]
            else:
                gated_token_id = token_id[:, :, 1: self.k + 1]
                gated_attn_mask = attn_mask[:, :, 1: self.k + 1]
            gated_token_weight = None

        return gated_token_id, gated_attn_mask, gated_token_weight


    @torch.no_grad()
    def dev(self, manager, loaders, load=True, log=False):
        self.eval()
        if load:
            manager.load(self)

        labels, preds = self._dev(manager, loaders)

        if self.rank == 0:
            metrics = compute_metrics(labels, preds, manager.metrics)
            metrics["main"] = metrics["auc"]
            self.logger.info(metrics)
            if log:
                manager._log(self.name, metrics)
        else:
            metrics = None

        if manager.distributed:
            dist.barrier(device_ids=[self.device])

        return metrics


    @torch.no_grad()
    def test(self, manager, loaders, load=True, log=False):
        self.eval()
        if load:
            manager.load(self)

        preds = self._test(manager, loaders)

        if manager.rank == 0:
            save_dir = "data/cache/results/{}/{}/{}".format(self.name, manager.scale, os.path.split(manager.checkpoint)[-1])
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir + "/prediction.txt"

            index = 1
            with open(save_path, "w") as f:
                for pred in preds:
                    array = np.asarray(pred)
                    rank_list = ss.rankdata(1 - array, method="min")
                    line = str(index) + " [" + ",".join([str(i) for i in rank_list]) + "]" + "\n"
                    f.write(line)
                    index += 1
            try:
                subprocess.run(f"zip -j {os.path.join(save_dir, 'prediction.zip')} {save_path}", shell=True)
            except:
                self.logger.warning("Zip Command Not Found! Skip zipping.")
            self.logger.info("written to prediction at {}!".format(save_path))

        if manager.distributed:
            dist.barrier(device_ids=[self.device])


    @torch.no_grad()
    def inspect(self, manager, loaders):
        assert hasattr(self, "weighter")

        tokenizer = AutoTokenizer.from_pretrained(manager.plm_dir)
        loader_news = loaders["news"]
        for i, x in enumerate(loader_news):
            token_ids = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
            try:
                gate_mask = x['cdd_gate_mask'].to(self.device)
            except:
                gate_mask = None
            token_weight = self.weighter(token_ids, attn_mask)
            if token_weight is not None:
                gated_token_ids, gated_attn_masks, gated_token_weights = self._compute_gate(token_ids, attn_mask, gate_mask, token_weight)
                for token_id, gated_token_id, gated_token_weight in zip(token_ids.tolist(), gated_token_ids.tolist(), gated_token_weights.tolist()):
                    token = tokenizer.convert_ids_to_tokens(token_id)
                    gated_token = tokenizer.convert_ids_to_tokens(gated_token_id)
                    print("-"*10 + "news text" + "-"*10)
                    print(tokenizer.decode(token_id))
                    print("-"*10 + "gated tokens" + "-"*10)
                    line = "; ".join([f"{i} ({round(p, 3)})" for i, p in zip(gated_token, gated_token_weight)])
                    print(line)
                    input()
            else:
                gated_token_ids, gated_attn_masks, gated_token_weights = self._compute_gate(token_ids, attn_mask, gate_mask, token_weight)
                for token_id, gated_token_id in zip(token_ids.tolist(), gated_token_ids.tolist()):
                    token = tokenizer.convert_ids_to_tokens(token_id)
                    gated_token = tokenizer.convert_ids_to_tokens(gated_token_id)
                    print("-"*10 + "news text" + "-"*10)
                    print(tokenizer.decode(token_id))
                    print("-"*10 + "gated tokens" + "-"*10)
                    line = " ".join([i for i in gated_token])
                    print(line)
                    input()



class TwoTowerBaseModel(BaseModel):
    def __init__(self, manager, name=None):
        """
        base class for two tower models (news encoder and user encoder), which we can cache all news and user representations in advance and speed up inference
        """
        super().__init__(manager, name)


    def _compute_logits(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        return score


    def _encode_news(self, x, cdd=True):
        if cdd:
            token_id = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
        news_token_embedding, news_embedding = self.newsEncoder(token_id, attn_mask)
        return news_token_embedding, news_embedding


    def _encode_user(self, x=None, his_news_embedding=None, his_mask=None):
        if x is None:
            user_embedding = self.userEncoder(his_news_embedding, his_mask=his_mask)
        else:
            _, his_news_embedding = self._encode_news(x, cdd=False)
            user_embedding = self.userEncoder(his_news_embedding, his_mask=x["his_mask"].to(self.device))
        return user_embedding


    def forward(self, x):
        _, cdd_news_embedding = self._encode_news(x)
        user_embedding = self._encode_user(x)

        logits = self._compute_logits(cdd_news_embedding, user_embedding)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss


    def infer(self, x):
        """
        infer logits with cache when evaluating; subclasses may adjust this function in case the user side encoding is different
        """
        cdd_idx = x["cdd_idx"].to(self.device, non_blocking=True)
        his_idx = x["his_idx"].to(self.device, non_blocking=True)
        cdd_embedding = self.news_embeddings[cdd_idx]
        his_embedding = self.news_embeddings[his_idx]
        user_embedding = self._encode_user(his_news_embedding=his_embedding, his_mask=x['his_mask'].to(self.device))
        logits = self._compute_logits(cdd_embedding, user_embedding)
        return logits


    @torch.no_grad()
    def encode_news(self, manager, loader_news):
        # every process holds the same copy of news embeddings
        news_embeddings = torch.zeros((len(loader_news.dataset), self.hidden_dim), device=self.device)

        # only encode news on the master node to avoid any problems possibly raised by gatherring
        if manager.rank == 0:
            start_idx = end_idx = 0
            for i, x in enumerate(tqdm(loader_news, ncols=80, desc="Encoding News")):
                _, news_embedding = self._encode_news(x)

                end_idx = start_idx + news_embedding.shape[0]
                news_embeddings[start_idx: end_idx] = news_embedding
                start_idx = end_idx

                if manager.debug:
                    if i > 5:
                        break
        # broadcast news embeddings to all gpus
        if manager.distributed:
            dist.broadcast(news_embeddings, 0)

        self.news_embeddings = news_embeddings


    def _dev(self, manager, loaders):
        self.encode_news(manager, loaders["news"])

        impr_indices = []
        masks = []
        labels = []
        preds = []

        for i, x in enumerate(tqdm(loaders["dev"], ncols=80, desc="Predicting")):
            logits = self.infer(x)

            masks.extend(x["cdd_mask"].tolist())
            impr_indices.extend(x["impr_index"].tolist())
            labels.extend(x["label"].tolist())
            preds.extend(logits.tolist())

        if manager.distributed:
            dist.barrier(device_ids=[self.device])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indices, masks, labels, preds))

            if self.rank == 0:
                impr_indices = []
                masks = []
                labels = []
                preds = []
                for output in outputs:
                    impr_indices.extend(output[0])
                    masks.extend(output[1])
                    labels.extend(output[2])
                    preds.extend(output[3])

                masks = np.asarray(masks, dtype=np.bool8)
                labels = np.asarray(labels, dtype=np.int32)
                preds = np.asarray(preds, dtype=np.float32)
                labels, preds = pack_results(impr_indices, masks, labels, preds)

        else:
            masks = np.asarray(masks, dtype=np.bool8)
            labels = np.asarray(labels, dtype=np.int32)
            preds = np.asarray(preds, dtype=np.float32)
            labels, preds = pack_results(impr_indices, masks, labels, preds)

        return labels, preds


    def _test(self, manager, loaders):
        self.encode_news(manager, loaders["news"])

        impr_indices = []
        masks = []
        preds = []

        for i, x in enumerate(tqdm(loaders["test"], ncols=80, desc="Predicting")):
            logits = self.infer(x)

            masks.extend(x["cdd_mask"].tolist())
            impr_indices.extend(x["impr_index"].tolist())
            preds.extend(logits.tolist())

        if manager.distributed:
            dist.barrier(device_ids=[self.device])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indices, masks, preds))

            if self.rank == 0:
                impr_indices = []
                masks = []
                preds = []
                for output in outputs:
                    impr_indices.extend(output[0])
                    masks.extend(output[1])
                    preds.extend(output[2])

                masks = np.asarray(masks, dtype=np.bool8)
                preds = np.asarray(preds, dtype=np.float32)
                preds, = pack_results(impr_indices, masks, preds)

        else:
            masks = np.asarray(masks, dtype=np.bool8)
            preds = np.asarray(preds, dtype=np.float32)
            preds, = pack_results(impr_indices, masks, preds)

        return preds
