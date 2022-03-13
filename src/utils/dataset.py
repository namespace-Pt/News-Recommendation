import os
import logging
import subprocess
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from multiprocessing import Pool
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from utils.util import load_pickle, save_pickle, construct_uid2index, construct_nid2index, sample_news



class MIND(Dataset):
    def __init__(self, manager, data_dir, load_news=True, load_behaviors=True) -> None:
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.his_size = manager.his_size
        self.impr_size = manager.impr_size

        self.max_title_length = manager.max_title_length
        self.max_abs_length = manager.max_abs_length
        self.title_length = manager.title_length
        self.abs_length = manager.abs_length

        self.negative_num = manager.negative_num

        self.cache_root = manager.cache_root
        self.data_root = manager.data_root

        data_dir_name = data_dir.split("/")[-1]
        self.news_cache_root = os.path.join(manager.cache_root, "MIND", data_dir_name, "news")
        if "train" in data_dir_name:
            self.behaviors_cache_dir = os.path.join(manager.cache_root, "MIND", data_dir_name, "behaviors")
        else:
            # cache by impr size
            self.behaviors_cache_dir = os.path.join(manager.cache_root, "MIND", data_dir_name, "behaviors", str(self.impr_size))

        news_num = manager.news_nums[data_dir_name] + 1

        # set all enable_xxx as attributes
        for k,v in vars(manager).items():
            if k.startswith("enable"):
                setattr(self, k, v)

        if manager.rank == 0:
            if not os.path.exists(os.path.join(self.news_cache_root, manager.news_cache_dir, "title_token_ids.pkl")):
                news_path = os.path.join(data_dir, manager.news_file)
                cache_news(news_path, self.news_cache_root, manager)
            if not os.path.exists(os.path.join(self.behaviors_cache_dir, "behaviors.pkl")):
                nid2index = load_pickle(os.path.join(self.news_cache_root, "nid2index.pkl"))
                cache_behaviors(os.path.join(data_dir, "behaviors.tsv"), self.behaviors_cache_dir, nid2index, manager)

        if manager.distributed:
            dist.barrier(device_ids=[manager.device])

        if manager.rank == 0:
            self.logger.info(f"Loading Cache at {data_dir_name}")

        if load_news:
            pad_token_id = manager.special_token_ids["[PAD]"]
            sep_token_id = manager.special_token_ids["[SEP]"]
            cls_token_id = manager.special_token_ids["[CLS]"]
            punc_token_ids = manager.special_token_ids["punctuations"]

            # index=0 is padded news
            token_ids = [[] for _ in range(news_num)]
            self.sequence_length = manager.sequence_length

            start_idx = 0
            if "title" in self.enable_fields:
                title_token_ids = load_pickle(os.path.join(self.news_cache_root, manager.news_cache_dir, "title_token_ids.pkl"))
                for i, token_id in enumerate(title_token_ids, start=1):
                    token_id = token_id[start_idx: start_idx + self.title_length]
                    # use [SEP] to separate title and abstract
                    if len(token_id) > 2 - start_idx:
                        token_id[-1] = sep_token_id
                        token_ids[i].extend(token_id.copy())
                if start_idx == 0:
                    start_idx += 1

            if "abs" in self.enable_fields:
                abs_token_ids = load_pickle(os.path.join(self.news_cache_root, manager.news_cache_dir, "abs_token_ids.pkl"))
                for i, token_id in enumerate(abs_token_ids, start=1):
                    # offset to remove an extra [CLS]
                    token_id = token_id[start_idx: self.abs_length + start_idx]
                    # use [SEP] to separate abs and abstract
                    if len(token_id) > 2 - start_idx:
                        token_id[-1] = sep_token_id
                        token_ids[i].extend(token_id.copy())
                if start_idx == 0:
                    start_idx += 1

            attn_masks = np.zeros((news_num, self.sequence_length), dtype=np.int64)
            if self.enable_gate == "weight":
                gate_masks = np.zeros((news_num, self.sequence_length), dtype=np.int64)

            for i, token_id in enumerate(token_ids):
                s_len = len(token_id)
                if s_len < self.sequence_length:
                    token_ids[i] = token_id + [pad_token_id] * (self.sequence_length - s_len)
                attn_masks[i][:s_len] = 1
                if self.enable_gate == "weight":
                    # deduplicate and remove punctuations and remove special token ids
                    # token_set = set()
                    # for j, x in enumerate(token_id):
                    #     if x not in token_set and x != cls_token_id and x != sep_token_id and x not in punc_token_ids:
                    #         gate_masks[i, j] = 1
                    #         token_set.add(x)
                    for j, x in enumerate(token_id):
                        if x != cls_token_id and x != sep_token_id and x not in punc_token_ids:
                            gate_masks[i, j] = 1

            self.token_ids = np.asarray(token_ids, dtype=np.int64)
            self.attn_masks = np.asarray(attn_masks, dtype=np.int64)
            if self.enable_gate == "weight":
                self.gate_masks = np.asarray(gate_masks, dtype=np.int64)

        if load_behaviors:
            behaviors = load_pickle(os.path.join(self.behaviors_cache_dir, "behaviors.pkl"))
            for k,v in behaviors.items():
                setattr(self, k, v)


    def __len__(self):
        if hasattr(self, "imprs"):
            return len(self.imprs)
        else:
            return len(self.token_ids)



class MIND_Train(MIND):
    def __init__(self, manager) -> None:
        data_dir = os.path.join(manager.data_root, "MIND", f"MIND{manager.scale}_train")
        super().__init__(manager, data_dir)

        self.negative_num = manager.negative_num


    def __getitem__(self, index):
        impr_index, positive = self.imprs[index]
        negatives = self.negatives[impr_index]
        histories = self.histories[impr_index]
        user_index = self.user_indices[impr_index]

        negatives, valid_num = sample_news(negatives, self.negative_num)
        cdd_idx = np.asarray([positive] + negatives, dtype=np.int64)
        cdd_mask = np.zeros(len(cdd_idx), dtype=np.int64)
        cdd_mask[:1 + valid_num] = 1

        his_idx = histories[:self.his_size]
        his_mask = np.zeros(self.his_size, dtype=np.int64)
        if len(his_idx) == 0:
            his_mask[0] = 1
        else:
            his_mask[:len(his_idx)] = 1
        # padding user history in case there are fewer historical clicks
        if len(his_idx) < self.his_size:
            his_idx = his_idx + [0] * (self.his_size - len(his_idx))
        his_idx = np.asarray(his_idx, dtype=np.int64)

        # the first entry is the positive instance
        label = 0

        cdd_token_id = self.token_ids[cdd_idx]
        his_token_id = self.token_ids[his_idx]
        cdd_attn_mask = self.attn_masks[cdd_idx]
        his_attn_mask = self.attn_masks[his_idx]

        return_dict = {
            "impr_index": impr_index,
            "user_index": user_index,
            "cdd_idx": cdd_idx,
            "his_idx": his_idx,
            "cdd_mask": cdd_mask,
            "his_mask": his_mask,
            "cdd_token_id": cdd_token_id,
            "his_token_id": his_token_id,
            "cdd_attn_mask": cdd_attn_mask,
            "his_attn_mask": his_attn_mask,
            "label": label
        }
        if self.enable_gate == "weight":
            cdd_gate_mask = self.gate_masks[cdd_idx]
            his_gate_mask = self.gate_masks[his_idx]
            return_dict["cdd_gate_mask"] = cdd_gate_mask
            return_dict["his_gate_mask"] = his_gate_mask

        return return_dict



class MIND_Dev(MIND):
    def __init__(self, manager) -> None:
        data_dir = os.path.join(manager.data_root, "MIND", f"MIND{manager.scale}_dev")
        super().__init__(manager, data_dir)


    def __getitem__(self, index):
        impr_index, impr_news = self.imprs[index]
        histories = self.histories[impr_index]
        user_index = self.user_indices[impr_index]

        # use -1 as padded news' label
        label = np.asarray(self.labels[index] + [-1] * (self.impr_size - len(impr_news)), dtype=np.int64)

        cdd_mask = np.zeros(self.impr_size, dtype=np.bool8)
        cdd_mask[:len(impr_news)] = 1
        cdd_idx = np.asarray(impr_news + [0] * (self.impr_size - len(impr_news)), dtype=np.int64)

        his_idx = histories[:self.his_size]
        his_mask = np.zeros(self.his_size, dtype=np.int64)
        if len(his_idx) == 0:
            his_mask[0] = 1
        else:
            his_mask[:len(his_idx)] = 1
        # padding user history in case there are fewer historical clicks
        if len(his_idx) < self.his_size:
            his_idx = his_idx + [0] * (self.his_size - len(his_idx))
        his_idx = np.asarray(his_idx, dtype=np.int64)

        cdd_token_id = self.token_ids[cdd_idx]
        his_token_id = self.token_ids[his_idx]
        cdd_attn_mask = self.attn_masks[cdd_idx]
        his_attn_mask = self.attn_masks[his_idx]

        return_dict = {
            "impr_index": impr_index,
            "user_index": user_index,
            "cdd_idx": cdd_idx,
            "his_idx": his_idx,
            "cdd_mask": cdd_mask,
            "his_mask": his_mask,
            "cdd_token_id": cdd_token_id,
            "his_token_id": his_token_id,
            "cdd_attn_mask": cdd_attn_mask,
            "his_attn_mask": his_attn_mask,
            "label": label
        }
        if self.enable_gate == "weight":
            cdd_gate_mask = self.gate_masks[cdd_idx]
            his_gate_mask = self.gate_masks[his_idx]
            return_dict["cdd_gate_mask"] = cdd_gate_mask
            return_dict["his_gate_mask"] = his_gate_mask

        return return_dict



class MIND_Test(MIND):
    def __init__(self, manager) -> None:
        data_dir = os.path.join(manager.data_root, "MIND", f"MIND{manager.scale}_test")
        super().__init__(manager, data_dir)


    def __getitem__(self, index):
        impr_index, impr_news = self.imprs[index]
        histories = self.histories[impr_index]
        user_index = self.user_indices[impr_index]

        cdd_mask = np.zeros(self.impr_size, dtype=np.bool8)
        cdd_mask[:len(impr_news)] = 1
        cdd_idx = np.asarray(impr_news + [0] * (self.impr_size - len(impr_news)), dtype=np.int64)

        his_idx = histories[:self.his_size]
        his_mask = np.zeros(self.his_size, dtype=np.int64)
        if len(his_idx) == 0:
            his_mask[0] = 1
        else:
            his_mask[:len(his_idx)] = 1
        # padding user history in case there are fewer historical clicks
        if len(his_idx) < self.his_size:
            his_idx = his_idx + [0] * (self.his_size - len(his_idx))
        his_idx = np.asarray(his_idx, dtype=np.int64)

        cdd_token_id = self.token_ids[cdd_idx]
        his_token_id = self.token_ids[his_idx]
        cdd_attn_mask = self.attn_masks[cdd_idx]
        his_attn_mask = self.attn_masks[his_idx]

        return_dict = {
            "impr_index": impr_index,
            "user_index": user_index,
            "cdd_idx": cdd_idx,
            "his_idx": his_idx,
            "cdd_mask": cdd_mask,
            "his_mask": his_mask,
            "cdd_token_id": cdd_token_id,
            "his_token_id": his_token_id,
            "cdd_attn_mask": cdd_attn_mask,
            "his_attn_mask": his_attn_mask,
        }
        if self.enable_gate == "weight":
            cdd_gate_mask = self.gate_masks[cdd_idx]
            his_gate_mask = self.gate_masks[his_idx]
            return_dict["cdd_gate_mask"] = cdd_gate_mask
            return_dict["his_gate_mask"] = his_gate_mask

        return return_dict


class MIND_News(MIND):
    def __init__(self, manager) -> None:
        data_mode = "test" if manager.mode == "test" else "dev"
        data_dir = os.path.join(manager.data_root, "MIND", f"MIND{manager.scale}_{data_mode}")
        super().__init__(manager, data_dir, load_news=True, load_behaviors=False)

        # cut off padded news
        # self.token_ids = self.token_ids[1:]
        # self.attn_masks = self.attn_masks[1:]
        # if hasattr(self, "gate_masks"):
        #     self.gate_masks = self.gate_masks[1:]


    def __getitem__(self, index):
        cdd_token_id = self.token_ids[index]
        cdd_attn_mask = self.attn_masks[index]

        return_dict =  {
            "cdd_idx": index,
            "cdd_token_id": cdd_token_id,
            "cdd_attn_mask": cdd_attn_mask,
        }
        if self.enable_gate == "weight":
            cdd_gate_mask = self.gate_masks[index]
            return_dict["cdd_gate_mask"] = cdd_gate_mask

        return return_dict




def tokenize_news(news_path, cache_dir, news_num, tokenizer, max_title_length, max_abs_length):
    title_token_ids = [[]] * news_num
    abs_token_ids = [[]] * news_num

    with open(news_path, 'r') as f:
        for idx, line in enumerate(tqdm(f, total=news_num, desc="Tokenizing News", ncols=80)):
            id, category, subcategory, title, abs, _, _, _ = line.strip("\n").split("\t")

            title_token_id = tokenizer.encode(title, max_length=max_title_length)
            title_token_ids[idx] = title_token_id

            abs_token_id = tokenizer.encode(abs, max_length=max_abs_length)
            abs_token_ids[idx] = abs_token_id

    save_pickle(title_token_ids, os.path.join(cache_dir, "title_token_ids.pkl"))
    save_pickle(abs_token_ids, os.path.join(cache_dir, "abs_token_ids.pkl"))


def tokenize_news_keywords(news_path, cache_dir, news_num, tokenizer, max_title_length, max_abs_length):
    title_token_ids = [[]] * news_num
    abs_token_ids = [[]] * news_num

    with open(news_path, 'r') as f:
        for idx, line in enumerate(tqdm(f, total=news_num, desc="Tokenizing News", ncols=80)):
            title, abs = line.strip("\n").split("\t")

            title_token_id = tokenizer.encode(title, max_length=max_title_length)
            title_token_ids[idx] = title_token_id

            abs_token_id = tokenizer.encode(abs, max_length=max_abs_length)
            abs_token_ids[idx] = abs_token_id

    save_pickle(title_token_ids, os.path.join(cache_dir, "title_token_ids.pkl"))
    save_pickle(abs_token_ids, os.path.join(cache_dir, "abs_token_ids.pkl"))


def cache_news(news_path, news_cache_root, manager):
    news_num = int(subprocess.check_output(["wc", "-l", news_path]).decode("utf-8").split()[0])

    # different news file corresponds to different cache directory
    news_cache_dir = os.path.join(news_cache_root, manager.news_cache_dir)
    os.makedirs(news_cache_dir, exist_ok=True)

    # TODO: bm25, entity and keyword
    tokenizer = AutoTokenizer.from_pretrained(manager.plm_dir)
    if manager.news_cache_dir == "original":
        tokenize_news(news_path, news_cache_dir, news_num, tokenizer, manager.max_title_length, manager.max_abs_length)
    else:
        tokenize_news_keywords(news_path, news_cache_dir, news_num, tokenizer, manager.max_title_length, manager.max_abs_length)

    if not os.path.exists(os.path.join(news_cache_root, "nid2index.pkl")):
        print(f"mapping news id to news index and save at {os.path.join(news_cache_root, 'nid2index.pkl')}...")
        construct_nid2index(news_path, news_cache_root)


def cache_behaviors(behaviors_path, cache_dir, nid2index, manager):
    if not os.path.exists(os.path.join(manager.cache_root, 'MIND', 'uid2index.pkl')):
        print(f"mapping user id to user index and save at {os.path.join(manager.cache_root, 'MIND', 'uid2index.pkl')}...")
        uid2index = construct_uid2index(manager.data_root, manager.cache_root)
    else:
        uid2index = load_pickle(os.path.join(manager.cache_root, 'MIND', 'uid2index.pkl'))

    os.makedirs(cache_dir, exist_ok=True)
    imprs = []
    histories = []
    user_indices = []
    impr_index = 0

    if "train" in behaviors_path:
        negatives = []
        with open(behaviors_path, "r") as f:
            for line in tqdm(f, desc="Caching User Behaviors", ncols=80):
                _, uid, _, history, impression = line.strip("\n").split("\t")

                history = [nid2index[x] for x in history.split()]
                interaction_pair = impression.split()
                impr_news = [nid2index[x.split("-")[0]] for x in interaction_pair]
                label = [int(i.split("-")[1]) for i in interaction_pair]
                uindex = uid2index[uid]

                negative = []
                for news, lab in zip(impr_news, label):
                    if lab == 1:
                        imprs.append((impr_index, news))
                    else:
                        negative.append(news)

                histories.append(history)
                negatives.append(negative)
                user_indices.append(uindex)
                impr_index += 1

        save_dict = {
            "imprs": imprs,
            "user_indices": user_indices,
            "histories": histories,
            "negatives": negatives,
        }
        save_pickle(save_dict, os.path.join(cache_dir, "behaviors.pkl"))

    elif "dev" in behaviors_path:
        labels = []
        with open(behaviors_path, "r") as f:
            for line in tqdm(f, desc="Caching User Behaviors", ncols=80):
                _, uid, _, history, impression = line.strip("\n").split("\t")

                history = [nid2index[x] for x in history.split()]
                interaction_pair = impression.split()
                impr_news = [nid2index[x.split("-")[0]] for x in interaction_pair]
                label = [int(i.split("-")[1]) for i in interaction_pair]
                uindex = uid2index[uid]

                for i in range(0, len(impr_news), manager.impr_size):
                    imprs.append((impr_index, impr_news[i: i+manager.impr_size]))
                    labels.append(label[i: i+manager.impr_size])

                # 1 impression correspond to 1 of each of the following properties
                histories.append(history)
                user_indices.append(uindex)
                impr_index += 1

        save_dict = {
            "imprs": imprs,
            "labels": labels,
            "histories": histories,
            "user_indices": user_indices,
        }
        save_pickle(save_dict, os.path.join(cache_dir, "behaviors.pkl"))

    elif "test" in behaviors_path:
        with open(behaviors_path, "r") as f:
            for line in tqdm(f, desc="Caching User Behaviors", ncols=80):
                _, uid, _, history, impression = line.strip("\n").split("\t")

                impr_news = [nid2index[x] for x in impression.split()]
                history = [nid2index[x] for x in history.split()]
                uindex = uid2index[uid]

                for i in range(0, len(impr_news), manager.impr_size):
                    imprs.append((impr_index, impr_news[i: i+manager.impr_size]))

                # 1 impression correspond to 1 of each of the following properties
                histories.append(history)
                user_indices.append(uindex)
                impr_index += 1

        save_dict = {
            "imprs": imprs,
            "histories": histories,
            "user_indices": user_indices,
        }
        save_pickle(save_dict, os.path.join(cache_dir, "behaviors.pkl"))

