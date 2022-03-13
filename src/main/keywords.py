import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-dr", "--data-root", dest="data_root", default="../../../Data")
parser.add_argument("-eg", "--enable-gate", dest="enable_gate", help="way to gate tokens", type=str, choices=["weight", "none", "bm25", "first", "keybert", "random"], default="weight")
args = parser.parse_args()

news_dir_all = ["MINDlarge_train", "MINDlarge_dev", "MINDlarge_test"]

if args.enable_gate == "bm25":
    from utils.util import BM25
    for news_dir in news_dir_all:
        news_path = os.path.join(args.data_root, "MIND", news_dir, "news.tsv")
        if not os.path.exists(news_path):
            continue

        bm25_title = BM25()
        bm25_abs = BM25()

        titles = []
        abstracts = []
        with open(news_path) as f:
            for line in f:
                id, category, subcategory, title, abs, _, _, _ = line.strip("\n").split("\t")
                titles.append(title)
                abstracts.append(abs)

        titles = bm25_title(titles)
        abstracts = bm25_abs(abstracts)

        new_news_path = os.path.join(os.path.split(news_path)[0], "bm25.tsv")
        with open(new_news_path, "w") as f:
            for title, abs in zip(titles, abstracts):
                f.write(title + "\t" + abs + "\n")


elif args.enable_gate == "keybert":
    from keybert import KeyBERT
    kw_model = KeyBERT()

    for news_dir in news_dir_all:
        news_path = os.path.join(args.data_root, "MIND", news_dir, "news.tsv")
        if not os.path.exists(news_path):
            continue

        titles = []
        abstracts = []
        with open(news_path) as f:
            new_news_path = os.path.join(os.path.split(news_path)[0], "keybert.tsv")
            g = open(new_news_path, "w")
            for line in tqdm(f, desc=news_path):
                id, category, subcategory, title, abs, _, _, _ = line.strip("\n").split("\t")

                title = kw_model.extract_keywords(title, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=100)
                title = " ".join([kwd[0] for kwd in title])

                abs = kw_model.extract_keywords(abs, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=100)
                abs = " ".join([kwd[0] for kwd in abs])

                g.write(title + "\t" + abs + "\n")
            g.close()

