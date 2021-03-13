import multiprocessing as mp
import os
def f(x):
    os.system("python scripts/sfi_fim.py -m dev -s large -e 1 --select=gating --topk=30 --his_size=50 --cuda=1 --save_step={}".format(x))
    return 0

if __name__ == '__main__':
    with mp.Pool(5) as p:
        print(p.map(f, [20000, 22000, 30000]))