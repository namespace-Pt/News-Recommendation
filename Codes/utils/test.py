import multiprocessing as mp
import os,sys
import subprocess
def f(hparams):
    for step in hparams['save_steps']:
        command = "python scripts/{}.py -m {} -s {} -e {} --select={}"
    subprocess.Popen("python scripts/sfi_fim.py -m dev -s large -e 1 --select=gating --topk=30 --his_size=50 --cuda=1 --save_step={}".format(x))
    return 0

if __name__ == '__main__':
    print(sys.argv)