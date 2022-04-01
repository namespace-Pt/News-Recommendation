import torch
import time
import logging
import argparse
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s (%(name)s) %(message)s")
logger = logging.getLogger(__file__)


parser = argparse.ArgumentParser()
parser.add_argument("-d","--device", dest="device",
                    help="device to run on, -1 means cpu", choices=[i for i in range(-1,10)], type=int, default=0)
args = parser.parse_args()


logger.info("I'm running on cuda:{} to stop the platform killing this job!".format(args.device))
a = torch.zeros((1),device=args.device)
while(1):
    if a.item() > 2:
        a -= 1
    else:
        a += 1