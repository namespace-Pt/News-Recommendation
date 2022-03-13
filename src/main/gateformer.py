import torch.multiprocessing as mp
from utils.manager import Manager
from models.GateFormer import *
from torch.nn.parallel import DistributedDataParallel as DDP
from models.modules.encoder import *
from models.modules.weighter import *


def main(rank, manager):
    """ train/dev/test the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()

    if manager.newsEncoder == "cnn":
        newsEncoder = CnnNewsEncoder(manager)
    elif manager.newsEncoder == "bert":
        newsEncoder = GatedBertNewsEncoder(manager)
    elif manager.newsEncoder == "tfm":
        newsEncoder = TfmNewsEncoder(manager)

    if manager.userEncoder == "rnn":
        userEncoder = RnnUserEncoder(manager)
    elif manager.userEncoder == "sum":
        userEncoder = SumUserEncoder(manager)
    elif manager.userEncoder == "avg":
        userEncoder = AvgUserEncoder(manager)
    elif manager.userEncoder == "attn":
        userEncoder = AttnUserEncoder(manager)
    elif manager.userEncoder == "tfm":
        userEncoder = TfmUserEncoder(manager)

    if manager.weighter == "cnn":
        weighter = CnnWeighter(manager)
    elif manager.weighter == "tfm":
        weighter = TfmWeighter(manager)
    elif manager.weighter == "bert":
        weighter = AllBertWeighter(manager)
    elif manager.weighter == "first":
        weighter = FirstWeighter(manager)

    # model = TwoTowerGateFormer(manager, newsEncoder, userEncoder, weighter).to(manager.device)
    model = UserOneTowerGateFormer(manager, newsEncoder, weighter).to(manager.device)

    if manager.mode == 'train':
        if manager.world_size > 1:
            model = DDP(model, device_ids=[manager.device], output_device=manager.device)
        manager.train(model, loaders)

    elif manager.mode == 'dev':
        model.dev(manager, loaders, load=True, log=True)

    elif manager.mode == 'test':
        model.test(manager, loaders, load=True, log=True)

    elif manager.mode == "inspect":
        manager.load(model)
        model.inspect(manager, loaders)


if __name__ == "__main__":
    config = {
        "enable_gate": "weight",
        "enable_fields": ["title"],
        "newsEncoder": "bert",
        "userEncoder": "rnn",
        "weighter": "cnn",
        "validate_step": "0.5e",
        "hold_step": "2e",
    }
    manager = Manager(config)

    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)