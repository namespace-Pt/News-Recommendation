import torch
import torch.multiprocessing as mp
from utils.manager import Manager
from models.OneTowerBert import OneTowerBert
from torch.nn.parallel import DistributedDataParallel as DDP


def main(rank, manager):
    """ train/dev/test the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()

    model = OneTowerBert(manager).to(manager.device)

    if manager.mode == 'train':
        if manager.world_size > 1:
            model = DDP(model, device_ids=[manager.device], output_device=manager.device)
        manager.train(model, loaders)

    elif manager.mode == 'dev':
        manager.load(model)
        model.dev(manager, loaders, log=True)

    elif manager.mode == 'test':
        manager.load(model)
        model.test(manager, loaders)


if __name__ == "__main__":
    config = {
        "batch_size_eval": 100,
        "enable_fields": ["title"],
        "validate_step": "0.5e",
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