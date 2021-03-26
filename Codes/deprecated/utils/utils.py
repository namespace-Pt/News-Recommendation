# @torch.no_grad()
# def _eval_mtp(i, model, hparams, dataloader, result_list):
#     """evaluate in multi-processing

#     Args:
#         i(int) Subprocess No.
#         model(nn.Module)
#         hparams(dict)
#         dataloader(torch.utils.data.DataLoader)
#         result_list(torch.multiprocessing.Manager().list()): contain evaluation results of every subprocesses
#     """
#     step = hparams['save_step'][i]
#     save_path = 'data/model_params/{}/{}_epoch{}_step{}_[hs={},topk={}].model'.format(
#         hparams['name'], hparams['scale'], hparams['epochs'], step, str(hparams['his_size']), str(hparams['k']))

#     logging.info(
#         "[No.{}, PID:{}] loading model parameters from {}...".format(i, os.getpid(), save_path))

#     model.load_state_dict(torch.load(
#         save_path, map_location=hparams['device']))

#     logging.info("[No.{}, PID:{}] evaluating...".format(i, os.getpid()))

#     imp_indexes, labels, preds = run_eval(model, dataloader, 10)
#     res = cal_metric(labels, preds, hparams['metrics'].split(','))

#     res['step'] = step
#     logging.info("\nevaluation results of process NO.{} is {}".format(i, res))

#     result_list.append(res)

def evaluate
# elif steps > 1:
    #     logging.info("evaluating in {} processes...".format(steps))
    #     model.share_memory()
    #     res_list = mp.Manager().list()
    #     mp.spawn(_eval_mtp, args=(model, hparams,
    #                               dataloader, res_list), nprocs=steps)
    #     with open('performance.log', 'a+') as f:
    #         d = {}
    #         for k, v in hparams.items():
    #             if k in hparam_list:
    #                 d[k] = v
    #         for name, param in model.named_parameters():
    #             if name in param_list:
    #                 d[name] = tuple(param.shape)
    #         f.write(str(d)+'\n')

    #         for result in res_list:
    #             f.write(str(result) + '\n')
    #         f.write('\n')

def tune
# without evaluating, only training

        # logging.info("evaluating in {} processes...".format(len(hparams['save_step'])))
        # with torch.no_grad():
        #     model.share_memory()
        #     res_list = mp.Manager().list()
            # mp.spawn(_eval_mtp, args=(model, hparams, loader_dev, res_list), nprocs=len(hparams['step_list']))

        #     with open('sfi-performance.log','a+') as f:
        #         for result in res_list:
        #             if result['auc'] > best_auc:
        #                 best_auc = result['auc']

        #                 d = {}
        #                 for k,v in hparams.items():
        #                     if k in hparam_list:
        #                         d[k] = v

        #                 for name, param in model.named_parameters():
        #                     if name in param_list:
        #                         d[name] = tuple(param.shape)

        #                 f.write(str(d)+'\n')
        #                 f.write(str(result) +'\n')
        #                 f.write('\n')