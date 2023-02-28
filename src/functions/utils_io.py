import config

import os, pickle
import torch
import numpy as np
from glob import glob
from datetime import datetime

class ModelSaver(object):

    def __init__(self, modelname, args, with_date=True):
        """
        Model saving functions
        """

        save_dir = self._prepare_savedir(modelname, args, with_date)
        self.save_dir = save_dir

        tflog_dir = self._prepare_tflog_dir()
        self.tflog_dir = tflog_dir

        os.makedirs(save_dir, exist_ok=True)

    def _prepare_savedir(self, modelname, args, with_date):
        """
        prepare save_dir
        """
        seed_num = "seed{}".format(args.seed)
        trainparams = prepare_trainparam_string(args)

        save_dir = os.path.join(config.model_dir, seed_num,
                                modelname, trainparams)
        if with_date:
            timestamp = self._get_timestamp()
            save_dir = os.path.join(save_dir, timestamp)

        return save_dir

    def _prepare_tflog_dir(self):
        """
        Process save_dir to prepare tflog_dir
        """
        modelinfo = self.save_dir.split("/")
        tflog_dir = os.path.join(*[config.tflog_dir] + modelinfo[3:])
        return tflog_dir

    def _get_timestamp(self):
        timestamp = datetime.now()
        timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
        return timestamp

    def save_condition(self, modelname, args):
        save_txt = f"{modelname}\n"
        for key, value in args.__dict__.items():
            save_txt += f"\t{key}: {value}\n"

        with open(self.save_dir+'/model_info.txt', 'w') as f:
            f.write(save_txt)
        with open(self.save_dir+'/model_info.pkl', 'wb') as fp:
            pickle.dump(args, fp)

        print("Saved model setting")

    def save_architecture(self, model):
        print("Saving model architecture ...")
        model = model.to("cpu")
        torch.save(model, self.save_dir + "/architecture")

    def save_model(self, epoch, valloss, model):
        epoch = config.ep_str.format(epoch)
        val_loss = '{:.4f}'.format(valloss)
        modelfile = config.modelfile.format(epoch, val_loss)

        filename  = os.path.join(self.save_dir, modelfile)
        torch.save(model.state_dict(), filename)
        print("Saved model as {}".format(filename))

    def save_bestmodel(self, loss_log, use_min=True):
        if use_min:
            best_idx = np.argmin(list(loss_log.values()))
        else:
            best_idx = np.argmax(list(loss_log.values()))

        best_epoch = list(loss_log.keys())[best_idx]
        best_epoch = config.ep_str.format(best_epoch)
        model_file = config.modelfile.format(best_epoch, "*")
        model_file = os.path.join(self.save_dir, model_file)
        model_file = glob(model_file)[0]
        best_file  = model_file.replace(best_epoch, "best")
        os.system("cp {} {}".format(model_file, best_file))
        print("Saved best model ({})".format(model_file))

class ModelLoader(object):

    def __init__(self, model_loc, device="cpu"):
        self.model_loc = model_loc
        self.device = device

    def _load_args(self):
        with open(self.model_loc + "/model_info.pkl", "rb") as fp:
            args = pickle.load(fp)
        return args

    def load_model(self, epoch=None):
        print(f"Loading model from {self.model_loc} ...")
        model = torch.load(self.model_loc+"/architecture")
        if epoch is None:
            state = glob(self.model_loc+"/best-*.pth")[0]
        else:
            state = glob(self.model_loc+"/ep{:04d}-*.pth".format(epoch))[0]
        print("Model: {}".format(state))
        model.load_state_dict(torch.load(state))
        model = model.to(self.device)
        return model

def prepare_trainparam_string(args):

    if args.clip_value == 0:
        clip_val = "None"
    else:
        clip_val = args.clip_value

    param = f"LR-{args.lr}_OPT-{args.optim}_BS-{args.bs}_CLIP-{clip_val}_"
    param += f"KERNEL-{args.kernel}_NFIL-{args.nfilter}_"
    param += f"EP-{args.ep}_FREQ-{args.freq}_CW-{args.cw}_"
    param += f"SrcData-{args.dataset}_AUG-{args.augment}"

    if args.lead == "all":
        param += "_LEAD-all"
    return param
