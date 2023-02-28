import torch
from importlib import import_module
from functions.train_base import BaseTrainer

torch.backends.cudnn.deterministic=True

class Executer(object):

    def __init__(self, args):
        torch.manual_seed(args.seed)

        self.args = args
        modelname = args.model
        model = self._load_model(modelname)

        self.trainer = self._load_trainer(model)

    def _load_trainer(self, model):
        Trainer = BaseTrainer(model, self.args)
        return Trainer

    def _load_model(self, modelname):
        modelfile = "architectures.{}".format(modelname)
        ModelClass = import_module(modelfile)
        Model = ModelClass.__dict__[modelname]
        model = Model(self.args)
        print("Loaded {} ...".format(ModelClass.__name__))
        return model

    def execute(self):
        self.trainer.run()

if __name__ == "__main__":
    from hyperparams import args

    executer = Executer(args)
    executer.execute()
