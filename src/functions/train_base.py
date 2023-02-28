import sys
sys.path.append("..")
import config

import torch
from torch.nn.utils import clip_grad_value_

import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter

from functions.dataset import Dataset
from functions.dataloader import DataLoader
from functions.utils_io import ModelSaver
import functions.utils_train as train_utils

EVAL_CRITERION = "score"
USEMIN = False

class BaseTrainer(object):

    def __init__(self, model, args):

        self.args = args
        self.model = model
        modelname = str(self.model.__class__.__name__)

        self._initial_process(model, modelname, args)
        self.model.to(self.args.device)

    def _initial_process(self, model, modelname, args):
        # Prepare model saver
        self.saver = ModelSaver(modelname, args)

        # Save hyperparameter and model architecture
        self.saver.save_condition(modelname, args)
        self.saver.save_architecture(model)

    def _prepare_dataset(self, datatype):

        if self.args.dataset == "cpsc":
            dirname = config.cpsc_dirname
        else:
            dirname = config.pnet_dirname
        data_loc = os.path.join(config.root, dirname, config.processed_dir,
                                f"fs{self.args.freq:04d}")

        dataset = Dataset(data_loc, datatype, self.args.freq, self.args.seed,
                          max_duration=self.args.max_duration, dataset=self.args.dataset,
                          num_class=self.args.o_dim, lead_type=self.args.lead)
        return dataset

    def _prepare_dataloader(self, datatype, is_eval=False):
        print("Preparing {} dataloader".format(datatype))
        dataset = self._prepare_dataset(datatype)

        if is_eval:
            augmentation = None
        else:
            augmentation = self.args.augment

        loader = DataLoader(dataset, batchsize=self.args.bs,
                            is_eval=is_eval, device=self.args.device,
                            augment=augmentation, seed=self.args.seed)
        return loader

    def _prepare_train_settings(self, label_counts, num_samples):
        if self.args.label == "3class":
            loss_func = train_utils.prepare_lossfunc_3class(
                self.args, label_counts, num_samples)
        elif self.args.label == "9class":
            loss_func = train_utils.prepare_lossfunc_9class(
                self.args, label_counts, num_samples)

        optimizer = train_utils.prepare_optim(
            self.args.optim, self.model, self.args.lr)

        self.optimizer = optimizer
        self.loss_func = loss_func

    def _calculate_loss(self, y_probs, y_trues):
        """
        Calculate loss between prediction and labels
        """
        loss = self.loss_func(y_probs, y_trues)
        loss = loss.sum()
        return loss

    def _train(self, epoch, iterator):
        train_loss, num_data = 0, 0
        self.model.train()

        for X, y in tqdm(iterator):
            self.optimizer.zero_grad()
            num_data += len(X)

            y_pred = self.model(X)
            loss = self._calculate_loss(y_pred, y)
            loss.backward()

            if self.args.clip_value:
                clip_grad_value_(self.model.parameters(), self.args.clip_value)
            self.optimizer.step()

            train_loss += float(loss)
        train_loss /= num_data
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return {"train_loss": train_loss}

    def _evaluate(self, epoch, iterator):

        eval_loss, num_data = 0, 0
        y_probs, y_trues = [], []
        self.model.eval()

        with torch.no_grad():
            for X, y in tqdm(iterator):
                num_data += len(X)

                y_prob = self.model(X)

                loss = self._calculate_loss(y_prob, y)
                eval_loss += float(loss)
                y_prob = torch.softmax(y_prob, dim=-1)
                y_probs.append(y_prob.cpu().detach().numpy())
                y_trues.append(y.cpu().detach().numpy())

        # Calculate average evaluation loss
        eval_loss /= num_data
        result = self._process_eval_result(y_probs, y_trues, eval_loss, epoch)
        return result

    def _process_eval_result(self, y_probs, y_trues,
                             eval_loss=0, epoch=0):

        # Calculate f1 scores
        y_probs, y_trues = np.concatenate(y_probs), np.concatenate(y_trues)
        if self.args.label == "3class":
            average_f1, class_f1 = train_utils.calculate_score_3class(y_probs, y_trues)
        elif self.args.label == "9class":
            average_f1, class_f1 = train_utils.calculate_score_9class(y_probs, y_trues)

        # Prepare print information
        print_info = (epoch, eval_loss, average_f1) + class_f1
        print_text = "Epoch: {} Valid loss: {:.4f}, F1score: {:.4f}\n"
        print_text += "(N: {:.4f}, A: {:.4f}, O {:.4f})"
        print(print_text.format(*print_info))

        # Prepare result dictinary
        result = {"eval_loss": eval_loss, "score": average_f1,
                  "f1-N": class_f1[0], "f1-A": class_f1[1],
                  "f1-O": class_f1[2]}
        return result

    def _validation_process(self, epoch, train_loader, valid_loader):
        # Train data evaluation
        print("Train data evaluation")
        train_score = self._evaluate(epoch, train_loader)[EVAL_CRITERION]
        self.writer.add_scalar(f"{EVAL_CRITERION}-Train", train_score, epoch)

        # Validation data evaluation
        print("Valid data evaluation")
        valid_loss = self._evaluate(epoch, valid_loader)
        for key, value in valid_loss.items():
            self.writer.add_scalar(key, value, epoch)

        # Calculate gap between train and valid
        valid_score = valid_loss[EVAL_CRITERION]
        score_gap = train_score - valid_score
        self.writer.add_scalar("score_gap", score_gap, epoch)

        # Save model and store epoch loss to logger
        self.loss_log[epoch] = valid_score
        self.saver.save_model(epoch, valid_score, self.model)

    def _check_label_counts(self, iterator):
        labels, counts = np.unique(iterator.dataset.label, return_counts=True)
        print("-"*80)
        print("Label distribution")

        if self.args.label == "3class":
            labelchars = config.pnet_labels[:3]
        else:
            labelchars = config.cpsc_dxs

        label_count_dict = defaultdict()
        for label, count in zip(labels, counts):
            label = int(label)
            ratio = count / counts.sum()
            print(f"\tLabel {labelchars[label]}: {count} ({ratio:.04f})")
            label_count_dict[labelchars[label]] = count
        print("-"*80)
        return label_count_dict

    def _remove_non_best(self):
        print("Removing non best model file ...")
        command = f"rm {self.saver.save_dir}/ep*.pth"
        os.system(command)
        print("Showing remaining files ...")
        command = f"ls {self.saver.save_dir}/"
        os.system(command)

    def run(self):
        train_loader = self._prepare_dataloader("train")
        valid_loader = self._prepare_dataloader("valid", is_eval=True)
        train_loader_eval = self._prepare_dataloader("train", is_eval=True)
        label_counts = self._check_label_counts(train_loader)
        num_samples = len(train_loader.dataset.data)
        self._prepare_train_settings(label_counts, num_samples)

        self.loss_log = defaultdict(float)
        self.writer = SummaryWriter(self.saver.tflog_dir)
        for epoch in range(1, self.args.ep+1):
            train_loss = self._train(epoch, train_loader)
            for key, value in train_loss.items():
                self.writer.add_scalar(key, value, epoch)

            if epoch % self.args.save_every == 0:
                self._validation_process(epoch, train_loader_eval,
                                         valid_loader)

        self.writer.close()
        self.saver.save_bestmodel(self.loss_log, use_min=USEMIN)
        self._remove_non_best()
        print("-"*80)
