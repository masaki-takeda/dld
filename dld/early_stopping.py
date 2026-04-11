import numpy as np
import torch
import os


class EarlyStopping:
    def __init__(self,
                 patience,
                 ignore_epochs,
                 save_dir,
                 fold,
                 classify_type,
                 metric,
                 debug=False):

        assert (metric == 'accuracy' or \
                metric == 'roc_auc' or \
                metric == 'pr_auc' or \
                metric == 'f1' or \
                metric == 'precision' or \
                metric == 'recall' or \
                metric == 'n_precision' or \
                metric == 'n_recall' or \
                metric == 'n_f1' or \
                metric == 'n_pr_auc')
        
        self.patience = patience
        self.ignore_epochs = ignore_epochs # Number of epochs to ignore the decision at the beggining
        self.save_dir = save_dir
        self.fold = fold
        
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_validation_metrics = None
        self.accompanied_train_metrics = None
        self.classify_type = classify_type
        self.metric = metric
        
        self.debug = debug

    def check_stopping(self,
                       validation_metrics,
                       train_metrics,
                       epoch,
                       model):
        if epoch < self.ignore_epochs and not self.debug:
            # Ignore decision at the beggining
            return False
        
        score = validation_metrics[self.metric]
        
        if self.debug:
            # When debugging, it terminates on the first call
            self.best_score = score
            self.best_validation_metrics = validation_metrics
            self.accompanied_train_metrics = train_metrics
            self.save(model)
            self.early_stop = True
            return self.early_stop
            
        if self.best_score is None:
            # First time
            self.best_score = score
            self.best_validation_metrics = validation_metrics
            self.accompanied_train_metrics = train_metrics
            self.save(model)
        elif score < self.best_score:
            # If the score is below the best score
            self.counter += 1
            print('EarlyStopping counter: {}/{}'.format(
                self.counter, self.patience))
            if self.counter >= self.patience:
                # The scores have been below the best score more than the specified number of times
                self.early_stop = True
        else:
            # Best score is updated
            self.best_score = score
            self.save(model)
            self.best_validation_metrics = validation_metrics
            self.accompanied_train_metrics = train_metrics
            self.counter = 0

        return self.early_stop

    def save(self, model):
        model_path = os.path.join(self.save_dir,
                                  "model_ct{}_{}.pt".format(self.classify_type,
                                                            self.fold))
        torch.save(model.state_dict(), model_path)
