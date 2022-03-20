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
                 debug=False):
        self.patience = patience
        self.ignore_epochs = ignore_epochs # Number of epochs to ignore the decision at first
        self.save_dir = save_dir
        self.fold = fold
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.max_validation_accuracy = 0.0
        self.accompanied_train_accuracy = 0.0
        self.classify_type = classify_type
        
        self.debug = debug

    def check_stopping(self,
                       validation_accuracy,
                       train_accuracy,
                       epoch,
                       model):
        if epoch < self.ignore_epochs and not self.debug:
            # During the epoch period of first ignoring
            return False
        
        score = validation_accuracy

        if self.debug:
            # When debugging, it terminates on the first call
            self.best_score = score
            self.max_validation_accuracy = validation_accuracy
            self.accompanied_train_accuracy = train_accuracy
            self.save(model)
            self.early_stop = True
            return self.early_stop
            
        if self.best_score is None:
            # For first call
            self.best_score = score
            self.max_validation_accuracy = validation_accuracy
            self.accompanied_train_accuracy = train_accuracy
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
            self.best_score = score
            self.save(model)
            self.max_validation_accuracy = validation_accuracy
            self.accompanied_train_accuracy = train_accuracy
            self.counter = 0

        return self.early_stop

    def save(self, model):
        model_path = os.path.join(self.save_dir,
                                  "model_ct{}_{}.pt".format(self.classify_type,
                                                            self.fold))
        torch.save(model.state_dict(), model_path)
