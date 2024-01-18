import os
import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-5, project_name=None, checkpoint_file_path=None, runtime=None):
        self.patience = patience
        self.delta = delta
        self.project_name = project_name
        self.runtiem = runtime
        self.file_path = os.path.join(checkpoint_file_path, f"{project_name}_cp_{runtime}.pt")
        self.lastest_file_path = os.path.join(checkpoint_file_path, f"{project_name}_cp_lastest.pt")

        self.valid_loss_min = None
        self.counter = 0

    def check_and_save(self, new_valid_loss, model):
        """
        Check condition of loop.
        If condition statisfied to be stopped, stop training.
        If not, continue training.
        """
        early_stop = False

        if self.valid_loss_min is None:
            self.valid_loss_min = new_valid_loss
            message = f"Early stopping start!"
        elif new_valid_loss < self.valid_loss_min - self.delta:
            message = f"V_loss decreased ({self.valid_loss_min:7.5f} --> {new_valid_loss:7.5f}). Saving model to lastest cp."
            self.save_checkpoint(new_valid_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            message = f"Early stopping counter : {self.counter} out of {self.patience}"
            if self.counter >= self.patience:
                early_stop = True
                message += " ### Early stopped! ###"

        return message, early_stop
    
    def save_checkpoint(self, valid_loss, model):
        """
        Save model to current path and lastest model.
        Update minimum loss.
        """
        torch.save(model.state_dict(), self.file_path)
        torch.save(model.state_dict(), self.lastest_file_path)
        self.valid_loss_min = valid_loss