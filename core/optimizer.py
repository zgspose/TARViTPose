import torch
import torch.optim as optim

# class Optimizer:
#     def __init__(self, model, cfg):
#         self.model = model
#         self.cfg = cfg

#     def get_optimizer(self):
#         optimizer_name = self.cfg.TRAIN.OPTIMIZER
#         lr = self.cfg.TRAIN.LR
        
#         # Add other optimizer parameters here if needed
#         weight_decay = self.cfg.TRAIN.WEIGHT_DECAY
#         betas = self.cfg.TRAIN.BETAS

#         if optimizer_name == 'adam':
#             return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
#         elif optimizer_name == 'adamw':
#             return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
#         else:
#             raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

class Optimizer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def get_optimizer(self):
        optimizer_name = self.cfg.TRAIN.OPTIMIZER
        lr = self.cfg.TRAIN.LR
        backbone_lr = self.cfg.TRAIN.BACKBONE_LR  # Learning rate for backbone
        weight_decay = self.cfg.TRAIN.WEIGHT_DECAY
        betas = self.cfg.TRAIN.BETAS

        # Separate parameters for the backbone and the other layers
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'backbone' in name:  # Assuming 'backbone' is part of the name of the backbone parameters
                backbone_params.append(param)
            else:
                other_params.append(param)

        # Define the optimizer with differential learning rates
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': other_params, 'lr': lr}
        ]

        # if the lr are diffent print the learning rates in yellow
        if lr != backbone_lr:
            print("\033[93m" + "Learning rate for backbone: ", backbone_lr, "\033[0m")
            print("\033[93m" + "Learning rate for other layers: ", lr, "\033[0m")
        else:
            print("Learning rate: ", lr)

        if optimizer_name == 'adam':
            return optim.Adam(param_groups, weight_decay=weight_decay, betas=betas)
        elif optimizer_name == 'adamw':
            return optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}")   


    
