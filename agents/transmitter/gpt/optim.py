# from transformers import OpenAIAdam
# from sched import scheduler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
# from agents.transmitter.utils import _ellipse
from parlai.core.utils import _ellipse


class GPTOptimizer:
    def __init__(self, model, opt):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_with_decay = []
        parameters_with_decay_names = []
        parameters_without_decay = []
        parameters_without_decay_names = []
        base_parameters = []
        base_parameters_names = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                # fine-tune BERT
                if any(t in n for t in ["transformer"]):
                    if any(t in n for t in no_decay):
                        parameters_without_decay.append(p)
                        parameters_without_decay_names.append(n)
                    else:
                        parameters_with_decay.append(p)
                        parameters_with_decay_names.append(n)
                else:
                    base_parameters.append(p)
                    base_parameters_names.append(n)

        weight_decay = opt['weight_decay']
        bert_learning_rate = opt['gpt_lr']
        base_learning_rate = opt['lr']
        optimizer_grouped_parameters = [
            {'params': parameters_with_decay, 'weight_decay': weight_decay, 'lr': bert_learning_rate},
            {'params': parameters_without_decay, 'weight_decay': 0.0, 'lr': bert_learning_rate},
            {'params': base_parameters, 'weight_decay': weight_decay, 'lr': base_learning_rate}
        ]
        #
        print('The following parameters will be optimized WITH decay:')
        print(_ellipse(parameters_with_decay_names, 5, ' , '))
        print('The following parameters will be optimized WITHOUT decay:')
        print(_ellipse(parameters_without_decay_names, 5, ' , '))
        print('The following parameters will be optimized NORMALLY:')
        print(_ellipse(base_parameters_names, 5, ' , '))

        # optimizer = OpenAIAdam(optimizer_grouped_parameters,
        #                        lr=opt['gpt_lr'],
        #                        warmup=opt['warmup_proportion'],
        #                        max_grad_norm=opt['gradient_clip'],
        #                        t_total=opt.get('optimizer_step', -1))
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=opt['gpt_lr'])
        t_total = opt.get('optimizer_step', -1)
        warmup = int(opt['warmup_proportion'] * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, t_total)
        self.max_grad_norm = opt['gradient_clip']
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    @property
    def params(self):
        for group in self.param_groups:
            for p in group['params']:
                yield p

    def step(self):
        # print(self.param_groups[0].keys())
        # print(len(self.param_groups))
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
