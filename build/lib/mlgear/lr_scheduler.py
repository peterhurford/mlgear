import random

from keras import backend as K
from keras.callbacks import Callback


class CustomLRScheduler(Callback):
    def __init__(self, warmup_epochs, max_lr, min_lr, patience=10,
                 factor=0.1, down_factor=0, up_factor=0, up_prob=0, up_every=0, success_backdown=0,
                 monitor='val_loss', mode='min', down_monitor=None, down_mode='min', up_monitor=None, up_mode='min',
                 verbose=0, label='CustomLRScheduler'):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.down_factor = down_factor
        self.up_factor = up_factor
        self.up_prob = up_prob
        self.up_every = up_every
        self.success_backdown = success_backdown
        self.monitor = monitor
        self.mode = mode
        self.down_monitor = down_monitor
        self.down_mode = down_mode
        self.up_monitor = up_monitor
        self.up_mode = up_mode
        self.verbose = verbose
        self.label = label
        self.epoch_count = 1
        self.init_lr = None
        self.best = None
        self.up_best = None
        self.up_count = 0
        self.notices = 0
        if self.down_monitor:
            self.monitor = self.down_monitor
            self.mode = self.down_mode
    
    
    def on_epoch_end(self, epoch, logs=None):
        # Get base LR
        lr = K.get_value(self.model.optimizer.lr)
        
        # Increment epoch count
        self.epoch_count = self.epoch_count + 1
        
        # Warmup
        if self.init_lr is None:
            self.init_lr = lr
        if lr < self.max_lr and self.epoch_count <= self.warmup_epochs:
            new_lr = self.init_lr + (((self.max_lr - self.init_lr) / self.warmup_epochs) * self.epoch_count)
            if self.verbose > 0:
                print('{} (warm up) - LR -> {}'.format(self.label, new_lr))
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 1:
                print('{} debug - warm up {}/{} LR {}/{}'.format(self.label, self.epoch_count, self.warmup_epochs, new_lr, self.max_lr))
            return None
    
        # Increment patience if not improvement
        current_score = logs.get(self.monitor)
        if self.best is None:
            self.best = current_score
        if (self.mode == 'min' and current_score >= self.best) or (self.mode == 'max' and current_score <= self.best):
            self.notices = self.notices + 1
        else:
            self.notices = 0
            self.best = current_score
        if self.verbose > 2:
            print('{} debug - mode {} curr score {} best {}'.format(self.label, self.mode, current_score, self.best))

        
        # Increment up_count by 1 if not improvement, decrement up_count by success_backdown if improvement
        if self.up_monitor:
            up_score = logs.get(self.up_monitor)
            mode = self.up_mode
        else:
            up_score = current_score
            mode = self.mode
        if self.up_best is None:
            self.up_best = up_score
        if (mode == 'min' and up_score < self.up_best) or (mode == 'max' and up_score > self.up_best):
            self.up_count = self.up_count - self.success_backdown
            self.up_best = up_score
            if self.up_count < 0:
                self.up_count = 0
        else:
            self.up_count = self.up_count + 1
        if self.verbose > 2:
            print('{} debug - up mode {} up score {} up best {}'.format(self.label, mode, up_score, self.up_best))
            
        # Bring LR down if patience threshold met
        if self.notices >= self.patience:
            if self.down_factor > 0:
                new_lr = lr * self.down_factor
            else:
                new_lr = lr * self.factor
            if new_lr < self.min_lr:
                new_lr = self.min_lr
            if self.verbose > 0:
                print('{} (down) - LR -> {}'.format(self.label, new_lr))
            K.set_value(self.model.optimizer.lr, new_lr)
            self.notices = 0 # Reset patience threshold
            if self.verbose > 1:
                print('{} debug - up_count {}/{} notices {}/{}'.format(self.label, self.up_count, self.up_every, self.notices, self.patience))
            return None
        
        # Bring LR up if up threshold met
        should_up = self.up_prob > 0 and random.random() <= self.up_prob
        should_up = should_up or (self.up_every > 0 and self.up_count >= self.up_every)
        if should_up:
            if self.up_factor > 0:
                new_lr = lr * self.up_factor
            else:
                new_lr = lr * (1 / self.factor)
            if new_lr > self.max_lr:
                new_lr = self.max_lr
            if self.verbose > 0:
                print('{} (up) - LR -> {}'.format(self.label, new_lr))
            K.set_value(self.model.optimizer.lr, new_lr)
            self.up_count = 0 # Reset up_count
            self.notices = 0  # Also reset notices
        if self.verbose > 1:
            print('{} debug - up_count {}/{} notices {}/{}'.format(self.label, self.up_count, self.up_every, self.notices, self.patience))
        return None
