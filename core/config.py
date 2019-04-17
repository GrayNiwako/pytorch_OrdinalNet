# -*- coding: utf-8 -*-
import os
from warnings import warn
from time import strftime as timestr

class Config(object):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_save_path = r'../my_datasets'
    classes_list = ['A', 'B', 'C', 'D', 'E', 'nodata']
    shuffle_train = True
    shuffle_val = True
    drop_last_train = True
    drop_last_val = False

    # efficiency config
    use_gpu = True  # if there's no cuda-available GPUs, this will turn to False automatically
    num_data_workers = 1  # how many subprocesses to use for data loading
    pin_memory = True  # only set to True when your machine's memory is large enough
    time_out = 0  # max seconds for loading a batch of data, 0 means non-limit
    max_epoch = 200  # how many epochs for training
    batch_size = 16  # how many scene images for a batch

    # weight S/L config
    weight_load_path = r'checkpoints/doublenet.pth'  # where to load pre-trained weight for further training
    weight_save_path = r'checkpoints/doublenet.pth'  # where to save trained weights for further usage
    log_root = r'logs'  # where to save logs, includes temporary weights of module and optimizer, train_record json list

    # module config
    module = 'alexnet'  # module in ['alexnet', 'resnet', 'lenet']
    image_resize = [224, 224]  # Height * Width
    loss_type = 'cross_entropy'  # loss_type in ['cross_entropy', 'mseloss']
    optimizer = 'sgd'  # optimizer in ['sgd', 'adam']
    lr = 0.01  # learning rate
    lr_decay = 0.95
    momentum = 0.9
    weight_decay = 1e-4  # weight decay (L2 penalty)

    # visualize config
    visdom_env = 'main'
    ckpt_freq = 5  # save checkpoint after these iterations

    def __init__(self, mode: str, **kwargs):
        self.mode = mode
        self.init_time = timestr('%Y%m%d.%H%M%S')
        # data config
        self.num_classes = len(self.classes_list)
        # efficiency config
        if self.use_gpu:
            from torch.cuda import is_available as cuda_available, device_count
            if cuda_available():
                self.num_gpu = device_count()
                self.gpu_list = list(range(self.num_gpu))
                assert self.batch_size % self.num_gpu == 0, \
                    "Can't split a batch of data with batch_size {} averagely into {} gpu(s)" \
                        .format(self.batch_size, self.num_gpu)
            else:
                warn("Can't find available cuda devices, use_gpu will be automatically set to False.")
                self.use_gpu = False
                self.num_gpu = 0
                self.gpu_list = []
        else:
            from torch.cuda import is_available as cuda_available
            if cuda_available():
                warn("Available cuda devices were found, please switch use_gpu to True for acceleration.")
            self.num_gpu = 0
            self.gpu_list = []
        if self.use_gpu:
            self.map_location = lambda storage, loc: storage
        else:
            self.map_location = "cpu"
        # weight S/L config
        self.vis_env_path = os.path.join(self.log_root, 'visdom')
        os.makedirs(os.path.dirname(self.weight_save_path), exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(self.vis_env_path, exist_ok=True)
        assert os.path.isdir(self.log_root)

        self.temp_weight_path = os.path.join(self.log_root, 'tmpmodel{}.pth'.format(self.init_time))
        self.temp_optim_path = os.path.join(self.log_root, 'tmp{}{}.pth'.format(self.optimizer, self.init_time))
        self.log_file = os.path.join(self.log_root, '{}.{}.log'.format(self.mode, self.init_time))
        self.val_result = os.path.join(self.log_root, 'validation_result{}.txt'.format(self.init_time))
        self.train_record_file = os.path.join(self.log_root, 'train.record.jsons')
        """
       record training process by core.make_checkpoint() with corresponding arguments of
       [epoch, start time, elapsed time, loss value, train accuracy, validate accuracy]
       DO NOT CHANGE IT unless you know what you're doing!!!
       """
        self.__record_fields__ = ['epoch', 'start', 'elapsed', 'loss', 'train_score', 'val_score']
        if len(self.__record_fields__) == 0:
            warn(
                '{}.__record_fields__ is empty, this may cause unknown issues when save checkpoint into {}' \
                    .format(type(self), self.train_record_file))
            self.__record_dict__ = '{{}}'
        else:
            self.__record_dict__ = '{{'
            for field in self.__record_fields__:
                self.__record_dict__ += '"{}":"{{}}",'.format(field)
            self.__record_dict__ = self.__record_dict__[:-1] + '}}'

    def __str__(self):
        """:return Configuration details."""
        str = "Configurations for %s:\n" % self.mode
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                str += "{:30} {}\n".format(a, getattr(self, a))
        return str
