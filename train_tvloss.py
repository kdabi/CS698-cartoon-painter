import time
import math
from tvloss import TotalVarianceLoss
from util.visualizer import Visualizer

# from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader

# from models.models import create_model
# from util.visualizer import Visualizer

class Options():
    def __init__(self):
        # super(Options, self).__init__()
        self.batchSize =  1
        self.beta1 =  0.5
        self.continue_train =  False
        self.dataroot =  "/data/kdabi/CS698O/Autopainter/CS698-cartoon-painter/Dataset_Generator"
        self.display_freq =  100
        self.display_id =  1
        self.display_port =  8097
        self.display_single_pane_ncols =  0
        self.display_winsize =  256
        self.epoch_count =  1
        self.fineSize =  256
        self.identity =  0.0
        self.input_nc =  3
        self.isTrain =  True
        self.lambda_A =  100.0
        self.lambda_B =  0.01
        self.lambda_C =  0.001
        self.loadSize =  286
        self.lr =  0.0002
        self.lr_decay_iters =  50
        self.max_dataset_size =  10000000
        self.model =  "tvloss"
        self.nThreads =  2
        self.n_layers_D =  3
        self.name =  "facades_tvloss"
        self.ndf =  64
        self.ngf =  64
        self.niter =  100
        self.niter_decay =  100
        self.no_dropout =  False
        self.no_flip =  False
        self.no_html =  False
        self.no_lsgan =  True
        self.output_nc =  3
        self.phase =  "train"
        self.pool_size =  0
        self.print_freq =  100
        self.resize_or_crop =  "resize_and_crop"
        self.save_epoch_freq =  5
        self.save_latest_freq =  5000
        self.serial_batches =  False
        self.which_direction =  "BtoA"
        self.which_epoch =  "latest"
        self.checkpoints_dir = "/data/kdabi/CS698O/Autopainter/CS698-cartoon-painter/saved_models"
        self.results_dir = "/data/kdabi/CS698O/Autopainter/CS698-cartoon-painter/saved_models"


opt = Options()


# opt = TrainOptions().parse()


data_loader = CustomDatasetDataLoader()
data_loader.initialize(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
# print('#training images = %d' % dataset_size)

model = TotalVarianceLoss(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
