import time
import math
from pix2pix import Pix2Pix

# from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
# from models.models import create_model
# from util.visualizer import Visualizer

class Options():
    def __init__(self):
        # super(Options, self).__init__()
        self.batchSize =  1
        self.beta1 =  0.5
        self.continue_train =  False
        self.dataroot =  "/data/anil/datasets/facades"
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
        self.lambda_B =  10.0
        self.loadSize =  286
        self.lr =  0.0002
        self.lr_decay_iters =  50
        self.max_dataset_size =  10000000
        self.model =  "pix2pix"
        self.nThreads =  2
        self.n_layers_D =  3
        self.name =  "facades_pix2pix"
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

opt = Options()


# opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
# dataset_size = len(data_loader)
# print('#training images = %d' % dataset_size)

model = Pix2Pix(opt)
# visualizer = Visualizer(opt)
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

        # if total_steps % opt.display_freq == 0:
        #     visualizer.display_current_results(model.get_current_visuals(), epoch)

        # if total_steps % opt.print_freq == 0:
        #     errors = model.get_current_errors()
        #     t = (time.time() - iter_start_time) / opt.batchSize
            # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            # if opt.display_id > 0:
            #     visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
        # print('epoch %d, total_steps %d' %
        #           (epoch, total_steps))

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