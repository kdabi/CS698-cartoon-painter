import time
import os
from pix2pix import Pix2Pix
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from test_model import TestModel
from util.visualizer import Visualizer
from util import html

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
         self.isTrain =  False
         self.lambda_A =  100.0
         self.lambda_B =  10.0
         self.loadSize =  286
         self.lr =  0.0002
         self.lr_decay_iters =  50
         self.max_dataset_size =  10000000
         self.model =  "pix2pix"
         self.nThreads =  1
         self.n_layers_D =  3
         self.name =  "facades_pix2pix"
         self.ndf =  64
         self.ngf =  64
         self.niter =  100
         self.niter_decay =  100
         self.no_dropout =  False
         self.no_flip =  True
         self.no_html =  False
         self.no_lsgan =  True
         self.output_nc =  3
         self.phase =  "test"
         self.pool_size =  0
         self.print_freq =  100
         self.resize_or_crop =  "resize_and_crop"
         self.save_epoch_freq =  5
         self.save_latest_freq =  5000
         self.serial_batches =  True
         self.which_direction =  "BtoA"
         self.which_epoch =  "latest"
         self.how_many = 106 # number of images on which testModel will run
         self.checkpoints_dir = "/data/kdabi/CS698O/Autopainter/CS698-cartoon-painter/saved_models"
         self.results_dir = "/data/kdabi/CS698O/Autopainter/CS698-cartoon-painter/saved_models"

opt = Options()
# data_loader = CreatDataLoader(opt)
data_loader = CustomDatasetDataLoader()
data_loader.initialize(opt)
dataset = data_loader.load_data()
model = TestModel(opt)
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('Generating output for image..... %s' %img_path)
    visualizer.save_images(webpage, visuals, img_path)


webpage.save()
