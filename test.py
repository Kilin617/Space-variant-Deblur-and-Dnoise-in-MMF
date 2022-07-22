import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from pytorch_msssim import ssim as SSIM
from PIL import Image

if __name__ == '__main__':

	opt = TestOptions().parse()
	opt.nThreads = 1   # test code only supports nThreads = 1
	opt.batchSize = 1  # test code only supports batchSize = 1
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True  # no flip
	opt.model = 'test'
	opt.dataset_mode = 'single'
	opt.fineSize = 0

	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)
	visualizer = Visualizer(opt)
	# create website
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
	# test
	avgPSNR = 0.0
	avgSSIM = 0.0
	counter = 0

	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		model.set_input(data)
		model.test()
		visuals = model.get_current_visuals()
		img_path = model.get_image_paths()
		with open('./results/time_consuming-2022-0514-1.txt', 'a') as f:
			f.write('Process image:{} \n'.format(img_path))
			print('process image... %s' % img_path)
		visualizer.save_images(webpage, visuals, img_path)

	webpage.save()
