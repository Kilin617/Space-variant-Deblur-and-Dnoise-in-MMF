import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from multiprocessing import freeze_support


def train(opt, _data_loader, model, visualizer):
	# load data
	dataset = _data_loader.load_data()
	dataset_size = len(_data_loader)
	print('#training images = %d' % dataset_size)

	total_steps = 0
	total_epoch = opt.niter + opt.niter_decay
	for epoch in range(opt.epoch_count, total_epoch + 1):
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()

			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				visualizer.display_current_results(results, epoch)

			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()

				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_errors(epoch, total_epoch, epoch_iter, dataset_size, errors, t)
				if opt.display_id > 0:
					for item in errors.items():
						visualizer.plot_current_errors_tuple(epoch, float(epoch_iter)/dataset_size, opt, item)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				model.save('latest')

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save('latest')
			model.save(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate()


if __name__ == '__main__':
	freeze_support()

	opt = TrainOptions().parse()
	opt.resize_or_crop = "crop"
	opt.save_latest_freq = 100

	data_loader = CreateDataLoader(opt)
	model = create_model(opt)
	visualizer = Visualizer(opt)
	train(opt, data_loader, model, visualizer)
