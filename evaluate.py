import argparse
import os
import sys
import time
from typing import Union

from call_methods import make_model
from options.evaluate_option import EvaluateOptions
from utils import fid_score, tb_visualizer
from utils.utils import delete_directory, delete_files_in_directory, set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(opt: argparse.Namespace) -> Union[float, None]:
    """
    Run the evaluation process

    Parameters
    ----------
    opt: argparse.Namespace
        The parsed arguments
    """

    # Set seed
    set_seed(opt.seed)
    if opt.is_conditional:
        opt.num_images = opt.num_images // opt.n_classes
    else:
        opt.label = 0

    model = make_model(opt.model_name, opt)
    visualizer = tb_visualizer.Visualizer(opt)
    start = time.time()
    model.load_trained_generator(opt.model_path)
    if opt.is_conditional:
        for i in range(opt.n_classes):
            model.set_label(i)
            model.predict()
            visualizer.log_image(model.vis_data, total_steps=i, is_train=False)
    else:
        model.predict()
        visualizer.log_image(model.vis_data, total_steps=opt.label, is_train=False)
    end = time.time()
    visualizer.log_time(end, start, epoch=1, is_train=False, training_end=True)
    opt.path = [opt.images_folder, visualizer.image_folder]

    fid = fid_score.run(opt)
    print(f"FID Score for {opt.model_path} is {fid}")
    delete_files_in_directory(visualizer.log_dir)
    delete_directory(visualizer.log_dir)

    visualizer.close()
    return fid


if __name__ == "__main__":
    opt = EvaluateOptions().parse()
    run(opt=opt)
