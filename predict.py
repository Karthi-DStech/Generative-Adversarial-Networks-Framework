import os
import sys
import time

from call_methods import make_model
from options.predict_option import PredictOptions
from utils import tb_visualizer
from utils.utils import set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run() -> None:
    """
    Run the prediction process

    Parameters
    ----------
    None

    Returns
    -------
    None

    Process
    -------
    1. Parse the prediction options
    2. Set the random seed
    3. Create the model
    4. Load the model weights
    5. Predict the output
    6. Save the output
    """
    opt = PredictOptions().parse()

    # Set seed
    set_seed(opt.seed)

    model = make_model(opt.model_name, opt)
    visualizer = tb_visualizer.Visualizer(opt)
    start = time.time()
    model.load_trained_generator(opt.model_path)
    model.predict()
    visualizer.log_image(model.vis_data, total_steps=opt.label, is_train=False)
    end = time.time()
    visualizer.log_time(end, start, epoch=1, is_train=False, training_end=True)
    visualizer.close()


if __name__ == "__main__":
    run()
