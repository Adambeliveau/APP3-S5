import matplotlib.pyplot as plt
from tqdm import tqdm


def add_plotting_args(plt_typle="", xaxis=None, yaxis=None, title="", xlabel="", ylabel=""):
    return {'plt_type': plt_typle,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel}


def build_plots(*args) -> None:
    """
     Builds plots from the data extracted from previous functions
    :param *args: contains dict with parameters to build plots
    :return: None
    """
    for plotting_args in tqdm(args, desc='building plots'):
        plt.figure()
        if plotting_args['xaxis'] is None:
            if plotting_args['plt_type'] == 'plot':
                plt.plot(plotting_args['yaxis'])
            elif plotting_args['plt_type'] == 'stem':
                plt.stem(plotting_args['yaxis'])
        else:
            if plotting_args['plt_type'] == 'plot':
                plt.plot(plotting_args['xaxis'], plotting_args['yaxis'])
            elif plotting_args['plt_type'] == 'stem':
                plt.stem(plotting_args['xaxis'], plotting_args['yaxis'])
        plt.title(plotting_args['title'])
        plt.xlabel(plotting_args['xlabel'])
        plt.ylabel(plotting_args['ylabel'])
