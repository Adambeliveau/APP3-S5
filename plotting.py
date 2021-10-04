import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def add_plotting_args(plt_type="", xaxis=None, yaxis=None, title="", xlabel="", ylabel="") -> dict:
    """
    used to return a dictionary containing info used for plotting
    :param plt_type: type of the plot -> plot or stem
    :param xaxis: x axis data
    :param yaxis: y axis data
    :param title: title for the plot
    :param xlabel: label for the x axis
    :param ylabel: label for the y axis
    :return: formatted dictionary
    """
    return {'plt_type': plt_type,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel}


def build_plots(pp: PdfPages, *args) -> None:
    """
     Builds plots from the data extracted from previous functions
    :param pp: pdf file where the figures are going to be saved
    :param args: contains dict with parameters to build plots
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
        pp.savefig()
    pp.close()
