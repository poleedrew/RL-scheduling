from matplotlib.font_manager import json_dump
from djsp_plotter import DJSP_Plotter
from djsp_logger import DJSP_Logger

if __name__ == '__main__':
    json_in_file = 'abz5.json'
    logger = DJSP_Logger()
    plotter = DJSP_Plotter(logger)
    logger.load(json_in_file)
    # print(logger)
    plotter.plot_googlechart_timeline('abz5_google_chart.html')
    plotter.plot_plotly_timeline('abz5_plotly.html')

