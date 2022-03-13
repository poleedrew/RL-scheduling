from djsp_logger import DJSP_Logger

if __name__ == '__main__':
    # in_file = '~/JSP/Ortools/JSP/ortools_result/abz5.json'
    in_file = '/mnt/nfs/work/oo12374/JSP/Ortools/JSP/ortools_result/abz5.json'
    logger = DJSP_Logger()
    logger.load(in_file)
    print(logger)
    print(logger.google_chart_front_text)
    print(logger.google_chart_back_text)
