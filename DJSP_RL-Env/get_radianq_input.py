from djsp_logger import DJSP_Logger

if __name__ == '__main__':
    logger = DJSP_Logger()
    logger.load('test/test_scheduling.pth')
    logger.radiantQ_json()