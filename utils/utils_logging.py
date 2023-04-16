import logging
import typing as tp
import tqdm


def init_logger(
    name: str,
    filename: tp.Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)

    str_format = '%(asctime)s [%(threadName)-12.12s] [%(name)s] [%(levelname)-5.5s]  %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=str_format, datefmt=date_format)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(formatter)
    logger.addHandler(tqdm_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(log_level)
    return logger


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, log_level: int = logging.INFO) -> None:
        super().__init__(log_level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
