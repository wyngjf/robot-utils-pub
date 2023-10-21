import logging


def disable_logging():
    logging.disable(logging.CRITICAL)


def enable_logging():
    logging.disable(logging.NOTSET)


def set_warning_to_exception():
    import warnings
    warnings.filterwarnings('error')
