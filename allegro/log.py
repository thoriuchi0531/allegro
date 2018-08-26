import logging

_loggers = {}


def get_logger(name=None, filename=None):

    if name not in _loggers.keys():

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                      '%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)

        if filename is not None:
            # create file handler which logs even debug messages
            fh = logging.FileHandler(filename)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        _loggers[name] = logger

        return logger

    else:
        return _loggers[name]
