
#   Stdlib
import math
import logging
import logging.config
import logging.handlers
import pdb
from collections import Counter
from pprint import pformat

#   3rd party
import yaml
from termcolor import colored

from ..python_utilities.logutils import *


CONFIG_LOGGING = """
    version: 1
    formatters:
        brief:
            fmt: '%(name)-12s: %(levelname)-8s %(message)s'
        long:
            fmt: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        color:
            (): ColorFormatter
            fmt: '%(name)-12s: %(levelname)-8s [%(module)s.%(funcName)s] %(message)s'
        border:
            (): BorderFormatter
            fmt: '%(name)-12s: %(levelname)-8s [%(module)s.%(funcName)s] %(message)s'
        border_long:
            (): BorderFormatter
            fmt: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        border_color:
            (): BorderColorFormatter
            fmt: '%(name)-12s: %(levelname)-8s [%(module)s.%(funcName)s] %(message)s'
    filters:
        extra:
            (): ExtraFilter
        extrakey:
            (): ValidationFilter
            prefix: 'Validation [extra key]'
        rate:
            (): RateFilter
    handlers:
        console:
            class: logging.StreamHandler
            formatter: border_color
            level: INFO
            filters: [rate, extra]
            stream: ext://sys.stdout
        file:
            class: logging.handlers.RotatingFileHandler
            formatter: border_long
            level: DEBUG
            filters: [rate, extra]
            filename: 'cluster.log'
            maxBytes: 1024
            backupCount: 3
    loggers:
        clustering:
            handlers: [console, file]
    root:
        level: DEBUG
"""

def setup_logging():
    config = yaml.load(CONFIG_LOGGING)
    for k in ['formatters', 'filters']:
        eval_class(config, k)
    return config


if __name__ == '__main__':
    config = setup_logging()
    logging.config.dictConfig(config)
    logger = logging.getLogger('clustering')
    logger.info('info - testing', extra=dict(border=True))
    logger.info('info - testing 2',)
    logger.warn('warn - testing', extra=dict(border=True))
    logger.error('error - testing', extra=dict(border=True))
    logger.critical('critical - testing', extra=dict(border=True))
