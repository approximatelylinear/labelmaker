
#   Stdlib
import os
import logging

#   3rd party
import yaml

#   Custom
from cluster_webapp.lib.python_utilities.logutils import *


#:  The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(THIS_DIR, 'data')

DOWNLOAD_PATH = os.path.join(DATA_PATH, 'clusters', 'labeled')
ALLOWED_EXTENSIONS = set(['txt', 'csv', 'xlsx', 'xls'])

RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
CLEAN_DATA_PATH = os.path.join(DATA_PATH, 'clean')
FINAL_DATA_PATH = os.path.join(DATA_PATH, 'final')
META_DATA_PATH = os.path.join(DATA_PATH, 'meta')
CLUSTERS_DATA_PATH = os.path.join(DATA_PATH, 'clusters')

FIXTURES_DATA_PATH = os.path.join(DATA_PATH, 'fixtures')


CONFIG_LOGGING = """
    version: 1
    formatters:
        brief:
            fmt: '%(name)-12s: %(levelname)-8s %(message)s'
        long:
            fmt: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        color:
            #(): cluster_webapp.lib.python_utilities.logutils.ColorFormatter
            fmt: '%(name)-12s: %(levelname)-8s [%(module)s.%(funcName)s] %(message)s'
        border:
            #(): cluster_webapp.lib.python_utilities.logutils.BorderFormatter
            fmt: '%(name)-12s: %(levelname)-8s [%(module)s.%(funcName)s] %(message)s'
        border_long:
            #(): cluster_webapp.lib.python_utilities.logutils.BorderFormatter
            fmt: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        border_color:
            #(): cluster_webapp.lib.python_utilities.logutils.BorderColorFormatter
            fmt: '%(name)-12s: %(levelname)-8s [%(module)s.%(funcName)s] %(message)s'
    filters:
        extra:
            (): cluster_webapp.lib.python_utilities.logutils.ExtraFilter
        extrakey:
            (): cluster_webapp.lib.python_utilities.logutils.ValidationFilter
            prefix: 'Validation [extra key]'
        rate:
            (): cluster_webapp.lib.python_utilities.logutils.RateFilter
    handlers:
        console_debug:
            class: logging.StreamHandler
            #formatter: border_color
            level: DEBUG
            #filters: [rate, extra]
            stream: ext://sys.stdout
        console:
            class: logging.StreamHandler
            #formatter: border_color
            level: INFO
            #filters: [rate, extra]
            stream: ext://sys.stdout
        file:
            class: logging.handlers.RotatingFileHandler
            #formatter: border_long
            level: DEBUG
            #filters: [rate, extra]
            filename: 'cluster.log'
            maxBytes: 1024
            backupCount: 3
    loggers:
        clustering:
            handlers: [console, file]
        base:
            handlers: [console, file]
        elasticsearch:
            handlers: [console_debug]
    root:
        level: DEBUG
"""


def setup_logging():
    config = yaml.load(CONFIG_LOGGING)
    logging.config.dictConfig(config)


