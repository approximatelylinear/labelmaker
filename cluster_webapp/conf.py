
import yaml
from cluster_webapp.lib.python_utilities.logutils import *


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
        console_debug:
            class: logging.StreamHandler
            formatter: border_color
            level: DEBUG
            filters: [rate, extra]
            stream: ext://sys.stdout
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
        base:
            handlers: [console, file]
        elasticsearch:
            handlers: [console_debug]
    root:
        level: DEBUG
"""

def setup_logging():
    config = yaml.load(CONFIG_LOGGING)
    return config


