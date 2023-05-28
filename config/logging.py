import logging
import logging.config


def config_logger():
    logging.config.fileConfig(fname="logging-dev.conf", disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    logger.info("logger configured")
