import logging
import os


def reset_logging():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


script_path = os.path.dirname(os.path.realpath(__file__))

for i in range(5):
    reset_logging()
    log_dir = os.path.join(script_path, 'test', f'seed-{i}')
    os.makedirs(log_dir, exist_ok=True)

    log_file_name = os.path.join(log_dir, 'optimisation.log')
    if os.path.isfile(log_file_name):
        filemode = "a"
    else:
        filemode = "w"
    logging.basicConfig(level=logging.NOTSET, filemode=filemode,
                        filename=log_file_name,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("New log file created")
