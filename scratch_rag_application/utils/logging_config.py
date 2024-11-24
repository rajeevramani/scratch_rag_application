# config/logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
import os


def setup_logging(log_file='app.log', main_module_name='__main__'):
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file)

    # Updated formatters to include class name and method
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s'
    )

    # First, set the root logger to WARNING to control external libraries
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # Configure application-specific logger
    app_logger = logging.getLogger('scratch_rag_application')
    app_logger.setLevel(logging.DEBUG)
    app_logger.propagate = False

    # Configure main module logger
    main_logger = logging.getLogger(main_module_name)
    main_logger.setLevel(logging.DEBUG)
    main_logger.propagate = False

    # File handler for application logs
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler for application logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Remove existing handlers if any
    app_logger.handlers.clear()
    main_logger.handlers.clear()

    # Add handlers to both loggers
    app_logger.addHandler(file_handler)
    app_logger.addHandler(console_handler)
    main_logger.addHandler(file_handler)
    main_logger.addHandler(console_handler)

    # Explicitly silence common noisy libraries
    for logger_name in [
        'matplotlib',
        'PIL',
        'chromadb',
        'langchain',
        'urllib3',
        'requests',
        'asyncio',
        'charset_normalizer',
        'bs4',
        'sentence_transformers'
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Allow through certain important library info logs
    logging.getLogger('chromadb.telemetry').setLevel(logging.INFO)
