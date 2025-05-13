
import os
import platform
import logging.config



# Initialize
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
version_file_path = os.path.join(base_path, 'version.txt')
LOGGING_NAME = f"LLM_TRAINER_{open(version_file_path).read().strip()}"
VERBOSE = True
RANK = int(os.getenv('RANK', -1))
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans



def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR

    class ColorFormatter(logging.Formatter):
        """Custom formatter to add colors to log messages using colorstr."""
        def format(self, record):
            if record.levelname == "ERROR":
                record.msg = colorstr("red", record.msg)
            elif record.levelname == "WARNING":
                record.msg = colorstr("yellow", record.msg)
            # elif record.levelname == "INFO":
            #     record.msg = colorstr("green", record.msg)
            elif record.levelname == "DEBUG":
                record.msg = colorstr("blue", record.msg)
            return super().format(record)

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                '()': ColorFormatter,  # Use the custom formatter
                'format': '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
            }
        },
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level
            }
        },
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False
            }
        }
    })
    

set_logging(LOGGING_NAME, verbose=VERBOSE) 
LOGGER = logging.getLogger(LOGGING_NAME)



def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr('blue', 'bold', 'hello world')
        >>> '\033[34m\033[1mhello world\033[0m'
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']



def log(message, level='info', color=False):
    if level.lower() == 'warning':
        LOGGER.warning(message)
    elif level.lower() == 'error':
        LOGGER.error(message)
    else:
        if color:
            LOGGER.info(colorstr(message))
        else:
            LOGGER.info(message)
