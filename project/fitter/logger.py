# import config
# conf = config.read_config()
LEVEL = "DEBUG"

import coloredlogs, logging
# Create a logger object.
logger = logging.getLogger(__name__)

# By default the install() function installs a handler on the root logger,
# this means that log messages from your code and log messages from the
# libraries that you use will all show up on the terminal.
# coloredlogs.install(level='DEBUG')

# If you don't want to see log messages from libraries, you can pass a
# specific logger object to the install() function. In this case only log
# messages originating from that logger will show up on the terminal.
coloredlogs.install(level=LEVEL, logger=logger)
