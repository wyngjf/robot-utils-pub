import logging
import coloredlogs
from icecream import install, ic
from rich import pretty
from rich.console import Console
console = Console()

install()
ic.configureOutput(includeContext=True)
coloredlogs.install(fmt='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s | %(lineno)d] %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d:%H:%M:%S')
pretty.install()
