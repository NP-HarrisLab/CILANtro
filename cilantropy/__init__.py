import logging
import sys

logger = logging.getLogger("cilantropy")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)

logger.addHandler(console)
logger.propagate = False
