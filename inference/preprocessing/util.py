import configparser

import numpy as np

config = configparser.ConfigParser()
config.read('../config.ini')


print(config['CODING']['med'])