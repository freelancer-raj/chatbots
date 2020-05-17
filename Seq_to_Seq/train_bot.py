import numpy as np
import tensorflow as tf
import re
import time
import os

working_directory = os.getcwd()
data_directory = os.path.join(working_directory, 'data')

# load data
lines = open(os.path.join(data_directory, 'movie_lines.txt'), encoding='utf-8', errors='ignore').read().split('\n')
conversaations = open(os.path.join(data_directory, 'movie_conversations.txt'), encoding='utf-8', errors='ignore').read().split('\n')







