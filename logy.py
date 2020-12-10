# import logging
# # logging.basicConfig(filename="my.log", level=logging.INFO)  # Overwrites
# logging.basicConfig(filename="my.log", level=logging.DEBUG)  # Overwrites
# print(22)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('test.log', 'a'))
print = logger.info

print('yo!')
