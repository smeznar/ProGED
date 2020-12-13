import sys  # To import from parent directory.

# from IPython.utils.io import Tee  # Log results using 3th package.
from tee_so import Tee  # Log using manually copied class from a forum.

# # 0.) Log output to log_<nickname><random>.log file

def create_log():
    log_nickname = ""
    if len(sys.argv) >= 2:
        log_nickname = sys.argv[1]
    random = str(np.random.random())
    print("log id:", log_nickname + random)
    try:
        log_object = Tee("examples/log_" + log_nickname + random + ".log")
    except FileNotFoundError:
        log_object = Tee("log_" + log_nickname + random + ".log")

