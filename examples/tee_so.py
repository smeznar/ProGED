import sys

"""Solution class for copying output to log file; copied from
https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file/616686#616686
"""

class Tee(object):
    def __init__(self, name, mode="w"):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
        # print("Destructor __del__ was called inside SO's Tee.")
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

class TeeFileOnly(object):
    def __init__(self, name, mode="w"):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
        # print("Destructor __del__ was called inside SO's Tee.")
    def write(self, data):
        self.file.write(data)
    def flush(self):
        self.file.flush()

class Mute(object):
    def __init__(self):
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
    def write(self, data):
        pass
    def flush(self):
        pass
        
# # # # # # # # # # # # # # # # # # # # # # # #
# redirect stdout, also file descriptors
# copied from:
# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
import os
import sys
from contextlib import contextmanager

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

# The same example works now if stdout_redirected() is used instead of redirect_stdout():

# import os
# import sys

# stdout_fd = sys.stdout.fileno()
# with open('output.txt', 'w') as f, stdout_redirected(f):
#     print('redirected to a file')
#     os.write(stdout_fd, b'it is redirected now\n')
#     os.system('echo this is also redirected')
# print('this is goes back to stdout')

