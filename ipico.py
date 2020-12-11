# from IPython.utils import io
# obj = io.Tee("new.log")
# print(123)
# print(5656)
# obj.close()

# from tee import Tee
# # log_stdout = Tee("tee.log")
# log_stdout = Tee(filename="tee.log")
# print(123, "tee")
# print(5656)
# log_stdout.close()
# 


from tee_so import Tee
log_stdout = Tee("tee.log")
# log_stdout = Tee(filename="tee.log")
print(101123, "tee")
print(5656)
# log_stdout.close()


