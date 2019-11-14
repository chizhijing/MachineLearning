from multiprocessing.pool import Pool
import time

def my_function(x):
    print('this is begin at',x)
    time.sleep(x)
    print('this is end at ',x)


if __name__=="__main__":
    pool =Pool(4)
    pool.map(my_function,[1,2,3,4])