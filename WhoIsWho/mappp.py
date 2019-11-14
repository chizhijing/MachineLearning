from multiprocessing.pool import Pool
from functools import  partial
import time

def my_function(y,z,x):
    print('this is begin at',x,y,z)
    time.sleep(x)
    print('this is end at ',x,y,z)


if __name__=="__main__":
    # pool =Pool(4)
    # n_f=partial(my_function,y='xxx',z='zzz')
    # pool.map(n_f,[1,2,3,4])
    l = [(10, 0.05, 0.2081435163183775), (10, 0.1, 0.20813988145692897), (10, 0.15000000000000002, 0.20847276288034908)]
    import pandas as pd
    pd.DataFrame(l).to_csv('e:\\ddfd.csv')
