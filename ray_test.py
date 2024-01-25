



import ray
import time
import numpy as np

import asyncio

'''
@ray.remote
class Writer(): 
    async def waiter(self, event): 
        
        print('waiting for it ...')
        print('ciaoo2')
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        
        await event.wait()
        print(current_time,'... got it!')
        return
'''





@ray.remote
class GlobalVarActor():
    def __init__(self):
         self.global_var = 0
         self.samples= np.array([])
         
         
    
    async def set_global_var(self, var,event):
        await event.wait
        self.global_var = (var)
        return self.global_var

    def get_samples(self):
            return (self.samples)

    def append_variable(self, var):
         print('var', var)
         self.samples = np.concatenate((self.samples, var))








@ray.remote 
class Main():

    def __init__(self, Writer, glob_var):
        self.writer =Writer.remote()
        self.glob_var = glob_var

    async def read(self,sample, event):
        print(sample)
        self.glob_var.append_variable.remote(np.array([sample]))
        #self.writer.waiter.remote(event)
        
        return   
    
#glob_var = GlobalVarActor.remote()
#mainn = Main.remote(Writer, glob_var)


if __name__ =='__main__':
    '''
    import time
    event = asyncio.Event()
    for i in range(10): 
        
        ray.get(mainn.read.remote(i, event))
        time.sleep(2)
        if len(ray.get(glob_var.get_samples.remote()))==5:
            print('lunghezza =5')
        event.set()
        
    print(ray.get(glob_var.get_samples.remote()))
    '''

    from bilby_simulator  import Bilby
    bilby = Bilby()
    print(bilby.samples)
    while True:
        bilby.produce_samples()
        print(bilby.samples)
    

    







