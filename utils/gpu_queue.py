from utils.parallel import MyPool
from multiprocessing import Manager
from utils.general import GPUManager, Log
import time
import importlib


class GPUQueue:
    """
    Self-defined GPUQueue, not usable yet
    """
    def __init__(self, gpu_list, indi_size=40):
        """
        :param gpu_list: List[int]
        """
        self.size = len(gpu_list)
        self.gpu_list = gpu_list
        self.manager = Manager()
        self.gpu_queue = self.manager.Queue(maxsize=self.size)
        self.indi_queue = self.manager.Queue(maxsize=indi_size)
        self.pool = MyPool(processes=self.size)
        self.gpu_manager = GPUManager()
        # add available GPU IDs into Queue
        for gpu_id in gpu_list:
            self.gpu_queue.put(gpu_id)

    def submit(self, module_name: str):
        self.indi_queue.put(module_name)

    def start_work(self, gpu_retry_times=3, query_interval=30):
        def call_back(res):
            """
            Callback function for releasing GPU
            """
            self.gpu_queue.put(res)

        while not self.indi_queue.empty():
            module_name = self.indi_queue.get()
            gpu_id = self.gpu_queue.get()
            retry_times = 1
            try:
                while gpu_id not in self.gpu_manager.get_free_gpu(gpu_list=self.gpu_list):
                    Log.warn(
                        "Retry#%d, GPU %d is busy! Check if others are using the same GPU!" % (retry_times, gpu_id))
                    retry_times += 1
                    time.sleep(query_interval)
                    if retry_times > gpu_retry_times:
                        raise RuntimeError("GPU %d is busy for %d query cycles." % gpu_retry_times)

                # get a free GPU -> gpu_id
                _module = importlib.import_module('.', module_name)
                _class = getattr(_module, 'RunModel')
                class_obj = _class()
                self.pool.apply_async(func=class_obj.do_work, args=(gpu_id,), callback=call_back)
            except RuntimeError:
                self.indi_queue.put(module_name)  # rearrange this individual for next round
                self.indi_queue.put(gpu_id)  # rearrange this gpu
