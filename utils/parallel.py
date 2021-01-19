"""
Code from https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
To allow subprocess to use PyTorch Multiprocessing DataLoader
"""
import multiprocessing.pool


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
