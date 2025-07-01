import sys
from pyscf import lib

class Logger(lib.logger.Logger):
    def __init__(self, stdout = sys.stdout, verbose: int = 3, rank: int = 0):
        super().__init__(stdout, verbose)
        self.rank = rank
    
    # Setters
    def set_stdout(self, stdout):
        self.stdout = stdout

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_rank(self, rank):
        self.rank = rank

    # Getters
    def get_stdout(self):
        return self.stdout

    def get_verbose(self):
        return self.verbose

    def get_rank(self):
        return self.rank

    # Print
    def flush_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.flush(self, msg, *args)

    def log_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.log(self, msg, *args)
    
    def error_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.error(self, msg, *args)
    
    def warn_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.warn(self, msg, *args)
    
    def info_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.info(self, msg, *args)
    
    def note_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.note(self, msg, *args)
    
    def debug_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.debug(self, msg, *args)
    
    def debug1_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.debug1(self, msg, *args)
    
    def debug2_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.debug2(self, msg, *args)
    
    def debug3_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.debug3(self, msg, *args)
    
    def debug4_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.debug4(self, msg, *args)
    
    def stdout_0(self, msg, *args):
        if self.rank == 0:
            lib.logger.stdout(self, msg, *args)
    
    def timer_0(self, msg, cpu0=None, wall0=None):
        if self.rank == 0:
            return lib.logger.timer(self, msg, cpu0=None, wall0=None)
    
    def timer_debug1_0(self, msg, cpu0=None, wall0=None):
        if self.rank == 0:
            return lib.logger.timer_debug1(self, msg, cpu0=None, wall0=None)

    def flush(self, msg, *args):
        lib.logger.flush(self, msg, *args)

    def log(self, msg, *args):
        lib.logger.log(self, msg, *args)
    
    def error(self, msg, *args):
        lib.logger.error(self, msg, *args)
    
    def warn(self, msg, *args):
        lib.logger.warn(self, msg, *args)
    
    def info(self, msg, *args):
        lib.logger.info(self, msg, *args)
    
    def note(self, msg, *args):
        lib.logger.note(self, msg, *args)
    
    def debug(self, msg, *args):
        lib.logger.debug(self, msg, *args)
    
    def debug1(self, msg, *args):
        lib.logger.debug1(self, msg, *args)
    
    def debug2(self, msg, *args):
        lib.logger.debug2(self, msg, *args)
    
    def debug3(self, msg, *args):
        lib.logger.debug3(self, msg, *args)
    
    def debug4(self, msg, *args):
        lib.logger.debug4(self, msg, *args)
    
    def stdout(self, msg, *args):
        lib.logger.stdout(self, msg, *args)
    
    def timer(self, msg, cpu0=None, wall0=None):
        return lib.logger.timer(self, msg, cpu0=None, wall0=None)
    
    def timer_debug1(self, msg, cpu0=None, wall0=None):
        return lib.logger.timer_debug1(self, msg, cpu0=None, wall0=None)

#import sys
#from mpi4py import MPI 
#from ad_afqmc import config

## MPI rank needed for the logger
#mpi_comm = config.setup_comm_no_print()
#rank = mpi_comm.COMM_WORLD.Get_rank()

# Logger
log = Logger()
