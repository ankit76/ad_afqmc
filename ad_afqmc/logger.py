from pyscf import lib

class Logger(lib.logger.Logger):
    def __init__(self, stdout, verbose: int, rank: int = 0):
        super().__init__(stdout, verbose)
        self.rank = rank

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
 
