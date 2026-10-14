import os

try:
    # Newer tensorboard_logger versions may be missing 'configure' / 'log_value'.
    from tensorboard_logger import configure, log_value
    _HAS_TB_LOGGER = True
except Exception:
    _HAS_TB_LOGGER = False


class Logger(object):
    """
    Thin wrapper that prefers tensorboard_logger when available and
    silently degrades to a no-op otherwise, so the training script
    never breaks just because logging isn't available.
    """

    def __init__(self, log_dir):
        self._remove(log_dir)
        self.global_step = 0
        if _HAS_TB_LOGGER:
            try:
                configure(log_dir)
            except Exception as e:
                print(f'[logger] tensorboard_logger.configure failed: {e}; '
                      f'falling back to no-op logger.')
                self._tb_logger_ok = False
                return
        self._tb_logger_ok = _HAS_TB_LOGGER

    def log_value(self, name, value):
        if not self._tb_logger_ok:
            return self
        try:
            log_value(name, value, self.global_step)
        except Exception:
            self._tb_logger_ok = False
        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        import shutil
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)