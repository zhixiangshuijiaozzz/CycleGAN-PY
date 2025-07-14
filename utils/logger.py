import os
from datetime import datetime

class Logger:
    def __init__(self, opt, session_dir: str = None):
        """
        如果传入 session_dir，则所有日志都写到 session_dir；
        否则写到 checkpoints/opt.name。
        """
        if session_dir:
            self.log_dir = session_dir
        else:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_name = os.path.join(self.log_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            log_file.write(
                '================ Training Loss (%s) ================\n' %
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

    def log(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
