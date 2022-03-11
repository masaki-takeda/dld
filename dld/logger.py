from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, path):
        print("start logging into: {}".format(path))
        self.writer = SummaryWriter(path)
        
    def log(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def flush(self):
        self.writer.file_writer.flush()
        
    def close(self):
        self.writer.close()
