import os

class Logger:

    def __init__(self, filename='log.txt'):
        self.logs = []
        self.logs.append(FileLogger(filename))
        self.logs.append(ConsoleLogger())

    def log(self, content):
        for logger in self.logs:
            logger.log(content)

    def close(self):
        for logger in self.logs:
            logger.close()


class ConsoleLogger:

    def log(self, content):
        print(content)

    def close(self):
        pass

class FileLogger:

    def __init__(self, filename):
        self.file = open(filename, 'w')

    def log(self, content):
        self.file.write(content + '\n')
        self.file.flush()

    def close(self):
        self.file.close()
