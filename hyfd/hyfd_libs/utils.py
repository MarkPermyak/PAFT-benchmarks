import os
import json
import datetime
import time


class Stats(object):
    def __init__(self, logger, headers, restart=False, stat_directory='./results/'):
        self.logger = logger
        self.headers = headers
        self.STAT_DIRECTORY = stat_directory
        self.STAT_FILE = 'hyfd_results.txt'

        if not os.path.isdir(self.STAT_DIRECTORY):
            try:
                os.mkdir(self.STAT_DIRECTORY)
                self.logger.info("Directory Created: {}".format(self.STAT_DIRECTORY))
            except:
                self.logger.error("Director does not exists: {}".format(self.STAT_DIRECTORY))
                self.logger.error("EXITING: Could not create directory: {}".format(self.STAT_DIRECTORY))
                exit()
        if not os.path.isfile(self.STAT_DIRECTORY+self.STAT_FILE) or restart:
            with open(self.STAT_DIRECTORY+self.STAT_FILE, 'w') as fout:
                self.logger.info("Results File Initialized: {}".format(self.STAT_DIRECTORY+self.STAT_FILE))
                fout.write('{}\n'.format('\t'.join(self.headers)))

    def log_results(self, results):
        with open(self.STAT_DIRECTORY+self.STAT_FILE, 'a') as fout:
            fout.write('{}\n'.format('\t'.join(results)))

 

class Output(object):
    def __init__(self, logger, db_path, output_directory='./json/', output_fname='{}-{}.json'):
        self.logger = logger
        self.dbname = db_path[db_path.rfind('/')+1:db_path.rfind('.')]
        self.OUTPUT_DIRECTORY = output_directory
        self.OUTPUT_FNAME = output_fname

        self.st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        
        
        if not os.path.isdir(self.OUTPUT_DIRECTORY):
            try:
                os.mkdir(self.OUTPUT_DIRECTORY)
                self.logger.info("Directory Created: {}".format(self.OUTPUT_DIRECTORY))
            except:
                self.logger.error("Director does not exists: {}".format(self.OUTPUT_DIRECTORY))
                self.logger.error("EXITING: Could not create directory: {}".format(self.OUTPUT_DIRECTORY))
                exit()
        self.fout_path = self.OUTPUT_DIRECTORY+self.OUTPUT_FNAME.format(self.dbname, self.st)

    def write(self, fds):
        with open(self.fout_path, 'w') as fout:
            json.dump(list(fds), fout)
            self.logger.info("FDs written in: {}".format(self.fout_path))