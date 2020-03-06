import os
from glob import glob
from spell.spell import LogParser


if __name__ == '__main__':
    input_dir = './data'
    output_dir = './result'
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    tau = 0.5

    parser = LogParser(indir=input_dir, outdir=output_dir, log_format=log_format,
                    tau=tau, vm_id='', logmain='HDFS')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for pwd in glob('./data/*'):
        log_name = os.path.basename(pwd)
        parser.parse(log_name)
