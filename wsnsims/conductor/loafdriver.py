import argparse
import csv
import datetime
import logging
import multiprocessing
import os
import time
import sys
from collections import namedtuple

import numpy as np

from wsnsims.conductor import sim_inputs
from wsnsims.core.environment import Environment
from wsnsims.core.results import Results
from wsnsims.loaf.loaf_sim import LOAF

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RUNS = 50
WAIT_TIME = 100

Parameters = namedtuple('Parameters',
                        ['segment_count', 'mdc_count', 'isdva', 'isdvsd',
                         'radio_range'])


def average_results(results):
    mean_max_delay = np.mean([x.max_delay for x in results])
    mean_balance = np.mean([x.balance for x in results])
    mean_lifetime = np.mean([x.lifetime for x in results])
    mean_energy = np.mean([x.ave_energy for x in results])
    mean_buffer = np.mean([x.max_buffer for x in results])

    result = Results(mean_max_delay, mean_balance, mean_lifetime, mean_energy,
                     mean_buffer)
    return result




def run_loaf(parameters):
    """

     :param parameters:
     :type parameters: Parameters
     :return:
     """

    env = Environment()
    env.segment_count = parameters.segment_count
    env.mdc_count = parameters.mdc_count
    env.isdva = parameters.isdva
    env.isdvsd = parameters.isdvsd
    env.comms_range = parameters.radio_range

    loaf_sim = LOAF(env)

    print(
        "Starting LOAF at {}".format(datetime.datetime.now().isoformat()))
    print("Using {}".format(parameters))
    start = time.time()
    runner = loaf_sim.run()

    results = Results(runner.maximum_communication_delay(),
                      runner.energy_balance(),
                      0.,
                      runner.average_energy(),
                      runner.max_buffer_size())

    print("Finished LOAF in {} seconds".format(time.time() - start))
    return results

def run(parameters):

    loaf_results = []
    

    with multiprocessing.Pool() as pool:

        while len(tocs_results) < RUNS or \
                        len(flower_results) < RUNS or \
                        len(minds_results) < RUNS or \
                        len(focus_results) < RUNS or \
                        len(loaf_results) < RUNS:

            loaf_workers = []

            if len(loaf_results) < RUNS:
                loaf_workers = [
                    pool.apply_async(run_loaf, (parameters,))
                    for _ in range(RUNS - len(loaf_results))]

            for result in loaf_workers:
                try:
                    loaf_results.append(result.get(timeout=WAIT_TIME))
                except Exception:
                    logger.exception('LOAF Exception')
                    continue

    mean_loaf_results = loaf_results[:RUNS]

    return (mean_loaf_results)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-o', type=os.path.realpath, default='results')

    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    start = time.time()
    seed = int(time.time())
    print("Random seed is %s", seed)
    np.random.seed(seed)

    parameters = [Parameters._make(p) for p in sim_inputs.conductor_params]

    headers = ['max_delay', 'balance', 'lifetime', 'ave_energy', 'max_buffer']
    # noinspection PyProtectedMember
    headers += parameters[0]._fields

    results_dir = args.outdir
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    loaf_filepath = os.path.join(results_dir, 'loaf.csv')
    
    loaf_exists = os.path.isfile(loaf_filepath)

    with open(tocs_filepath, 'w', newline='') as tocs_csv, \
            open(loaf_filepath, 'w', newline='') as loaf_csv:

        loaf_writer = csv.DictWriter(loaf_csv, fieldnames=headers)
        
        if not loaf_exists:
            loaf_writer.writeheader()

        for parameter in parameters:
            loaf_res = run(parameter)

            for res in loaf_res:
                loaf_writer.writerow(
                   {**res._asdict(), **parameter._asdict()})
                loaf_csv.flush()

    finish = time.time()
    delta = finish - start
    print("Completed simulation in {} seconds".format(delta))


if __name__ == '__main__':
    main()
