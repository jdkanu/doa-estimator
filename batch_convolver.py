import argparse
import os
from time import time
from multiprocessing import Pool
import random


def convolve(irfile_path, speech_path, output_path):
    pass


def main():
    parser = argparse.ArgumentParser(prog='batch_colvolver',
                                     description="""Batch convolve IR folder with speech folder""")
    parser.add_argument("--irfolder", "-i", help="Directory containing IR files", type=str, required=True)
    parser.add_argument("--speechfolder", "-s", help="Directory containing speech clips", type=str, required=True)
    parser.add_argument("--output", "-o", help="Output directory", type=str, required=True)
    parser.add_argument("--nthreads", "-n", type=int, default=1, help="Number of threads to use")

    args = parser.parse_args()
    irpath = args.irfolder
    speechpath = args.speechfolder
    nthreads = args.nthreads
    outpath = args.output

    if not os.path.exists(irpath):
        print('IR folder {} non-exist, abort!'.format(irpath))
        return

    if not os.path.isfile(speechpath):
        print('Speech folder {} non-exist, abort!'.format(speechpath))
        return

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    irlist = [os.path.join(root, name) for root, dirs, files in os.walk(irpath)
              for name in files if name.endswith(".wav")]
    speechlist = [os.path.join(root, name) for root, dirs, files in os.walk(speechpath)
              for name in files if name.endswith((".wav", ".flac"))]

    ts = time()
    pool = Pool(processes=nthreads)
    res = []
    try:
        # Create a pool to communicate with the worker threads
        for irfile_path in irlist:
            output_path = irfile_path.replace(irpath, outpath)
            new_dir = os.path.dirname(output_path)
            speech_path = random.choice(speechlist)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            pool.apply_async(convolve, args=(irfile_path, speech_path, output_path,))
    except Exception as e:
        print(e)
        pool.close()
    pool.close()
    pool.join()

    print('Took {}'.format(time() - ts))


if __name__ == '__main__':
    main()