import os
import numpy as np
from pydub import AudioSegment
from feature_extractor import extract_feature
import argparse
from time import time
from multiprocessing import Pool
from shutil import copy2


def add_noise(signal_path, noise_path, SNR, save_path):
    speech = AudioSegment.from_wav(signal_path)
    noise = AudioSegment.from_wav(noise_path)
    speech_dB = speech.dBFS
    noise_dB = noise.dBFS

    gain = (speech_dB - noise_dB) - SNR
    noise = noise.apply_gain(gain)
    noisy_speech = speech.overlay(noise, )
    noisy_speech.export(save_path, format='wav')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='augmentation',
                                     description="""Script to augment dataset""")
    parser.add_argument("--input", "-i", default="data", help="Directory where data and labels are", type=str)
    parser.add_argument("--noise", "-no", required=True, help="Directory where the noise file is", type=str)
    parser.add_argument("--SNR", "-s", required=True, help="SNR ratio for adding noise", type=float)
    parser.add_argument("--output", "-o", default=".", help="Directory to write results", type=str)
    parser.add_argument("--nthreads", "-n", type=int, default=1, help="Number of threads to use")

    args = parser.parse_args()

    signaldir = args.input
    noisepath = args.noise
    SNR = args.SNR
    nthreads = args.nthreads
    outpath = args.output

    if not os.path.exists(signaldir):
        os.error('input directory {} non-exist, abort!'.format(signaldir))

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    copy2(os.path.join(signaldir, 'labels.csv'), outpath)

    ts = time()
    # Convert all houses
    try:
        # # Create a pool to communicate with the worker threads
        pool = Pool(processes=nthreads)
        for subdir, dirs, files in os.walk(signaldir):
            for f in files:
                if f.endswith('.wav'):
                    filename = os.path.join(subdir, f)
                    savepath = filename.replace(signaldir, outpath)
                    pool.apply_async(add_noise, args=(filename, noisepath, SNR, savepath))
    except Exception as e:
        print(str(e))
        pool.close()
    pool.close()
    pool.join()

    print('Took {}'.format(time() - ts))
