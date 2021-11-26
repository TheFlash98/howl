
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path

import sys, getopt
'''
python extract_words.py -i /home/sarthak/Projects/Augnito/datasets/augnito_india_kws_20211126_20211126 \
    -o /home/sarthak/Projects/Augnito/datasets/extract_word_aug_dataset \
    -w start
'''
def main(argv):
    input_dir = ''
    output_dir = ''
    word = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:w:",["idir=","odir=","word="])
    except getopt.GetoptError:
        print('extract_words.py -i <inputdir> -o <outputdir> -w <word>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputdir> -o <outputdir> -w <word>')
            sys.exit()
        elif opt in ("-i", "--idir"):
            input_dir = arg
        elif opt in ("-o", "--odir"):
            output_dir = arg
        elif opt in ("-w", "--word"):
            word = arg
    print('\n')
    print('Input directory:', input_dir)
    print('Output directory:', output_dir)
    print('Word:', word)

    input_wav_files = list(Path(input_dir).glob('*.wav'))
    print('Total number of wav files', len(input_wav_files))
    print('\n')
    pbar = tqdm(iterable=input_wav_files, desc=f"extracting word {word}")
    for wav_file in pbar:
        with open(str(wav_file.parent) + '/' + str(wav_file.stem) + '.hyp') as f:
            for line in iter(f.readline, ""):
                line = line.split()
                if word in line:
                    rate, data = wavfile.read(wav_file)
                    start_timestamp = float(line[2])
                    end_timestamp = float(line[2]) + float(line[3])
                    start_frame = int(rate * start_timestamp)
                    end_frame = int(rate * end_timestamp)
                    clipped_data = data[start_frame : end_frame]
                    output_wav_file = output_dir + '/' + str(wav_file.stem) + "_" + str(start_timestamp) + "_" + str(end_timestamp) + '.wav'
                    wavfile.write(output_wav_file, rate, clipped_data)

if __name__ == "__main__":
   main(sys.argv[1:])