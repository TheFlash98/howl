'''
./extract_loudest_section /home/sarthak/Projects/Augnito/howl/data/hey/common_voice_en_22192412.wav
/home/sarthak/Projects/Augnito/howl/data/hey_loud/common_voice_en_22192412.wav
'''
from pathlib import Path
from subprocess import call

input_wav_files = Path("/home/sarthak/Projects/Augnito/howl/data/hey/")
all_files = list(input_wav_files.glob('*.wav'))
output_wav_files = "/home/sarthak/Projects/Augnito/howl/data/hey_loud"
samples = all_files[10:20]
commands = ['/home/sarthak/Projects/Augnito/extract_loudest_section/gen/bin/extract_loudest_section', 
            '', 
            '/home/sarthak/Projects/Augnito/howl/data/hey_loud']
for sample in all_files:
    commands[1] = str(sample)
    call(commands)

