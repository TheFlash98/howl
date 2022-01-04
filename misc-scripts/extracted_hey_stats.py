from scipy.io import wavfile
from pathlib import Path

hey_extracted_path = Path('/home/sarthak/Projects/Augnito/howl/data/hey/')
all_files = list(hey_extracted_path.glob('*.wav'))
print(len(all_files))
c = 0
c1 = 0
avg_len = 0
for file in all_files:
    rate, data = wavfile.read(file)
    if(len(data) > 16000):
        c += 1
    elif (len(data) < 16000):
        c1 += 1
    else: 
        print(file)
    avg_len += len(data)
print(c, c1, avg_len/len(all_files)) # 616 1275 15439.045406546991