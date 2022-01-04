import logging
import json
from scipy.io import wavfile
from tqdm import tqdm
path = "/home/sarthak/Projects/Augnito/datasets/hey-ff/speaker-id-split-medium"
logging.info(f"Loading flat dataset from {path}...")
dev_jsonl = path + '/aligned-metadata-dev.jsonl'
test_jsonl = path + '/aligned-metadata-test.jsonl'
train_jsonl = path + '/aligned-metadata-training.jsonl'
metadatalist = []
c = 0
with open(dev_jsonl) as f:
    for json_str in iter(f.readline, ""):
        metadata = json.loads(json_str)
        metadatalist.append(metadata)
for metadata in metadatalist:
    if(metadata['transcription'].split()[0] == 'hey'):
        c += 1
print("Total hey count %i" %c)
with open(test_jsonl) as f:
    for json_str in iter(f.readline, ""):
        metadata = json.loads(json_str)
        metadatalist.append(metadata)
c = 0
for metadata in metadatalist:
    if(metadata['transcription'].split()[0] == 'hey'):
        c += 1
print("Total hey count %i" %c)
with open(train_jsonl) as f:
    for json_str in iter(f.readline, ""):
        metadata = json.loads(json_str)
        metadatalist.append(metadata)
c = 0
for metadata in metadatalist:
    if(metadata['transcription'].split()[0] == 'hey'):
        c += 1
print("Total hey count %i" %c)
audio_path = path + '/audio'
output_path = "/home/sarthak/Projects/Augnito/howl/data/hey_all"
pbar = tqdm(iterable=metadatalist, desc=f"writing clipped data from")
for metadata in pbar:
    if(metadata['transcription'].split()[0] == 'hey'):
        rate, data = wavfile.read(audio_path + '/' + metadata['path'])
        timestamps = metadata['end_timestamps']
        start_timestamp = timestamps[0] - 100 if timestamps[0] > 100 else 0
        end_timestamp = 0
        rep = []
        for timestamp in timestamps:
            if timestamp not in rep:
                rep.append(timestamp)
            if len(rep) == 4:
                end_timestamp = timestamp
        if end_timestamp == 0:
            end_timestamp = timestamps[2]
        start_frame = int(rate * start_timestamp / 1000)
        end_frame = int(rate * end_timestamp / 1000)
        clipped_data = data[start_frame : end_frame]
        print((end_timestamp - start_timestamp))
        if ((end_timestamp - start_timestamp) > 200):
            wavfile.write(output_path + '/' + metadata['path'], rate, clipped_data)


