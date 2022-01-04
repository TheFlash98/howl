import logging
import json
from scipy.io import wavfile
from tqdm import tqdm
path = "/home/sarthak/Projects/Augnito/datasets/hey-ff/speaker-id-split-medium"
logging.info(f"Loading flat dataset from {path}...")
short_jsonl = path + '/short.jsonl'
dev_jsonl = path + '/align-metadata-dev.jsonl'
test_jsonl = path + '/align-metadata-test.jsonl'
train_jsonl = path + '/align-metadata-training.jsonl'
metadatalist = []
with open(short_jsonl) as f:
    for json_str in iter(f.readline, ""):
        metadata = json.loads(json_str)
        metadatalist.append(metadata)
c = 0
for metadata in metadatalist:
    if(metadata['transcription']['transcription'] == 'hey'):
        c += 1
print("Total hey count %i" %c)
audio_path = path + '/audio'
output_path = "/home/sarthak/Projects/Augnito/howl/data/hey"
pbar = tqdm(iterable=metadatalist, desc=f"writing clipped data from {short_jsonl}")
for metadata in pbar:
    if(metadata['transcription']['transcription'] == 'hey'):
        rate, data = wavfile.read(audio_path + '/' + metadata['path'])
        timestamps = metadata['transcription']['end_timestamps']
        start_timestamp = timestamps[0] - 100 if timestamps[0] > 100 else 0
        end_timestamp = timestamps[2]
        start_frame = int(rate * start_timestamp / 1000)
        end_frame = int(rate * end_timestamp / 1000)
        clipped_data = data[start_frame : end_frame]
        wavfile.write(output_path + '/' + metadata['path'], rate, clipped_data)


