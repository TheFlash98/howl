from pathlib import Path
from collections import defaultdict, Counter
from functools import partial


from tqdm import trange, tqdm

import torch
from torch.utils import data
from torchsummary import summary
from howl.client import HowlClient
from howl.context import InferenceContext
from howl.data.transform import compose, ZmuvTransform, StandardAudioTransform,\
    NoiseTransform, batchify, TimeshiftTransform, truncate_length
from howl.model import RegisteredModel, Workspace
from howl.model.inference import FrameInferenceEngine, SequenceInferenceEngine
from howl.data.dataset.base import AudioClipMetadata, DatasetType
from howl.data.dataset.dataset import AudioClassificationDataset
from howl.data.dataloader import StandardAudioDataLoaderBuilder

from .args import ArgumentParserBuilder, opt


def main():
    def evaluate_accuracy(data_loader, prefix: str, save: bool = False):
        std_transform.eval()
        model.eval()
        pbar = tqdm(data_loader, desc=prefix, leave=True, total=len(data_loader))
        num_corr = 0
        num_tot = 0
        counter = Counter()
        for idx, batch in enumerate(pbar):
            batch = batch.to(device)
            scores = model(zmuv_transform(std_transform(batch.audio_data)),
                           std_transform.compute_lengths(batch.lengths))
            num_tot += scores.size(0)
            labels = batch.labels.to(device)
            counter.update(labels.tolist())
            num_corr += (scores.max(1)[1] == labels).float().sum().item()
            acc = num_corr / num_tot
            pbar.set_postfix(accuracy=f'{acc:.4}')
            # print("HERE FROM EVALUATE ACCURACY")
            # print(scores, labels, num_corr)
        return num_corr / num_tot

    def load_data(path, set_type, **dataset_kwargs):
        file_map = defaultdict(lambda: DatasetType.TRAINING)
        with (path / "testing_list.txt").open() as f:
            file_map.update({k: DatasetType.TEST for k in f.read().split("\n")})
        with (path / "validation_list.txt").open() as f:
            file_map.update({k: DatasetType.DEV for k in f.read().split("\n")})
        all_list  = list(path.glob("*/*.wav"))
        metadata_list = []
        for test in all_list:
            key = str(Path(test.parent.name) / test.name)
            if file_map[key] != set_type:
                continue
            metadata_list.append(AudioClipMetadata(path=test.absolute(), transcription=test.parent.name))
        return AudioClassificationDataset(
            metadata_list=metadata_list, label_map=label_map, set_type=set_type, **dataset_kwargs
        )

            
    apb = ArgumentParserBuilder()
    apb.add_options(
        opt("--model", type=str, choices=RegisteredModel.registered_names(), default="las"),
        opt("--workspace", type=str, default=str(Path("workspaces") / "default")),
    )
    args = apb.parser.parse_args()
    ws = Workspace(Path(args.workspace), delete_existing=False)
    settings = ws.load_settings()

    use_frame = settings.training.objective == "frame"
    ctx = InferenceContext(settings.training.vocab, token_type=settings.training.token_type, use_blank=not use_frame)

    device = torch.device(settings.training.device)
    std_transform = StandardAudioTransform().to(device).eval()
    zmuv_transform = ZmuvTransform().to(device)
    model = RegisteredModel.find_registered_class(args.model)(ctx.num_labels).to(device).eval()
    zmuv_transform.load_state_dict(torch.load(str(ws.path / "zmuv.pt.bin"), map_location='cpu'))

    ws.load_model(model, best=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    dataset_path = "/home/sarthak/Projects/Augnito/datasets/google-speech-commands"
    sr = settings.audio.sample_rate
    ds_kwargs = dict(sr=sr, mono=settings.audio.use_mono)
    vocab = settings.training.vocab
    label_map = defaultdict(lambda: len(vocab))
    label_map.update({k: idx for idx, k in enumerate(vocab)})
    batchifier = partial(batchify, label_provider=lambda x: x.label)
    truncater = partial(truncate_length, length=int(settings.training.max_window_size_seconds * sr))
    test_ds = load_data(Path(dataset_path), DatasetType.TEST, **ds_kwargs)
    prep_dl = StandardAudioDataLoaderBuilder(test_ds, collate_fn=compose(truncater, batchifier)).build(1)
    print(evaluate_accuracy(prep_dl, "sample"))


if __name__ == "__main__":
    main()
