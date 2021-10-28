from pathlib import Path
from collections import defaultdict, Counter
from functools import partial
import logging
import json

from tqdm import trange, tqdm

import torch
from torch.utils import data
from torchsummary import summary
from howl.context import InferenceContext
from howl.data.transform import compose, ZmuvTransform, StandardAudioTransform,\
    NoiseTransform, batchify, WakeWordFrameBatchifier, truncate_length, DatasetMixer
from howl.model import RegisteredModel, Workspace
from howl.data.dataset.base import AudioClipMetadata, DatasetType
from howl.data.dataset.dataset import AudioClassificationDataset
from howl.data.dataset import RecursiveNoiseDatasetLoader, Sha256Splitter, WakeWordDataset
from howl.data.dataloader import StandardAudioDataLoaderBuilder
from howl.model.inference import FrameInferenceEngine, SequenceInferenceEngine
from howl.model import ConfusionMatrix
from .create_raw_dataset import print_stats
from howl.settings import SETTINGS


from .args import ArgumentParserBuilder, opt


def main():

    def evaluate_accuracy(data_loader, prefix: str, save: bool = False, mixer: DatasetMixer = None):
        std_transform.eval()
        model.eval()
        pbar = tqdm(data_loader, desc=prefix, leave=True, total=len(data_loader))
        num_corr = 0
        num_tot = 0
        counter = Counter()
        for idx, batch in enumerate(pbar):
            if mixer is not None:
                (batch,) = mixer([batch])
            batch_audio_data = batch.audio_data.to(device)
            scores = model(zmuv_transform(std_transform(batch_audio_data.unsqueeze(0))), None)
            num_tot += scores.size(0)
            labels = torch.tensor([label_map[batch.metadata.transcription]]).to(device)
            num_corr += (scores.max(1)[1] == labels).float().sum().item()
            acc = num_corr / num_tot
            pbar.set_postfix(accuracy=f'{acc:.4}')
        return num_corr / num_tot
    def evaluate_engine(
        dataset: AudioClassificationDataset,
        prefix: str,
        save: bool = False,
        positive_set: bool = False,
        write_errors: bool = True,
        mixer: DatasetMixer = None,
    ):
        std_transform.eval()

        # if use_frame:
        #     engine = FrameInferenceEngine(
        #         int(SETTINGS.training.max_window_size_seconds * 1000),
        #         int(SETTINGS.training.eval_stride_size_seconds * 1000),
        #         model,
        #         zmuv_transform,
        #         ctx,
        #     )
        # else:
        engine = SequenceInferenceEngine(model, zmuv_transform, ctx)
        model.eval()
        conf_matrix = ConfusionMatrix()
        pbar = tqdm(dataset, desc=prefix)
        if write_errors:
            with (ws.path / "errors.tsv").open("a") as f:
                print(prefix, file=f)
        for idx, ex in enumerate(pbar):
            if mixer is not None:
                (ex,) = mixer([ex])
            audio_data = ex.audio_data.to(device)
            engine.reset()
            seq_present = engine.infer(audio_data)
            if seq_present != positive_set and write_errors:
                with (ws.path / "errors.tsv").open("a") as f:
                    f.write(
                        f"{ex.metadata.transcription}\t{int(seq_present)}\t{int(positive_set)}\t{ex.metadata.path}\n"
                    )
            conf_matrix.increment(seq_present, positive_set)
            pbar.set_postfix(dict(mcc=f"{conf_matrix.mcc}", c=f"{conf_matrix}"))

        logging.info(f"{conf_matrix}")
        # if args.eval:
        #     threshold = engine.threshold
        #     with (ws.path / (str(round(threshold, 2)) + "_results.csv")).open("a") as f:
        #         f.write(f"{prefix},{threshold},{conf_matrix.tp},{conf_matrix.tn},{conf_matrix.fp},{conf_matrix.fn}\n")
    def do_evaluate():
        evaluate_engine(ww_test_pos_ds, "Test positive", positive_set=True)
        evaluate_engine(ww_test_neg_ds, "Test negative", positive_set=False)
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
            metadata_list.append(AudioClipMetadata(path=test.absolute(), transcription=test.parent.name, 
                                end_timestamps=[0,0,0,0,0,0,0,0,0,0]))
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
    print(model)
    print("Model's state_dict:")
    # weights_dict = defaultdict()
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #     weights_dict.update({param_tensor : model.state_dict()[param_tensor].tolist()})
    # for key in weights_dict.keys():
    #     print(key, type(weights_dict[key]))
    # with open(str(ws.path / "weights.json"), "w") as outfile:
    #     json.dump(weights_dict, outfile)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    dataset_path = "/home/sarthak/Projects/Augnito/datasets/google-speech-commands-v2"
    sr = settings.audio.sample_rate
    ds_kwargs = dict(sr=sr, mono=settings.audio.use_mono)
    vocab = settings.training.vocab
    label_map = defaultdict(lambda: len(vocab))
    label_map.update({k: idx for idx, k in enumerate(vocab)})
    test_ds = load_data(Path(dataset_path), DatasetType.TEST, **ds_kwargs)
    label_map = defaultdict(lambda: len(SETTINGS.training.vocab))
    label_map.update({k: idx for idx, k in enumerate(SETTINGS.training.vocab)})
    ww_train_ds, ww_dev_ds, ww_test_ds = (
        AudioClassificationDataset(metadata_list=[], label_map=label_map, set_type=DatasetType.TRAINING, **ds_kwargs),
        AudioClassificationDataset(metadata_list=[], label_map=label_map, set_type=DatasetType.DEV, **ds_kwargs),
        AudioClassificationDataset(metadata_list=[], label_map=label_map, set_type=DatasetType.TEST, **ds_kwargs),
    )
    
    #train_ds, dev_ds, test_ds = loader.load_splits(ds_path, **ds_kwargs)
    ww_test_ds.extend(test_ds)
    ww_test_pos_ds = ww_test_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    print_stats("Test pos dataset", ctx, ww_test_pos_ds)
    ww_test_neg_ds = ww_test_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    print_stats("Test neg dataset", ctx, ww_test_neg_ds)
    print(settings.training.use_noise_dataset)
    if settings.training.use_noise_dataset:
        noise_ds = RecursiveNoiseDatasetLoader().load(
            Path(settings.raw_dataset.noise_dataset_path), sr=settings.audio.sample_rate, mono=settings.audio.use_mono
        )
        logging.info(f"Loaded {len(noise_ds.metadata_list)} noise files.")
        noise_ds_train, noise_ds_dev = noise_ds.split(Sha256Splitter(50))
        noise_ds_dev, noise_ds_test = noise_ds_dev.split(Sha256Splitter(50))
        test_mixer = DatasetMixer(noise_ds_test, seed=0, do_replace=False)
        train_mixer = DatasetMixer(noise_ds_train, seed=0, do_replace=False)
        all_mixer = DatasetMixer(noise_ds, seed=0, do_replace=False)
    print(len(test_ds))
    print(evaluate_accuracy(test_ds, f"Noisy test set with {0} noise files"))
    if settings.training.use_noise_dataset:
        print(evaluate_accuracy(test_ds, f"Noisy test set with {len(noise_ds_train.metadata_list)} noise files", mixer=train_mixer))
        print(evaluate_accuracy(test_ds, f"Noisy test set with {len(noise_ds_test.metadata_list)} noise files", mixer=test_mixer))
        print(evaluate_accuracy(test_ds, f"Noisy test set with {len(noise_ds.metadata_list)} noise files", mixer=all_mixer))
    do_evaluate()
if __name__ == "__main__":
    main()
