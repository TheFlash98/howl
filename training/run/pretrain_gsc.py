from collections import defaultdict, Counter
from functools import partial
from pathlib import Path
import logging

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn

from .args import ArgumentParserBuilder, opt
from howl.context import InferenceContext
from howl.data.dataset import GoogleSpeechCommandsDatasetLoader, AudioClassificationDataset
from howl.data.dataloader import StandardAudioDataLoaderBuilder
from howl.data.transform import compose, ZmuvTransform, StandardAudioTransform,\
    NoiseTransform, batchify, TimeshiftTransform, WakeWordFrameBatchifier, truncate_length,\
    AudioSequenceBatchifier, DatasetMixer
from howl.settings import SETTINGS
from howl.model import RegisteredModel, Workspace, ConfusionMatrix
from howl.utils.random import set_seed
from howl.data.dataset.base import AudioClipMetadata, DatasetType
from howl.data.dataset import RecursiveNoiseDatasetLoader, Sha256Splitter, WakeWordDataset
from howl.data.tokenize import WakeWordTokenizer
from .create_raw_dataset import print_stats
from howl.model.inference import FrameInferenceEngine, SequenceInferenceEngine
import warnings

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
        if save and not args.eval:
            writer.add_scalar(f'{prefix}/Metric/acc', acc, epoch_idx)
            ws.increment_model(model, acc / 10)
        elif args.eval:
            tqdm.write(str(counter))
            tqdm.write(str(acc))

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

        if use_frame:
            engine = FrameInferenceEngine(
                int(SETTINGS.training.max_window_size_seconds * 1000),
                int(SETTINGS.training.eval_stride_size_seconds * 1000),
                model,
                zmuv_transform,
                ctx,
            )
        else:
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
        if save and not args.eval:
            writer.add_scalar(f"{prefix}/Metric/tp", conf_matrix.tp, epoch_idx)
            ws.increment_model(model, conf_matrix.tp)
        if args.eval:
            threshold = engine.threshold
            with (ws.path / (str(round(threshold, 2)) + "_results.csv")).open("a") as f:
                f.write(f"{prefix},{threshold},{conf_matrix.tp},{conf_matrix.tn},{conf_matrix.fp},{conf_matrix.fn}\n")
    def do_evaluate():
        evaluate_engine(ww_dev_pos_ds, "Dev positive", positive_set=True)
        evaluate_engine(ww_dev_neg_ds, "Dev negative", positive_set=False)
        evaluate_engine(ww_test_pos_ds, "Test positive", positive_set=True)
        evaluate_engine(ww_test_neg_ds, "Test negative", positive_set=False)
    apb = ArgumentParserBuilder()
    apb.add_options(opt('--model', type=str, choices=RegisteredModel.registered_names(), default='las'),
                    opt('--workspace', type=str, default=str(Path('workspaces') / 'default')),
                    opt('--load-weights', action='store_true'),
                    opt('--eval', action='store_true'))
    args = apb.parser.parse_args()

    ws = Workspace(Path(args.workspace), delete_existing=not args.eval)
    writer = ws.summary_writer
    set_seed(SETTINGS.training.seed)
    loader = GoogleSpeechCommandsDatasetLoader(SETTINGS.training.vocab)
    sr = SETTINGS.audio.sample_rate
    ds_kwargs = dict(sr=sr, mono=SETTINGS.audio.use_mono)
    train_ds, dev_ds, test_ds = loader.load_splits(Path(SETTINGS.dataset.dataset_path), **ds_kwargs)
    use_frame = SETTINGS.training.objective == "frame"
    ctx = InferenceContext(SETTINGS.training.vocab, token_type=SETTINGS.training.token_type, use_blank=not use_frame)
    label_map = defaultdict(lambda: len(SETTINGS.training.vocab))
    label_map.update({k: idx for idx, k in enumerate(SETTINGS.training.vocab)})
    ww_train_ds, ww_dev_ds, ww_test_ds = (
        AudioClassificationDataset(metadata_list=[], label_map=label_map, set_type=DatasetType.TRAINING, **ds_kwargs),
        AudioClassificationDataset(metadata_list=[], label_map=label_map, set_type=DatasetType.DEV, **ds_kwargs),
        AudioClassificationDataset(metadata_list=[], label_map=label_map, set_type=DatasetType.TEST, **ds_kwargs),
    )
    
    #train_ds, dev_ds, test_ds = loader.load_splits(ds_path, **ds_kwargs)
    ww_train_ds.extend(train_ds)
    ww_dev_ds.extend(dev_ds)
    ww_test_ds.extend(test_ds)
    print_stats("Wake word dataset", ctx, ww_train_ds, ww_dev_ds, ww_test_ds)

    ww_dev_pos_ds = ww_dev_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    print_stats("Dev pos dataset", ctx, ww_dev_pos_ds)
    ww_dev_neg_ds = ww_dev_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    print_stats("Dev neg dataset", ctx, ww_dev_neg_ds)
    ww_test_pos_ds = ww_test_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    print_stats("Test pos dataset", ctx, ww_test_pos_ds)
    ww_test_neg_ds = ww_test_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    print_stats("Test neg dataset", ctx, ww_test_neg_ds)
    sr = SETTINGS.audio.sample_rate
    device = torch.device(SETTINGS.training.device)
    std_transform = StandardAudioTransform().to(device).eval()
    zmuv_transform = ZmuvTransform().to(device)
    batchifier = partial(batchify, label_provider=lambda x: x.label)
    truncater = partial(truncate_length, length=int(SETTINGS.training.max_window_size_seconds * sr))
    train_comp = compose(truncater,
                         TimeshiftTransform().train(),
                         NoiseTransform().train(),
                         batchifier)
    prep_dl = StandardAudioDataLoaderBuilder(train_ds, collate_fn=batchifier).build(1)
    prep_dl.shuffle = True
    train_dl = StandardAudioDataLoaderBuilder(train_ds, collate_fn=train_comp).build(SETTINGS.training.batch_size)
    dev_dl = StandardAudioDataLoaderBuilder(dev_ds, collate_fn=compose(truncater, batchifier)).build(SETTINGS.training.batch_size)
    test_dl = StandardAudioDataLoaderBuilder(test_ds, collate_fn=compose(truncater, batchifier)).build(SETTINGS.training.batch_size)

    model = RegisteredModel.find_registered_class(args.model)(len(loader.vocab) + 1).to(device)
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, SETTINGS.training.learning_rate, weight_decay=SETTINGS.training.weight_decay)
    logging.info(f'{sum(p.numel() for p in params)} parameters')
    criterion = nn.CrossEntropyLoss()

    if (ws.path / 'zmuv.pt.bin').exists():
        zmuv_transform.load_state_dict(torch.load(str(ws.path / 'zmuv.pt.bin')))
    else:
        for idx, batch in enumerate(tqdm(prep_dl, desc='Constructing ZMUV')):
            batch.to(device)
            zmuv_transform.update(std_transform(batch.audio_data))
            if idx == 2000:  # TODO: quick debugging, remove later
                break
        logging.info(dict(zmuv_mean=zmuv_transform.mean, zmuv_std=zmuv_transform.std))
    torch.save(zmuv_transform.state_dict(), str(ws.path / 'zmuv.pt.bin'))

    if args.load_weights:
        ws.load_model(model, best=True)
    if args.eval:
        ws.load_model(model, best=True)
        evaluate_accuracy(dev_dl, 'Dev')
        evaluate_accuracy(test_dl, 'Test')
        return

    ws.write_args(args)
    ws.write_settings(SETTINGS)
    writer.add_scalar('Meta/Parameters', sum(p.numel() for p in params))
    dev_acc = 0
    for epoch_idx in trange(SETTINGS.training.num_epochs, position=0, leave=True):
        model.train()
        std_transform.train()
        pbar = tqdm(train_dl,
                    total=len(train_dl),
                    position=1,
                    desc='Training',
                    leave=True)
        for batch in pbar:
            batch.to(device)
            audio_data = zmuv_transform(std_transform(batch.audio_data))
            scores = model(audio_data, std_transform.compute_lengths(batch.lengths))
            optimizer.zero_grad()
            model.zero_grad()
            labels = batch.labels.to(device)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))
            writer.add_scalar('Training/Loss', loss.item(), epoch_idx)

        for group in optimizer.param_groups:
            group['lr'] *= SETTINGS.training.lr_decay
        dev_acc = evaluate_accuracy(dev_dl, 'Dev', save=True)
    test_acc = evaluate_accuracy(test_dl, 'Test')

    print("model: ", args.model)
    print("dev_acc: ", dev_acc)
    print("test_acc: ", test_acc)
    do_evaluate()


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
