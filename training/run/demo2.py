from pathlib import Path

import torch
from howl.client import HowlClient2
from howl.context import InferenceContext
from howl.data.transform import ZmuvTransform
from howl.model import RegisteredModel, Workspace
from howl.model.inference import FrameInferenceEngine, SequenceInferenceEngine
from howl.model.inference2 import FrameInferenceEngine2, SequenceInferenceEngine2

from .args import ArgumentParserBuilder, opt


def main():
    apb = ArgumentParserBuilder()
    apb.add_options(
        opt("--model", type=str, choices=RegisteredModel.registered_names(), default="las"),
        opt("--workspace", type=str, default=str(Path("workspaces") / "default")),
        opt("--workspace2", type=str, default=str(Path("workspaces") / "default"))
    )
    args = apb.parser.parse_args()
    ws = Workspace(Path(args.workspace), delete_existing=False)
    settings = ws.load_settings()
    print(settings.training.vocab)
    use_frame = settings.training.objective == "frame"
    ctx = InferenceContext(settings.training.vocab, token_type=settings.training.token_type, use_blank=not use_frame)
    ws2 = Workspace(Path(args.workspace2), delete_existing=False)
    settings2 = ws2.load_settings_2()
    print(settings.training.vocab, settings2.training.vocab)
    
    ctx2 = InferenceContext(settings2.training.vocab, token_type=settings2.training.token_type, use_blank=not use_frame)

    device = torch.device(settings.training.device)
    zmuv_transform = ZmuvTransform().to(device)
    model = RegisteredModel.find_registered_class(args.model)(ctx.num_labels).to(device).eval()
    zmuv_transform.load_state_dict(torch.load(str(ws.path / "zmuv.pt.bin"), map_location='cpu'))


    ws.load_model(model, best=True)
    model.streaming()

    device2 = torch.device(settings2.training.device)
    zmuv_transform2 = ZmuvTransform().to(device2)
    model2 = RegisteredModel.find_registered_class(args.model)(ctx2.num_labels).to(device).eval()
    zmuv_transform2.load_state_dict(torch.load(str(ws2.path / "zmuv.pt.bin"), map_location='cpu'))
    ws2.load_model(model2, best=True)
    model2.streaming()
    if use_frame:
        engine = FrameInferenceEngine(
            int(settings.training.max_window_size_seconds * 1000),
            int(settings.training.eval_stride_size_seconds * 1000),
            model,
            zmuv_transform,
            ctx,
        )

        engine2 = FrameInferenceEngine2(
            int(settings2.training.max_window_size_seconds * 1000),
            int(settings2.training.eval_stride_size_seconds * 1000),
            model2,
            zmuv_transform2,
            ctx2,
        )
    else:
        engine = SequenceInferenceEngine(model, zmuv_transform, ctx)
        engine2  = SequenceInferenceEngine2(model2, zmuv_transform2, ctx2)
    # chunk size of 1000 is required for GSC dataset 
    
    client = HowlClient2(engine, engine2, ctx, ctx2, 0, int(settings.training.max_window_size_seconds * 1000))
    client.start().join()


if __name__ == "__main__":
    main()
