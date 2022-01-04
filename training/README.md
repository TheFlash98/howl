# Training New Models

Before training a model using howl, a lot of environment variable need to be set. These include the path to your dataset, the words you want the model to detect and others. All these relevant variables can be found in [env/gsc_train.env](https://github.com/TheFlash98/howl/blob/master/envs/gsc_train.env). Here are the ones that will need to be changes before training

- `MAX_WINDOW_SIZE_SECONDS` - is set to 1 seconds by default. It's the size of an audio chunk sent to the model of inferening / training. GSC dataset has wav files of length 1 seconds, this will vary according to your input wav file length.
- `USE_NOISE_DATASET` - set `True` is noised is needed during training
- `INFERENCE_SEQUENCE` - a list of format `[0, 1, ... n - 1]` where n is the number of words the model will learn
- `VOCAB` - list of words the model will learn example `["yes", "no"]`
- `DATASET_PATH` - the path to the gsc dataset that will be used for training
- `USE_NOISE_DATASET` - path to the noise wav files to be added during training

Once all the relevant environment variables are set, simply source the file as `source env/gsc_train.env`. The device to run the training on can also be selected through by altering a single line in [howl/settings.py](https://github.com/TheFlash98/howl/blob/master/howl/settings.py#L41). Once everything is set run - 

`python -m training.run.pretrain_gsc_noisy --model res8 --workspace workspaces/<workspace-name>` 

The `<workspace-name>` can be anything arbitrary. A folder under the `workspaces` directory will be created with that name which will contain the final trained models and other related files. To test out your model run it as a demo using the following command and say your wake word

`python -m training.run.demo --model res8 --workspace workspaces/<workspace-name>`