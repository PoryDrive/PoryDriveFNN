# PoryDriveFNN
A Feed-Forward Neural Network trained by a classical autonomous AI.

This is forked from [mrbid/porydrive](https://github.com/mrbid/porydrive).

Focused on using Tensorflow Keras on Linux. Training and gameplay has been capped to ~144 FPS.

I store all the trained models I liked in [/models](/models) numbered in order of when I trained them, generally higher numbers will be better models than the lower numbers - or at least that is what I think.

I have excluded the dataset files because they are huge, at the time of writing this I use a 1GB `dataset_x.dat`. The best way to produce your own is to use [/multitraingui](/multitraingui) and then execute [/multitrain/cat.sh](/multitrain/cat.sh) to concatenate them all into one file. Run each `go.sh` file in each respective folder in [/multitrain](/multitrain) for a few hours, then close them all, once the datasets are not being written to anymore run `cat.sh` to do the concatenation. `combmain.sh` will concatenate any datasets in the root folder, and then overwrite the dataset in the root folder with the new dataset.

It is much more efficient to make a dataset using [/multitraincli](/multitraincli) which is a multi-process model with file locking so that no concatenation aggregation is needed. You can run more instances than the gui version and specify how many rounds each should play before safely exiting. Do not manually terminate the processes or it could lead to dataset corruption. The amount of rounds to execute is the only command line parameter, `cd multitraincli;./porydrive 32;`, for example, would execute one process for 32 rounds. There is also an example script supplied [multitraincli/go.sh](multitraincli/go.sh) which will execute 16 processes for 512 rounds.