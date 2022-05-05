# PoryDriveFNN
A Feed-Forward Neural Network trained by a classical autonomous AI.

This is forked from [mrbid/porydrive](https://github.com/mrbid/porydrive).

Focused on using Tensorflow Keras on Linux. Training and gameplay has been set to ~144 FPS.

## info

I store all the trained models I liked in [/models](/models) numbered in order of when I trained them, generally higher numbers will be better models than the lower numbers - or at least that is what I think.

I have excluded the dataset files because they are huge, at the time of writing this I use a 1GB `dataset_x.dat`. The best way to produce your own is to use [/multitraingui](/multitraingui) and then execute [/multitrain/cat.sh](/multitrain/cat.sh) to concatenate them all into one file. Run each `go.sh` file in each respective folder in [/multitrain](/multitrain) for a few hours, then close them all, once the datasets are not being written to anymore run `cat.sh` to do the concatenation. `combmain.sh` will concatenate any datasets in the root folder, and then overwrite the dataset in the root folder with the new dataset.

It is much more efficient to make a dataset using [/multitraincli](/multitraincli) which is a multi-process model with file locking so that no concatenation aggregation is needed. You can run more instances than the gui version and specify how many rounds each should play before safely exiting. Do not manually terminate the processes or it could lead to dataset corruption.

## input

#### Input training data _(4-byte float32 per parameter)_
- car normal dir x
- car normal dir y
- (car_pos - porygon_pos) normal dir x
- (car_pos - porygon_pos) normal dir y
- angle between both normal dir's _(Dot product)_
- Euclidean distance between car and porygon

#### Training data targets
- car steering angle
- car speed

## argv

#### porydrive
- First command line MSAA level
- Second command line FPS limit
- Third command line "datalogger mode toggle".

Porydrive at 16 MSAA and 144 FPS: `./porydrive 16 144`<br>
Porydrive at 0 MSAA and 60 FPS: `./porydrive 0 60`<br>
Porydrive in datalogging mode: `./porydrive 0 0 1`

#### porydrivecli
- The first command line parameter is the amount of rounds to execute, `cd multitraincli;./porydrive 8;`, for example, would execute one process for 8 rounds.
- The second command line parameter is the amount of seconds before the process times out, e.g 8 rounds and timeout after 33 seconds, `cd multitraincli;./porydrive 8 33;`.

There is also an example script supplied [multitraincli/go.sh](multitraincli/go.sh). This script is set to execute 128 processes each time because it is best to launch 600+ processes in batches running `go.sh` multiple times otherwise they will all lag and quit at the same time.

## nan

I've created gigabytes of different datasets at this point using both [/multitraingui](/multitraingui) (GUI) and [/multitraincli](/multitraincli) (CLI) and this is what you need to know..

I can generate a dataset using CLI of almost 1GB and usually it will train just fine, but some times and more often when I approach 1GB the training process will start to [NaN](https://en.wikipedia.org/wiki/NaN). But, if I create datasets using GUI I have not noticed this happen yet, so far I have created datasets up to 2GB this way.

The main difference between CLI and GUI dataset aggregation:
- **GUI:** ~1GB dataset takes 10 hours using 10 processes running simultaneously.
- **CLI:** ~1GB dataset takes 10 minutes using 600 processes running simultaneously.

It is much easier to create more CLI processes on one machine than GUI processes because they are much more light weight and require no communication with a Graphics Processing Unit (GPU).

It's just worth keeping this in mind if your dataset starts producing a NaN loss when training with it. I am not completely sure why this is yet, I am checking that the CLI processes do not write NaN floats to the dataset, I check for any write corruption to the extent that I can in real-time which is just that the number of bytes written is correct. It is possible that the high frequency and resource demanding nature of the CLI processes all running at once could be causing bytes to be miss-written to file creating NaN's in the dataset, checking the bytes after writing them could detect this and is an option that comes at a cost to performance but one I will probably be adding.