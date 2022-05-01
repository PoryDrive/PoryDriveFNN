# PoryDriveFNN
A Feed-Forward Neural Network trained by a classical autonomous AI.

This is forked from [mrbid/porydrive](https://github.com/mrbid/porydrive).

I store all the trained models I liked in [/models](/models) numbered in order of when I trained them, generally higher numbers will be better models than the lower numbers - or at least that is what I think.

I have excluded the dataset files because they are huge, at the time of writing this I use a 1GB `dataset_x.dat`. The best way to produce your own is to use [/multitrain](/multitrain) and then execute [/multitrain/cat.sh](/multitrain/cat.sh) to concatenate them all into one file. Run each `go.sh` file in each respective folder in [/multitrain](/multitrain) for a few hours, then close them all, once the datasets are not being written to anymore run `cat.sh` to do the concatenation. This will concatenate any datasets in the root folder, and then overwrite the dataset in the root folder.
