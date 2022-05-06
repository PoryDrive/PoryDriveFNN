# PoryDriveFNN
A Feed-Forward Neural Network trained by a classical autonomous AI.

[![Screenshot of the PoryDrive game](https://raw.githubusercontent.com/mrbid/porydrive/main/screenshot.png)](https://www.youtube.com/watch?v=EoPIzxVX-9E "PoryDrive Game Video")

This is forked from [mrbid/porydrive](https://github.com/mrbid/porydrive).

Focused on using Tensorflow Keras on Linux. Training and gameplay has been set to ~144 updates per second, frame rate can be limited independently.

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

There is also an example script supplied [multitraincli/go.sh](multitraincli/go.sh). This script is set to execute 128 processes each time because it is best to launch 600+ processes in batches running `go.sh` multiple times otherwise they will all lag and quit all at once if all launch at the same time.

## nan

I've created gigabytes of different datasets at this point using both [/multitraingui](/multitraingui) (GUI) and [/multitraincli](/multitraincli) (CLI) and this is what you need to know..

I can generate a dataset using CLI of almost 1GB and usually it will train just fine, but some times and more often when I approach 1GB the training process will start to [NaN](https://en.wikipedia.org/wiki/NaN). But, if I create datasets using GUI I have not noticed this happen yet, so far I have created datasets up to 2GB this way.

The main difference between CLI and GUI dataset aggregation:
- **GUI:** ~1GB dataset takes 10 hours using 10 processes running simultaneously.
- **CLI:** ~1GB dataset takes 10 minutes using 600 processes running simultaneously.

It is much easier to create more CLI processes on one machine than GUI processes because they are much more light weight and require no communication with a Graphics Processing Unit (GPU).

It's just worth keeping this in mind if your dataset starts producing a NaN loss when training with it. I am not completely sure why this is yet, I am checking that the CLI processes do not write NaN floats to the dataset _(before they are written)_, I check for any write corruption to the extent that I can in real-time which is just that the number of bytes written is correct. It is possible that the high frequency and resource demanding nature of the CLI processes all running at once could be causing bytes to be miss-written to file creating NaN's in the dataset, checking the bytes after writing them could detect this and is an option that comes at a cost to performance but one I will probably be adding.

## config
It is possible to tweak the car physics by creating a `config.txt` file in the exec/working directory of the game, here is an example of such config file with the default car physics variables.
```
maxspeed 0.0095
acceleration 0.0025
inertia 0.00015
drag 0.00038
steeringspeed 1.2
steerinertia 233
minsteer 0.32
maxsteer 0.55
steeringtransfer 0.023
steeringtransferinertia 280

ad_min_dstep = 0.01
ad_max_dstep = 0.06
ad_min_speedswitch = 2
ad_maxspeed_reductor = 0.5
```
#### car physics variables
- `maxspeed` - top travel speed of car.
- `acceleration` - increase of speed with respect to time.
- `inertia` - minimum speed before car will move from a stationary state.
- `drag` - loss in speed with respect to time.
- `steeringspeed` - how fast the wheels turn.
- `steerinertia` - how much of the max steering angle is lost as the car increases in speed _(crude steering loss)_.
- `minsteer` - minimum steering angle as scalar _(1 = 180 degree)_ attainable after steering loss caused by `steeringintertia`.
- `maxsteer` - maximum steering angle as scalar _(1 = 180 degree)_ attainable at minimal speeds.
- `steeringtransfer` - how much the wheel rotation angle translates into rotation of the body the wheels are connected to _(the car)_.
- `steeringtransferinertia` - how much the `steeringtransfer` reduces as the car speed increases, this is related to `steerinertia` to give the crude effect of traction loss of the front tires as speed increases and the inability to force the wheels into a wider angle at higher speeds.

#### auto drive variables
- `ad_min_dstep` - minimum delta-distance from the porygon that can trigger a change in steering direction. The delta-distance is the amount of change in the distance since the last update.
- `ad_max_dstep` - maximum delta-distance from the porygon, once this is set any distance above this limit will trigger a change in steering direction.
- `ad_min_speedswitch` - minimum distance from the porygon before the speed of the car begins to linearly reduce as it approaches the porygon.
- `ad_maxspeed_reductor` - the rate at which the speed reduces as the car approaches the porygon with respect to `ad_min_speedswitch`.

## game
Drive around and "collect" Porygon, each time you collect a Porygon a new one will randomly spawn somewhere on the map. A Porygon colliding with a purple cube will cause it to light up blue, this can help you find them. Upon right clicking the mouse you will switch between Ariel and Close views, in the Ariel view it is easier to see which of the purple cubes that the Porygon is colliding with.

#### keyboard
 - `ESCAPE` = Focus/Unfocus Mouse Look
 - `N` = New Game
 - `W` = Drive Forward
 - `A` = Turn Left
 - `S` = Drive Backward
 - `D` = Turn Right
 - `Space` = Breaks
 - `1-5` = Car Physics config selection _(5 loads from file)_

#### keyboard dev
 - `R` = Increment porygon collected count
 - `F` = FPS to console
 - `P` = Player stats to console
 - `O` = Toggle auto drive
 - `I` = Toggle neural drive
 - `L` = Toggle dataset logging

#### mouse
 - `Mouse Button4` = Zoom Snap Close/Ariel
 - `Mouse Click Right` = Zoom Snap Close/Ariel
 - `Middle Scroll` = Zoom in/out
