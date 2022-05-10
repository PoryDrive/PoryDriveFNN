# PoryDriveFNN
A Feed-Forward Neural Network trained by a classical autonomous AI.

[![Screenshot of the PoryDrive game](https://raw.githubusercontent.com/mrbid/porydrive/main/screenshot.png)](https://www.youtube.com/watch?v=EoPIzxVX-9E "PoryDrive Game Video")

This is forked from [mrbid/porydrive](https://github.com/mrbid/porydrive).

Focused on using Tensorflow Keras on Linux. Training and gameplay has been set to ~144 updates per second, frame rate can be limited independently.

## how

Create a dataset using [/multicapturecli](/multicapturecli) _(I've already trained so many models in [PoryDriveFNN_models](https://github.com/PoryDrive/PoryDriveFNN_models) that you don't really need to aggregate a dataset to train your own)_.

- [`shuff.py`](shuff.py) - _(optional)_ shuffle the dataset & zeros any [NaN's](https://en.wikipedia.org/wiki/NaN).
- [`train.py`](train.py) - train a model from the dataset `python3 train.py <layers 0-4> <units per layer> <batches> <optimiser: adam,nesterov,etc> <cpu only 1/0>`
- [`pred.py`](pred.py) - run the predictor daemon so that the `./porydrive` program can communicate with the Tensorflow Keras backend `python3 pred.py <model_path>`.

Then run `./porydrive` and press `I` to enter Neural Drive mode.

If you use a pre-trained model then you just need to start at the `pred.py` step.

It is possible to train datasets using [/multicapturegui](/multicapturegui) or `./porydrive` _(press `O` to enable Auto Drive and then `L` to enable the datalogger)_ but these methods are now legacy and only suitable for smaller datasets. You could expect a 1GB dataset from these in the time [/multicapturecli](/multicapturecli) generated 500GB. [/multicapturecli](/multicapturecli) will generate a very small percentage of corruption in the dataset it produces such as NaN's and that is why it is important to use `shuff.py` before you train using a dataset from it, as where the other former methods are much less likely to produe datasets with corruption.

## info

Generally FNN networks under 1024 `layer_units` will train faster on the CPU than the GPU and vice-versa for networks of more than 1024 `layer_units`. Remove the `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` line (14) from [pred.py](pred.py) if you intend to use larger networks.

Trained models are kept in a seperate repository; [PoryDriveFNN_models](https://github.com/PoryDrive/PoryDriveFNN_models). Nesterov optimiser and tanh activation functions seem to work best for this purpose. An example of a small network trained with Nesterov would be `python3 train.py 4 384 32 nesterov 1` or `python3 train2.py 16 32 32 tanh nesterov 1`.

A 1GB dataset to train a 4 layer network with 384-869 _(384 is my preferred)_ units per layer _(top-down so technically less as each layer reduces the units by a factor of two.)_ and batches set at 32 _(because the sample rate is 144 times per second that is roughly one batch per just less than one quater of a second of data. You could push this to 64 for faster training but I always find 32 batches produces a slightly better network)_ these networks are relatively fast to train, can be trained concurrently on multiple CPU's and are what I consider to be the "sweet spot". Trained with Nesterov accelerated gradient as mentioned above.

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
- The first command line parameter is the amount of rounds to execute, `cd multicapturecli;./porydrive 8;`, for example, would execute one process for 8 rounds.
- The second command line parameter is the amount of seconds before the process times out, e.g 8 rounds and timeout after 33 seconds, `cd multicapturecli;./porydrive 8 33;`.
- The third command line parameter is the minimum score to log, if I set this to 0.9 it will only save datasets 0.9 and 1.0 to file; `cd multicapturecli;./porydrive 8 33 0.9;`

There is also an example script supplied [multicapturecli/go.sh](multicapturecli/go.sh). This script is set to execute a number of processes, it is best to stagger the launch of processes in batches running `go.sh` multiple times otherwise they may all lag and quit all at once if all launched at the same time.

#### train.py
`python3 train.py <layers 0-4> <layer units> <batches> <optimiser: adam,nesterov,etc> <cpu only 1/0>`

#### train2.py
_train2.py targeted at SELU style networks using many layers with few units._<br>
`python3 train.py <layers> <layer units> <batches> <activator> <optimiser> <cpu only 1/0>`<br>

#### pred.py
`python3 pred.py <model_path>`

## datasets

The main difference between CLI and GUI dataset aggregation:
- **GUI:** ~1GB dataset takes 10 hours using 10 processes running simultaneously.
- **CLI:** ~1GB dataset takes 10 minutes using 600 processes running simultaneously.

It is much easier to create more CLI processes on one machine than GUI processes because they are much more light weight and require no communication with a Graphics Processing Unit (GPU).

CLI generates scored datasets, the higher the score the better performing the dataset.

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
steering_deadzone = 0.013
steeringtransfer 0.023
steeringtransferinertia 280
suspension_pitch = 3
suspension_pitch_limit = 0.03
suspension_roll = 30
suspension_roll_limit = 9

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
- `steering_deadzone` - minimum angle of steering considered the deadzone or cutoff, within this angle the steering angle will always be forced to zero.
- `steeringtransfer` - how much the wheel rotation angle translates into rotation of the body the wheels are connected to _(the car)_.
- `steeringtransferinertia` - how much the `steeringtransfer` reduces as the car speed increases, this is related to `steerinertia` to give the crude effect of traction loss of the front tires as speed increases and the inability to force the wheels into a wider angle at higher speeds.
- `suspension_pitch` - suspension pitch increment scalar
- `suspension_pitch_limit` - max & min pitch limit
- `suspension_roll` - suspension roll increment scalar
- `suspension_roll_limit` - max & min roll limit

#### auto drive variables
- `ad_min_dstep` - minimum delta-distance from the porygon that can trigger a change in steering direction. The delta-distance is the amount of change in the distance since the last update.
- `ad_max_dstep` - maximum delta-distance from the porygon, once this is set any distance above this limit will trigger a change in steering direction.
- `ad_min_speedswitch` - minimum distance from the porygon before the speed of the car begins to linearly reduce as it approaches the porygon.
- `ad_maxspeed_reductor` - the rate at which the speed reduces as the car approaches the porygon with respect to `ad_min_speedswitch`.

## game
Drive around and "collect" Porygon, each time a Porygon is collected a new one will randomly spawn somewhere on the map. A Porygon colliding with a purple cube will cause it to light up blue, this can help find them. Upon right clicking the mouse the view will switch between Ariel and Close views, in the Ariel view it is easier to see which of the purple cubes that the Porygon is colliding with.

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
