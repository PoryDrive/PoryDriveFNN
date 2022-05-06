xterm -e "python3 train.py 384 32 adam 1 models/small" &
xterm -e "python3 train.py 768 64 adam 1 models/big" &

xterm -e "python3 train.py 384 32 sgd 1 models/small" &
xterm -e "python3 train.py 768 64 sgd 1 models/big" &

xterm -e "python3 train.py 384 32 momentum 1 models/small" &
xterm -e "python3 train.py 768 64 momentum 1 models/big" &

xterm -e "python3 train.py 384 32 nesterov 1 models/small" &
xterm -e "python3 train.py 768 64 nesterov 1 models/big" &

xterm -e "python3 train.py 384 32 nadam 1 models/small" &
xterm -e "python3 train.py 768 64 nadam 1 models/big" &

xterm -e "python3 train.py 384 32 adagrad 1 models/small" &
xterm -e "python3 train.py 768 64 adagrad 1 models/big" &