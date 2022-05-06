xterm -e "python3 train.py 384 32 adam models/small" &
xterm -e "python3 train.py 768 64 adam models/big" &

xterm -e "python3 train.py 384 32 sgd models/small" &
xterm -e "python3 train.py 768 64 sgd models/big" &

xterm -e "python3 train.py 384 32 momentum models/small" &
xterm -e "python3 train.py 768 64 momentum models/big" &

xterm -e "python3 train.py 384 32 nesterov models/small" &
xterm -e "python3 train.py 768 64 nesterov models/big" &

xterm -e "python3 train.py 384 32 nadam models/small" &
xterm -e "python3 train.py 768 64 nadam models/big" &

xterm -e "python3 train.py 384 32 adagrad models/small" &
xterm -e "python3 train.py 768 64 adagrad models/big" &