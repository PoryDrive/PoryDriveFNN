#python3 train.py 384 32 adam 0 models/small
#python3 train.py 768 64 adam 0 models/big

#python3 train.py 384 32 sgd 0 models/small
#python3 train.py 768 64 sgd 0 models/big

#python3 train.py 384 32 momentum 0 models/small
#python3 train.py 768 64 momentum 0 models/big

python3 train.py 384 32 nesterov 0 models/small
python3 train.py 768 64 nesterov 0 models/big

python3 train.py 384 32 nadam 0 models/small
python3 train.py 768 64 nadam 0 models/big

python3 train.py 384 32 adagrad 0 models/small
python3 train.py 768 64 adagrad 0 models/big