clang main.c glad_gl.c -I inc -Ofast -lglfw -lm -o porydrive
xterm -e "python3 pred.py" &
./porydrive