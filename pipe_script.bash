
export PORT=8888
export REMOTEMACHINE=c51-11

ssh -N -L localhost:$PORT:localhost:$PORT $REMOTEMACHINE
