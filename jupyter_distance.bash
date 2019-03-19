
export PORT=8888
export REMOTEMACHINE=c51-11

ssh $REMOTEMACHINE << EOF
 export PATH=/cal/softs/anaconda/anaconda3/bin:$PATH
 source activate /cal/homes/ladjal/.conda/envs/tf_C51
 jupyter notebook --no-browser --port=$PORT
EOF


 
