CONTAINER_NAME="structure_gen"
IMAGE_NAME="structure_gen"
HOST_PORT=8888
CONTAINER_PORT=8888
HOST_DIR=$(pwd)/..
CONTAINER_DIR=/app

docker run -it --gpus all \
  --name $CONTAINER_NAME \
  -p $HOST_PORT:$CONTAINER_PORT \
  -v $HOST_DIR:$CONTAINER_DIR \
  $IMAGE_NAME