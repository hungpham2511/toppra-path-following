DIR="$(dirname $(readlink -f $0))"


echo "Found source directory at $DIR"
echo "Adding path to PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:$DIR
