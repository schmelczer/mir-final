#!/bin/bash

set -e

if [ -z "$1" ]
then
    echo "Provide the path to a Jupyter notebook: $0 example.ipynb"
    exit -1
fi

if ! command -v tmux -V &> /dev/null
then
    echo "tmux could not be found, installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install tmux
    else
        sudo apt install -y tmux
    fi
fi

if ! command -v /usr/bin/time --version &> /dev/null
then
    echo "time could not be found, installing..."
    sudo apt install -y time
fi

VENV_PATH=$(dirname "$0")
# VENV_PATH="$VENV_PATH/.env/bin/activate" 

TUTORIAL="Press CTRL+B D to detach from the session"
SESSION_NAME="$( LC_ALL=C eval printf '%s' "${1//[^a-zA-Z0-9]/_}" )"
COMMAND="echo $TUTORIAL && /usr/bin/time /bin/python3 -m nbconvert --to notebook --execute --show-input $1 --output rendered-$1 2>&1 | tee $SESSION_NAME.log"
tmux new-session -s "$SESSION_NAME" "$COMMAND"
