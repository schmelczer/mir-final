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

VENV_PATH=$(dirname "$0")
VENV_PATH="$VENV_PATH/.env/bin/activate"

TUTORIAL="Press CTRL+B D to detach from the session"
SCRIPT=`python3 -m nbconvert --to script $1 --stdout`
SESSION_NAME="$( LC_ALL=C eval printf '%s' "${1//[^a-zA-Z0-9]/_}" )"
COMMAND="echo $TUTORIAL && source $VENV_PATH && (python3 - 2>&1 <<'EOF'"$'\n'"$SCRIPT"$'\nEOF\n) | tee '"$SESSION_NAME.log"

tmux new-session -s "$SESSION_NAME" "$COMMAND"
