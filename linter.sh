#!/bin/bash -e
# Adapted from https://github.com/facebookresearch/detectron2/blob/main/dev/linter.sh
{
  black --version | grep -E "22\." > /dev/null
} || {
  echo "Linter requires 'black==22.*' !"
  exit 1
}

ISORT_VERSION=$(isort --version-number)
if [[ "$ISORT_VERSION" != 4.3* ]]; then
  echo "Linter requires isort==4.3.21 !"
  exit 1
fi

set -v

echo "Running isort ..."
isort -y -sp . --atomic

echo "Running black ..."
black -l 100 .

echo "Running flake8 ..."
if [ -x "$(command -v flake8)" ]; then
  flake8 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && arc lint
