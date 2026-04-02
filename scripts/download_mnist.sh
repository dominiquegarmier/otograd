#!/bin/sh
set -eu

target_dir="${1:-data/mnist}"
base_url="https://storage.googleapis.com/cvdf-datasets/mnist"

mkdir -p "$target_dir"

for file in \
  train-images-idx3-ubyte.gz \
  train-labels-idx1-ubyte.gz \
  t10k-images-idx3-ubyte.gz \
  t10k-labels-idx1-ubyte.gz
do
  archive_path="$target_dir/$file"
  raw_path="${archive_path%.gz}"
  tmp_raw_path="$raw_path.tmp"

  if [ ! -f "$raw_path" ]; then
    curl -fL -C - "$base_url/$file" -o "$archive_path"
    gzip -dc "$archive_path" > "$tmp_raw_path"
    mv "$tmp_raw_path" "$raw_path"
  fi
done

printf 'MNIST ready in %s\n' "$target_dir"
