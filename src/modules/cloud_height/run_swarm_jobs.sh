#!/bin/bash
set -euo pipefail

# === CONFIG ===
MAX_CONCURRENT=2
STACK_NAME=cloudheight
IMAGE_NAME=local/cloud-height:latest
IMAGE_LIST="/home/paul/CTH_emulator/products.txt"
POLL_INTERVAL=5
# ==============

# --- Auto-infer volumes from compose file ---
# INPUT_MOUNT="/mnt/raid/data/s2get"
# OUTPUT_MOUNT="/mnt/raid/data/CTH_emulator_dataset"

# Ensure swarm is active
if ! docker info | grep -q "Swarm: active"; then
  echo "Initializing Docker Swarm..."
  docker swarm init --advertise-addr 127.0.0.1
fi

# Build image using docker build
echo "📦 Building image..."
docker build -t "$IMAGE_NAME" .

# Read scenes list
mapfile -t SCENES < "$IMAGE_LIST"
TOTAL=${#SCENES[@]}
echo "🖼️ Found $TOTAL scenes to process."

count_running() {
  docker service ls --filter "name=${STACK_NAME}_" --format "{{.Name}}" | wc -l
}

cleanup_finished() {
  for svc in $(docker service ls --filter "name=${STACK_NAME}_" --format "{{.Name}}"); do
    replicas=$(docker service ls --filter name="$svc" --format "{{.Replicas}}" | cut -d'/' -f1)
    if [[ "$replicas" == "0" ]]; then
      echo "🧹 Removing finished service: $svc"
      docker service rm "$svc" >/dev/null 2>&1 || true
    fi
  done
}

i=0
while [[ $i -lt $TOTAL ]]; do
  cleanup_finished
  running=$(count_running)

  if (( running < MAX_CONCURRENT )); then
    SCENE_DIR="${SCENES[$i]}"
    svc_name="${STACK_NAME}_${i}"

    echo "🚀 Starting service: $svc_name for scene $SCENE_DIR"

    docker service create \
    --name "$svc_name" \
    --restart-condition none \
    --limit-cpu 1 \
    --mount type=bind,src=/mnt/raid/data/CTH_emulator_dataset,dst=/app/CTH_emulator_dataset \
    --mount type=tmpfs,destination=/dev/shm,tmpfs-size=2g \
    "$IMAGE_NAME" \
    bash -c "source ~/.bashrc && conda activate geoenv && python ./src/main.py --config /app/config.yaml --scene_dir /app/s2get/${SCENE_DIR} --plot --save --log"
    ((i++))
  else
    echo "⏳ $running services running, waiting..."
    sleep $POLL_INTERVAL
  fi
done

echo "✅ All jobs submitted. Monitoring for completion..."

while [[ $(count_running) -gt 0 ]]; do
  cleanup_finished
  remaining=$(count_running)
  echo "⌛ $remaining services still running..."
  sleep $POLL_INTERVAL
done

echo "🎉 All scenes processed and cleaned up!"
