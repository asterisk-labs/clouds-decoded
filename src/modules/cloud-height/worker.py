import os
import subprocess
from rq import Worker, Queue, Retry
import redis

def process_scene(scene_dir):
    print(f"🚀 Processing {scene_dir}")
    cmd = (
        f"bash -c 'source ~/.bashrc && "
        f"conda activate geoenv && "
        f"python ./src/main.py "
        f"--config /app/config.yaml "
        f"--scene_dir {scene_dir} "
        f"--plot --save --log'"
    )
    subprocess.run(cmd, shell=True, check=True)
    print(f"✅ Finished {scene_dir}")

def main():
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    conn = redis.from_url(redis_url)

    # Directly pass the connection to the Queue and Worker
    q = Queue("cloud_height", connection=conn,default_timeout=3600)
    worker = Worker([q], connection=conn)
    
    worker.work(with_scheduler=False)

if __name__ == "__main__":
    main()
