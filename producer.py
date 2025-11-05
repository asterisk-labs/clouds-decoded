import redis
from rq import Queue
import os
import glob

def main():
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    conn = redis.from_url(redis_url)
    q = Queue("cloud_height", connection=conn,default_timeout=3600)

    for scene in glob.glob("/app/CTH_emulator_dataset/Sentinel-2/MSI/L1C/*/*/*/*.SAFE/"):
        q.enqueue("worker.process_scene", scene)
        print(f"📤 Enqueued job for {scene}")

if __name__ == "__main__":
    main()
