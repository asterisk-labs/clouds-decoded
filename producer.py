import redis
from rq import Queue
import os
import glob

def main():
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    conn = redis.from_url(redis_url)
    q = Queue("cloud_height", connection=conn,default_timeout=3600)

    processed = list([os.path.splitext(os.path.basename(scene))[0] for scene in glob.glob("/app/out_dir/pcloud/*")])
    for scene in glob.glob("/app/CTH_emulator_dataset/Sentinel-2/MSI/L1C/*/*/*/*.SAFE/"):
        if scene[-1] == "/":
            product_id = os.path.splitext(scene.split("/")[-2])[0]
        else:
            product_id = os.path.splitext(os.path.basename(scene))[0]
        if not product_id  in processed :
            q.enqueue("worker.process_scene", scene)
            print(f"📤 Enqueued job for {scene}")
        else:
            print(f"Ignored job for {scene}, already processed")


if __name__ == "__main__":
    main()