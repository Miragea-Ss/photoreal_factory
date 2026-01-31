# engine/job_queue.py
from queue import Queue
from threading import Thread
from engine.worker import run_job

JOB_QUEUE = Queue()


def worker_loop():
    while True:
        job = JOB_QUEUE.get()
        if job is None:
            break
        try:
            run_job(**job)
        finally:
            JOB_QUEUE.task_done()


def start_workers(num_workers=1):
    for _ in range(num_workers):
        t = Thread(target=worker_loop, daemon=True)
        t.start()


def submit_job(job: dict):
    JOB_QUEUE.put(job)
