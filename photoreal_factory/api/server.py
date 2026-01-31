# api/server.py
from fastapi import FastAPI
from engine.job_queue import submit_job, start_workers

app = FastAPI()
start_workers(num_workers=1)


@app.post("/submit")
def submit(job: dict):
    submit_job(job)
    return {"status": "queued"}
