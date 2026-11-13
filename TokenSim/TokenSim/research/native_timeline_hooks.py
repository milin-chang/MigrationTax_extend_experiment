"""Install per-instance observers around existing synchronous/native yields."""

from TokenSim.llm.engine.messages import Task


def install(timeline):
    engine = timeline.engine
    for worker in engine.request_workers:
        if not engine._is_schedulable_worker(worker):
            continue
        original_put = worker.msg_queue.put

        def put(message, *, original=original_put, worker=worker):
            result = original(message)
            if message.task == Task.ADD:
                timeline.queued(worker, message.requests or [])
            return result

        worker.msg_queue.put = put
        original_add = worker.scheduler.add_requests

        def add(requests, *, original=original_add, worker=worker):
            result = original(requests)
            timeline.queued(worker, requests, scheduler=True)
            return result

        worker.scheduler.add_requests = add
        original_prepare = worker.prepare_step

        def prepare(*, original=original_prepare, worker=worker):
            started = worker.env.now
            requests, latency = original()
            timeline.prepared(worker, requests, latency, started)
            return requests, latency

        worker.prepare_step = prepare
        original_complete = worker.complete_step

        def complete(requests, latency, *, original=original_complete, worker=worker):
            before = {req.id: req.generation_idx for req in requests}
            result = original(requests, latency)
            timeline.advanced(worker, requests, before)
            return result

        worker.complete_step = complete
        original_finalize = worker.finalize_connector_step

        def finalize(*, original=original_finalize, worker=worker):
            result = original()
            timeline.finish_step(worker, "load_only")
            return result

        worker.finalize_connector_step = finalize

    original_compute = engine.run_complete_compute

    def compute(worker_id, latency):
        started = engine.env.now
        timeline.compute_start(worker_id)
        try:
            result = yield from original_compute(worker_id, latency)
        except BaseException:
            timeline.finish_step(engine.workers[worker_id], "failed_compute")
            raise
        timeline.compute_end(worker_id, started)
        return result

    engine.run_complete_compute = compute
    migration = getattr(engine, "native_migration", None)
    if migration is not None:
        original_log = migration.log

        def log(kind, req=None, **values):
            result = original_log(kind, req, **values)
            timeline.transfer(kind, req, values)
            return result

        migration.log = log
