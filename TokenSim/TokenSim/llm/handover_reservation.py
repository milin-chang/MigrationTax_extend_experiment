"""Reference-counted destination prefix reservations across queue intervals."""


def reserve_destination(req, worker):
    if getattr(req, "handover_d_reservation", None) is not None:
        return
    manager = worker.scheduler.block_manager
    plan = manager.kv_cache_manager.plan_reuse(req)
    blocks = tuple(plan.hit_blocks)
    for block in blocks:
        manager.gpu_allocator.acquire(block)
    req.handover_d_reservation = (manager, blocks)
    req.handover_d_cached_tokens = plan.hit_tokens


def release_destination(req):
    reservation = getattr(req, "handover_d_reservation", None)
    if reservation is None:
        return
    manager, blocks = reservation
    for block in blocks:
        manager.gpu_allocator.free(block)
    req.handover_d_reservation = None


def release_on_completion(env, req):
    yield req.completion_event
    release_destination(req)


def wake_for_capacity(worker):
    from TokenSim.llm.engine.messages import Task, WorkerStatus

    if worker.status == WorkerStatus.WAITING and worker.scheduler.waiting:
        worker.engine.send_task(worker, Task.ADD, [])
