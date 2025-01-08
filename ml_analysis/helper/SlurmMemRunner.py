import asyncio
import json
import os
from pathlib import Path

from dask_jobqueue import JobQueueCluster
from dask_jobqueue.runner import BaseRunner, Role
from dask_jobqueue.slurm import WorldTooSmallException
from distributed import Scheduler


class SLURMMemRunner(BaseRunner):
    def __init__(self, *args, scheduler_file="scheduler-{job_id}.json", **kwargs):
        try:
            self.proc_id = int(os.environ["SLURM_PROCID"])
            self.world_size = self.n_workers = int(os.environ["SLURM_NTASKS"])
            self.job_id = int(os.environ["SLURM_JOB_ID"])
        except KeyError as e:
            raise RuntimeError(
                "SLURM_PROCID, SLURM_NTASKS, and SLURM_JOB_ID must be present in the environment."
            ) from e
        if not scheduler_file:
            scheduler_file = kwargs.get("scheduler_options", {}).get("scheduler_file")

        if not scheduler_file:
            raise RuntimeError(
                "scheduler_file must be specified in either the "
                "scheduler_options or as keyword argument to SlurmRunner."
            )

        # Encourage filename uniqueness by inserting the job ID
        scheduler_file = scheduler_file.format(job_id=self.job_id)
        scheduler_file = Path(scheduler_file)

        if isinstance(kwargs.get("scheduler_options"), dict):
            kwargs["scheduler_options"]["scheduler_file"] = scheduler_file
        else:
            kwargs["scheduler_options"] = {"scheduler_file": scheduler_file}
        if isinstance(kwargs.get("worker_options"), dict):
            kwargs["worker_options"]["scheduler_file"] = scheduler_file
            kwargs["worker_options"]["local_directory"] = os.environ["TMPDIR"]+"/"+os.environ["SLURM_PROCID"]
            Path(kwargs["worker_options"]["local_directory"]).mkdir(parents=True, exist_ok=True)
        else:
            kwargs["worker_options"] = {"scheduler_file": scheduler_file}

        self.scheduler_file = scheduler_file

        super().__init__(*args, **kwargs)

    async def get_role(self) -> str:
        if self.scheduler and self.client and self.world_size < 3:
            raise WorldTooSmallException(
                f"Not enough Slurm tasks to start cluster, found {self.world_size}, "
                "needs at least 3, one each for the scheduler, client and a worker."
            )
        elif self.scheduler and self.world_size < 2:
            raise WorldTooSmallException(
                f"Not enough Slurm tasks to start cluster, found {self.world_size}, "
                "needs at least 2, one each for the scheduler and a worker."
            )
        self.n_workers -= int(self.scheduler) + int(self.client)
        if self.proc_id == 0 and self.scheduler:
            return Role.scheduler
        elif self.proc_id == 1 and self.client:
            return Role.client
        else:
            return Role.worker

    async def set_scheduler_address(self, scheduler: Scheduler) -> None:
        return

    async def get_scheduler_address(self) -> str:
        while not self.scheduler_file or not self.scheduler_file.exists():
            await asyncio.sleep(0.2)
        cfg = json.loads(self.scheduler_file.read_text())
        return cfg["address"]

    async def get_worker_name(self) -> str:
        return self.proc_id