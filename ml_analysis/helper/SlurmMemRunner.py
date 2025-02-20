import asyncio
import json
import os
import signal
import traceback
from contextlib import suppress
from json import JSONDecodeError
from pathlib import Path

from dask_jobqueue.runner import BaseRunner, Role
from distributed import Scheduler, Status, rpc
from distributed.comm import CommClosedError


class SLURMMemRunner(BaseRunner):
    def __init__(self, *args, in_proc_id=-1, fold_no=-1, scheduler_file="scheduler-{job_id}.json", **kwargs):
        self.in_proc_id = in_proc_id
        self.fold_no = fold_no
        try:
            self.proc_id = int(os.environ["SLURM_PROCID"])
            self.job_id = int(os.environ["SLURM_JOB_ID"])
        except KeyError as e:
            raise RuntimeError(
                "SLURM_PROCID, SLURM_NTASKS, and SLURM_JOB_ID must be present in the environment."
            ) from e
        if in_proc_id < 0 :
            raise RuntimeError("In-Proc ID must be positive!")
        if fold_no < 0 :
            raise RuntimeError("Fold No must be positive!")
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
            kwargs["worker_options"]["local_directory"] = os.environ["TMPDIR"]+"/"+os.environ["SLURM_PROCID"]+f"_{self.in_proc_id}+{self.fold_no}"
            Path(kwargs["worker_options"]["local_directory"]).mkdir(parents=True, exist_ok=True)
        else:
            kwargs["worker_options"] = {"scheduler_file": scheduler_file}

        self.scheduler_file = scheduler_file

        super().__init__(*args, **kwargs)

    async def get_role(self) -> str:
        if self.proc_id == 0 and self.in_proc_id == 0 and self.scheduler:
            return Role.scheduler
        elif self.proc_id == 0 and self.in_proc_id == 1 and self.client:
            return Role.client
        else:
            return Role.worker

    async def set_scheduler_address(self, scheduler: Scheduler) -> None:
        return

    async def get_scheduler_address(self) -> str:
        while not self.scheduler_file or not self.scheduler_file.exists():
            await asyncio.sleep(0.2)
        while True:
            try:
                cfg = json.loads(self.scheduler_file.read_text())
                return cfg["address"]
            except JSONDecodeError:
                print(traceback.format_exc())
                await asyncio.sleep(0.2)

    async def get_worker_name(self) -> str:
        return f"{self.proc_id}_{self.in_proc_id}"

    async def _start(self) -> None:
        self.role = await self.get_role()
        if self.role == Role.scheduler:
            await self.start_scheduler()
            os.kill(
                os.getpid(), signal.SIGTERM
            )  # Shutdown with a signal to give the event loop time to close
            await asyncio.sleep(15)
            os.kill(
                os.getpid(), signal.SIGKILL
            )
        elif self.role == Role.worker:
            await self.start_worker()
            os.kill(
                os.getpid(), signal.SIGTERM
            )  # Shutdown with a signal to give the event loop time to close
            await asyncio.sleep(15)
            os.kill(
                os.getpid(), signal.SIGKILL
            )
        elif self.role == Role.client:
            self.scheduler_address = await self.get_scheduler_address()
            if self.scheduler_address:
                self.scheduler_comm = rpc(self.scheduler_address)
            await self.before_client_start()
        self.status = Status.running

    async def _close(self) -> None:
        if self.status == Status.running:
            if self.scheduler_comm:
                with suppress(CommClosedError):
                    await self.scheduler_comm.terminate()
            self.status = Status.closed

