#!/usr/bin/env python3
"""Submit one ordinary Ray task per evaluation experiment.

Each experiment independently requests ``gpus_per_exp`` GPUs and
``cpus_per_exp`` CPUs. Ray decides which node runs it and keeps it pending until
one node has enough free resources. No placement groups, node affinity, manual
node discovery, or coordinator-side slot accounting are used.

Multiple Ray Jobs may run this coordinator at the same time. Their experiment
tasks share the same cluster resource pool and start as resources become free.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
from ray.exceptions import RayTaskError

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


# ============================================================================
# Process and output-lock cleanup
# ============================================================================


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def remove_lock(lock_dir: Path) -> None:
    try:
        owner_file = lock_dir / "owner.json"
        if owner_file.exists():
            owner_file.unlink()
        lock_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        # Do not recursively remove an unexpected non-empty directory.
        pass


def kill_process_tree(root_pid: int, pgid: int, grace_seconds: float) -> None:
    """Terminate an evaluator process group and any detached descendants."""
    processes: list[Any] = []

    if psutil is not None and root_pid > 0:
        try:
            root = psutil.Process(root_pid)
            processes = root.children(recursive=True) + [root]
            for proc in reversed(processes):
                try:
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            processes = []

    if pgid > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        group_alive = False
        if pgid > 0:
            try:
                os.killpg(pgid, 0)
                group_alive = True
            except (ProcessLookupError, PermissionError):
                pass

        descendants_alive = False
        if psutil is not None:
            descendants_alive = any(
                proc.is_running()
                for proc in processes
                if proc.pid != os.getpid()
            )

        if not group_alive and not descendants_alive:
            return
        time.sleep(0.25)

    if psutil is not None:
        for proc in reversed(processes):
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    if pgid > 0:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def watchdog_main(args: argparse.Namespace) -> int:
    """Clean evaluator descendants after abrupt Ray worker termination."""
    parent_pid = args.watchdog_parent_pid
    evaluator_pid = args.watchdog_evaluator_pid
    evaluator_pgid = args.watchdog_evaluator_pgid
    lock_dir = Path(args.watchdog_lock_dir)

    while True:
        parent_alive = pid_exists(parent_pid)
        evaluator_alive = pid_exists(evaluator_pid)

        if not evaluator_alive:
            remove_lock(lock_dir)
            return 0

        if not parent_alive:
            kill_process_tree(
                evaluator_pid,
                evaluator_pgid,
                args.process_kill_grace_seconds,
            )
            remove_lock(lock_dir)
            return 0

        time.sleep(args.watchdog_poll_seconds)


class EvaluationLockBusy(RuntimeError):
    pass


def acquire_lock(
    lock_dir: Path,
    *,
    exp_name: str,
    node_id: str,
    hostname: str,
    attempt: int,
    wait_seconds: float,
) -> None:
    """Prevent two Jobs from writing the same experiment output concurrently."""
    deadline = time.monotonic() + wait_seconds

    while True:
        try:
            lock_dir.mkdir()
        except FileExistsError:
            if time.monotonic() < deadline:
                time.sleep(1.0)
                continue

            try:
                owner = (lock_dir / "owner.json").read_text(encoding="utf-8")
            except OSError:
                owner = "unknown"
            raise EvaluationLockBusy(
                f"Output lock remains busy: {lock_dir}; owner={owner}"
            )

        payload = {
            "experiment": exp_name,
            "hostname": hostname,
            "node_id": node_id,
            "ray_worker_pid": os.getpid(),
            "attempt": attempt,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (lock_dir / "owner.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return


# ============================================================================
# One Ray task = one complete experiment
# ============================================================================


@ray.remote
def run_experiment(
    config: dict[str, Any],
    exp_name: str,
    coordinator_attempt: int,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent
    math_eval_script = project_root / config["math_eval_script"]
    eval_python = Path(config["python_executable"])
    model_path = Path(config["ray_data_home"]) / "models" / exp_name
    output_dir = (
        Path(config["ray_data_home"])
        / "eval_outputs"
        / f"{exp_name}_{config['sample_num']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluation.log"
    lock_dir = output_dir / ".evaluation_running"

    context = ray.get_runtime_context()
    try:
        node_id = str(context.get_node_id())
    except Exception:
        node_id = "unknown"

    hostname = socket.gethostname()
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")

    if not eval_python.is_file():
        raise FileNotFoundError(f"Evaluation Python not found: {eval_python}")
    if not math_eval_script.is_file():
        raise FileNotFoundError(f"Evaluation script not found: {math_eval_script}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    acquire_lock(
        lock_dir,
        exp_name=exp_name,
        node_id=node_id,
        hostname=hostname,
        attempt=coordinator_attempt,
        wait_seconds=float(config["lock_wait_seconds"]),
    )

    command = [
        str(eval_python),
        "-u",
        str(math_eval_script),
        "--model_name_or_path", str(model_path),
        "--data_dir", config["data_dir"],
        "--data_names", config["data_names"],
        "--output_dir", str(output_dir),
        "--split", config["split"],
        "--prompt_type", config["prompt_type"],
        "--num_test_sample", str(config["num_test_sample"]),
        "--max_tokens_per_call", str(config["max_tokens_per_call"]),
        "--seed", str(config["seed"]),
        "--temperature", str(config["temperature"]),
        "--n_sampling", str(config["sample_num"]),
        "--top_p", str(config["top_p"]),
        "--top_k", str(config["top_k"]),
        "--start", str(config["start"]),
        "--end", str(config["end"]),
        "--pipeline_parallel_size", str(config["pipeline_parallel_size"]),
        "--use_vllm",
        "--apply_chat_template",
        "--save_outputs",
    ]

    env = os.environ.copy()
    env["RAY_DATA_HOME"] = config["ray_data_home"]
    env["TOKENIZERS_PARALLELISM"] = "false"

    header = (
        "\n======================================================\n"
        f"[START] experiment={exp_name}\n"
        f"host={hostname}\n"
        f"node_id={node_id}\n"
        f"attempt={coordinator_attempt}\n"
        f"CUDA_VISIBLE_DEVICES={visible_gpus}\n"
        f"model={model_path}\n"
        f"output={output_dir}\n"
        "======================================================\n"
    )
    print(header, end="", flush=True)

    process: subprocess.Popen[str] | None = None
    watchdog: subprocess.Popen[Any] | None = None
    return_code: int | None = None

    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            log_file.write(header)
            log_file.write("COMMAND: " + " ".join(command) + "\n")

            # The evaluator and all descendants get a dedicated process group.
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            evaluator_pgid = os.getpgid(process.pid)

            # This detached watchdog survives abrupt Ray-worker termination.
            watchdog = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--watchdog",
                    "--watchdog-parent-pid", str(os.getpid()),
                    "--watchdog-evaluator-pid", str(process.pid),
                    "--watchdog-evaluator-pgid", str(evaluator_pgid),
                    "--watchdog-lock-dir", str(lock_dir),
                    "--watchdog-poll-seconds",
                    str(config["watchdog_poll_seconds"]),
                    "--process-kill-grace-seconds",
                    str(config["process_kill_grace_seconds"]),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )

            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                print(f"[{exp_name}] {line}", end="", flush=True)

            return_code = process.wait()

    finally:
        if process is not None and process.poll() is None:
            try:
                evaluator_pgid = os.getpgid(process.pid)
            except ProcessLookupError:
                evaluator_pgid = -1
            kill_process_tree(
                process.pid,
                evaluator_pgid,
                float(config["process_kill_grace_seconds"]),
            )

        remove_lock(lock_dir)

        if watchdog is not None and watchdog.poll() is None:
            watchdog.terminate()
            try:
                watchdog.wait(timeout=5)
            except subprocess.TimeoutExpired:
                watchdog.kill()

    if return_code != 0:
        raise RuntimeError(
            f"Experiment {exp_name} exited with code {return_code}; "
            f"see {log_path}"
        )

    result = {
        "experiment": exp_name,
        "hostname": hostname,
        "node_id": node_id,
        "visible_gpus": visible_gpus,
        "attempt": coordinator_attempt,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }
    print(f"[SUCCEEDED] {result}", flush=True)
    return result


# ============================================================================
# Thin coordinator: submit, monitor, retry, and write status
# ============================================================================


@dataclass
class PendingExperiment:
    exp_name: str
    attempt: int = 0
    ready_at: float = 0.0
    reason: str = "initial"


@dataclass
class InFlightTask:
    ref: ray.ObjectRef
    exp_name: str
    attempt: int
    submitted_at: float
    task_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", action="append")
    parser.add_argument(
        "--python-executable",
        default=os.environ.get("EVAL_PYTHON", sys.executable),
    )
    parser.add_argument("--math-eval-script", default="math_eval.py")
    parser.add_argument("--ray-data-home", default=os.environ.get("RAY_DATA_HOME", ""))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", ""))
    parser.add_argument(
        "--data-names",
        default="math500,aime24,minerva_math,olympiadbench,amc23,aime26,gpqa_diamond,mmlu_stem,livecodebench_v6",
    )
    parser.add_argument("--sample-num", type=int, default=16)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-test-sample", type=int, default=-1)
    parser.add_argument("--prompt-type", default="qwen25-math-cot")
    parser.add_argument("--max-tokens-per-call", type=int, default=38912)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)

    parser.add_argument("--gpus-per-exp", type=float, default=4.0)
    parser.add_argument("--cpus-per-exp", type=float, default=16.0)
    parser.add_argument("--max-task-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=15.0)
    parser.add_argument("--lock-wait-seconds", type=float, default=90.0)
    parser.add_argument("--watchdog-poll-seconds", type=float, default=1.0)
    parser.add_argument("--process-kill-grace-seconds", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--status-interval", type=float, default=20.0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--status-file", default="")

    # Internal watchdog mode.
    parser.add_argument("--watchdog", action="store_true")
    parser.add_argument("--watchdog-parent-pid", type=int, default=0)
    parser.add_argument("--watchdog-evaluator-pid", type=int, default=0)
    parser.add_argument("--watchdog-evaluator-pgid", type=int, default=0)
    parser.add_argument("--watchdog-lock-dir", default="")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.watchdog:
        return
    if not args.exp:
        raise ValueError("At least one --exp is required.")
    if len(args.exp) != len(set(args.exp)):
        duplicates = sorted({name for name in args.exp if args.exp.count(name) > 1})
        raise ValueError(f"Duplicate experiment names: {duplicates}")
    if args.gpus_per_exp <= 0:
        raise ValueError("gpus-per-exp must be positive.")
    if args.cpus_per_exp <= 0:
        raise ValueError("cpus-per-exp must be positive.")
    if args.max_task_retries < 0:
        raise ValueError("max-task-retries must be non-negative.")
    if args.retry_delay_seconds < 0:
        raise ValueError("retry-delay-seconds must be non-negative.")


def task_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "python_executable": args.python_executable,
        "math_eval_script": args.math_eval_script,
        "ray_data_home": args.ray_data_home,
        "data_dir": args.data_dir,
        "data_names": args.data_names,
        "sample_num": args.sample_num,
        "split": args.split,
        "num_test_sample": args.num_test_sample,
        "prompt_type": args.prompt_type,
        "max_tokens_per_call": args.max_tokens_per_call,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "start": args.start,
        "end": args.end,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "lock_wait_seconds": args.lock_wait_seconds,
        "watchdog_poll_seconds": args.watchdog_poll_seconds,
        "process_kill_grace_seconds": args.process_kill_grace_seconds,
    }


def exception_names(exc: BaseException) -> set[str]:
    names: set[str] = set()
    current: BaseException | None = exc

    for _ in range(10):
        if current is None:
            break
        names.add(type(current).__name__)

        next_exc: BaseException | None = None
        if isinstance(current, RayTaskError):
            try:
                candidate = current.as_instanceof_cause()
                if candidate is not current:
                    next_exc = candidate
            except Exception:
                pass

        if next_exc is None:
            next_exc = current.__cause__ or current.__context__
        current = next_exc

    return names


def classify_failure(exc: BaseException) -> str:
    names = exception_names(exc)
    if names.intersection(
        {
            "NodeDiedError",
            "WorkerCrashedError",
            "RaySystemError",
            "LocalRayletDiedError",
            "OwnerDiedError",
        }
    ):
        return "system_failure"
    if "EvaluationLockBusy" in names:
        return "lock_busy"
    if "TaskCancelledError" in names:
        return "cancelled"
    return "application_failure"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def coordinator_main(args: argparse.Namespace) -> int:
    validate_args(args)

    project_root = Path(__file__).resolve().parent
    if not (project_root / args.math_eval_script).is_file():
        raise FileNotFoundError(
            f"Missing {args.math_eval_script} beside {Path(__file__).name}"
        )

    ray.init(address="auto", log_to_driver=True)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    status_file = (
        Path(args.status_file)
        if args.status_file
        else Path(args.ray_data_home)
        / "eval_outputs"
        / f"eval_queue_{run_id}_status.json"
    )

    pending: deque[PendingExperiment] = deque(
        PendingExperiment(exp_name=name) for name in args.exp
    )
    in_flight: dict[ray.ObjectRef, InFlightTask] = {}
    completed: list[str] = []
    failed: dict[str, str] = {}
    retry_count: dict[str, int] = {name: 0 for name in args.exp}
    config = task_config(args)

    stopping = False
    last_status = 0.0

    def stop_handler(signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        print(f"[COORDINATOR] signal={signum}; stopping", flush=True)

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    def submit(item: PendingExperiment) -> None:
        task_name = (
            f"eval::{run_id}::{item.exp_name}::attempt-{item.attempt}"
        )
        ref = run_experiment.options(
            name=task_name,
            num_cpus=args.cpus_per_exp,
            num_gpus=args.gpus_per_exp,
            max_retries=0,
        ).remote(config, item.exp_name, item.attempt)

        in_flight[ref] = InFlightTask(
            ref=ref,
            exp_name=item.exp_name,
            attempt=item.attempt,
            submitted_at=time.monotonic(),
            task_name=task_name,
        )
        print(
            f"[SUBMIT] exp={item.exp_name} attempt={item.attempt} "
            f"reason={item.reason} GPUs={args.gpus_per_exp:g} "
            f"CPUs={args.cpus_per_exp:g}",
            flush=True,
        )

    def submit_ready() -> None:
        """Submit every ready item; Ray itself queues tasks lacking resources."""
        now = time.monotonic()
        count = len(pending)
        for _ in range(count):
            item = pending.popleft()
            if item.ready_at <= now:
                submit(item)
            else:
                pending.append(item)

    def resource_snapshot() -> dict[str, Any]:
        try:
            total = {
                key: float(value)
                for key, value in ray.cluster_resources().items()
            }
            available = {
                key: float(value)
                for key, value in ray.available_resources().items()
            }
            return {"cluster": total, "available": available}
        except Exception as exc:
            return {"error": {"message": repr(exc)}}

    def payload() -> dict[str, Any]:
        now = time.monotonic()
        return {
            "run_id": run_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": (
                "in_flight means submitted to Ray; an item may be RUNNING or "
                "pending for cluster resources"
            ),
            "resources_per_experiment": {
                "GPU": args.gpus_per_exp,
                "CPU": args.cpus_per_exp,
            },
            "summary": {
                "waiting_retry": len(pending),
                "in_flight": len(in_flight),
                "completed": len(completed),
                "failed": len(failed),
            },
            "waiting_retry": [
                {
                    "experiment": item.exp_name,
                    "attempt": item.attempt,
                    "reason": item.reason,
                    "ready_in_seconds": max(0.0, item.ready_at - now),
                }
                for item in pending
            ],
            "in_flight": [
                {
                    "experiment": task.exp_name,
                    "attempt": task.attempt,
                    "submitted_seconds_ago": now - task.submitted_at,
                    "task_name": task.task_name,
                }
                for task in in_flight.values()
            ],
            "completed": completed,
            "failed": failed,
            "retry_count": retry_count,
            "ray_resources": resource_snapshot(),
        }

    def save_status() -> None:
        atomic_json(status_file, payload())

    def print_status() -> None:
        state = payload()
        summary = state["summary"]
        available = state["ray_resources"].get("available", {})
        print(
            "[STATUS] "
            f"in_flight={summary['in_flight']} "
            f"waiting_retry={summary['waiting_retry']} "
            f"succeeded={summary['completed']} "
            f"failed={summary['failed']} "
            f"available_GPU={available.get('GPU', 0):g} "
            f"file={status_file}",
            flush=True,
        )

    try:
        while pending or in_flight:
            if stopping:
                break

            submit_ready()

            if time.monotonic() - last_status >= args.status_interval:
                print_status()
                save_status()
                last_status = time.monotonic()

            if not in_flight:
                time.sleep(args.poll_seconds)
                continue

            ready, _ = ray.wait(
                list(in_flight),
                num_returns=1,
                timeout=args.poll_seconds,
            )
            if not ready:
                continue

            ref = ready[0]
            task = in_flight.pop(ref)

            try:
                result = ray.get(ref)
            except Exception as exc:
                category = classify_failure(exc)

                if (
                    category in {"system_failure", "lock_busy"}
                    and task.attempt < args.max_task_retries
                ):
                    attempt = task.attempt + 1
                    retry_count[task.exp_name] = attempt
                    pending.append(
                        PendingExperiment(
                            exp_name=task.exp_name,
                            attempt=attempt,
                            ready_at=(
                                time.monotonic() + args.retry_delay_seconds
                            ),
                            reason=category,
                        )
                    )
                    print(
                        f"[REQUEUE] exp={task.exp_name} reason={category} "
                        f"attempt={attempt}/{args.max_task_retries}",
                        flush=True,
                    )
                elif category == "cancelled" and stopping:
                    print(
                        f"[CANCELLED] exp={task.exp_name} during shutdown",
                        flush=True,
                    )
                else:
                    failed[task.exp_name] = repr(exc)
                    print(
                        f"[FAILED] exp={task.exp_name} "
                        f"category={category} error={exc!r}",
                        flush=True,
                    )
            else:
                completed.append(task.exp_name)
                print(
                    f"[COMPLETED] exp={task.exp_name} "
                    f"host={result['hostname']} "
                    f"GPUs={result['visible_gpus']} "
                    f"attempt={task.attempt}",
                    flush=True,
                )

            save_status()

        print_status()
        save_status()

    finally:
        if stopping:
            for ref, task in list(in_flight.items()):
                print(f"[CANCEL] exp={task.exp_name}", flush=True)
                try:
                    ray.cancel(ref, force=True)
                except Exception:
                    pass
        save_status()

    print("\n================ FINAL SUMMARY ================", flush=True)
    print(f"Status file: {status_file}", flush=True)
    print(f"Succeeded ({len(completed)}):", flush=True)
    for exp_name in completed:
        print(f"  - {exp_name}", flush=True)
    print(f"Failed ({len(failed)}):", flush=True)
    for exp_name, error in failed.items():
        print(f"  - {exp_name}: {error}", flush=True)
    print("================================================", flush=True)

    if stopping:
        return 130
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    if args.watchdog:
        return watchdog_main(args)
    return coordinator_main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[COORDINATOR FATAL] {exc!r}", file=sys.stderr, flush=True)
        raise
