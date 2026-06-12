#!/usr/bin/env python3
import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import List

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from egs.lt_ai_blkt.local.dwn import ParquetKeeper
from egs.lt_ai_blkt.local.parquet_utils import count_rows, iter_text_rows
from egs.lt_ai_blkt.local.utils import has_upper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="""Input parquet file or directory with parquet shards.
        """,
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="""Input parquet column that contains text.
        """,
    )
    parser.add_argument(
        "--tagger-url",
        type=str,
        action="append",
        required=True,
        help="""Tagger URL. Can be provided multiple times and/or as a comma-separated list.
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="""Output parquet path (file name prefix or .parquet file name).
            """,
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=512,
        help="""Target shard size in MB (default: 512).
        """,
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=".tagger_last_conversion.json",
        help="""Path to conversion state json file.
        """,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="""How many input rows between state checkpoints (default: 100).
        """,
    )
    args = parser.parse_args()
    args.tagger_url = _parse_tagger_urls(args.tagger_url)
    return args


def _parse_tagger_urls(values: List[str]) -> List[str]:
    urls: List[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        urls.extend([part for part in parts if part])
    if not urls:
        raise ValueError("No valid --tagger-url values provided")
    return urls


MAX_CHARS = 10000


def _make_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.0,
        backoff_max=30,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["POST"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class Data:
    def __init__(self):
        self.buffer: str = ""
        self.read_lines = 0

    def append(self, line: str):
        self.buffer += line + "\n"
        self.read_lines += 1

    def take_sentences(self, resp, is_last) -> List[str]:
        res = []
        sent = ""
        take_chars = 0
        up_to = 0
        rl = len(resp)
        for i, w in enumerate(resp):
            tp = w.get("type")
            if tp == "SENTENCE_END":
                if sent:
                    up_to = take_chars
                    res.append(sent.strip())
                    sent = ""
                continue
            s = w.get("string", "")
            take_chars += len(s)
            if tp == "WORD":
                mi = w.get("mi", "")
                if has_upper(s):
                    s = s + f"(={w.get('mi', '')})"
                elif mi == "Dl" or mi == "De":
                    logging.info(f"Got word: '{w}'")
                    s = s + f"(={mi})"
            elif tp == "SEPARATOR":
                s = s.replace("\n", " ")
            elif tp == "SPACE":
                pass
            elif tp == "NUMBER":
                pass
            else:
                raise RuntimeError(f"Unknown type: {tp}")
            sent += s
            if not is_last and i > rl * 0.95:
                break
        self.buffer = self.buffer[up_to:]
        return res


@dataclass
class Progress:
    sentences: int = 0
    words: int = 0


class TaggerState:
    def __init__(
            self,
            input_path: str,
            output_path: str,
            read_rows: int,
            sentences_written: int,
            words_written: int,
            output_shard_count: int,
            total_rows: int,
    ):
        self.input = input_path
        self.output = output_path
        self.read_rows = read_rows
        self.sentences_written = sentences_written
        self.words_written = words_written
        self.output_shard_count = output_shard_count
        self.total_rows = total_rows

    @staticmethod
    def from_dict(data: dict) -> "TaggerState":
        return TaggerState(
            input_path=data.get("input", ""),
            output_path=data.get("output", ""),
            read_rows=data.get("read_rows", 0),
            sentences_written=data.get("sentences_written", 0),
            words_written=data.get("words_written", 0),
            output_shard_count=data.get("output_shard_count", 0),
            total_rows=data.get("total_rows", 0),
        )

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "output": self.output,
            "read_rows": self.read_rows,
            "sentences_written": self.sentences_written,
            "words_written": self.words_written,
            "output_shard_count": self.output_shard_count,
            "total_rows": self.total_rows,
        }


def write_out(keeper: ParquetKeeper, sentences):
    wl = 0
    for s in sentences:
        wl += len(s.split())
        keeper.feed_text(s)
    return len(sentences), wl


def _load_state(state_file: str):
    path = Path(state_file)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return TaggerState.from_dict(json.load(f))


def _save_state(state_file: str, state: TaggerState):
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=True, indent=2)


def call(session: requests.Session, url: str, txt: str):
    logging.debug(f"Calling tagger {url} with text of length {len(txt)}")
    headers = {"Content-Type": "application/json"}
    try:
        resp = session.post(url, data=txt.encode("utf-8"), headers=headers, timeout=30)
    except requests.RequestException as err:
        raise RuntimeError(f"failed to send request: {err}") from err

    logging.debug("resp'%s'", resp.status_code)
    if resp.ok:
        try:
            resp_json = resp.json()
        except ValueError as err:
            raise RuntimeError(f"failed to deserialize response: {err}") from err
        return resp_json

    body_str = resp.content.decode("utf-8", errors="replace")
    raise RuntimeError(f"Failed to make request: {resp.status_code} {body_str}")


def split(session: requests.Session, url: str, data, is_last):
    txt = data.buffer[:MAX_CHARS]
    res = call(session, url, txt)
    # logging.debug(res)
    return data.take_sentences(res, is_last)


def _split_line(session: requests.Session, url: str, line: str) -> List[str]:
    data = Data()
    data.append(line)
    out: List[str] = []

    ll = len(data.buffer)
    while ll > MAX_CHARS:
        sentences = split(session, url, data, False)
        out.extend(sentences)
        if ll == len(data.buffer):
            logging.warning(
                "Buffer length did not decrease in worker, dropping remaining chunk to avoid infinite loop"
            )
            data.buffer = ""
            break
        ll = len(data.buffer)

    if data.buffer:
        out.extend(split(session, url, data, True))

    return out


def _worker_loop(
        worker_name: str,
        url: str,
        line_queue: Queue,
        sentence_queue: Queue,
        stop_event: Event,
        error_queue: Queue,
):
    session = _make_http_session()
    while True:
        item = line_queue.get()
        try:
            if item is None:
                break
            if stop_event.is_set():
                continue
            idx, line = item
            sentences = _split_line(session, url, line)
            sentence_queue.put((idx, sentences, None))
        except Exception as err:
            stop_event.set()
            error_queue.put(RuntimeError(f"{worker_name} failed: {err}"))
        finally:
            line_queue.task_done()
    logging.info(f"{worker_name}) exiting")


def _start_workers(
        urls: List[str],
        line_queue: Queue,
        sentence_queue: Queue,
        stop_event: Event,
        error_queue: Queue,
) -> List[Thread]:
    threads: List[Thread] = []
    for i, url in enumerate(urls):
        name = f"tagger-worker-{i + 1}"
        th = Thread(
            target=_worker_loop,
            args=(name, url, line_queue, sentence_queue, stop_event, error_queue),
            name=name,
            daemon=False,
        )
        th.start()
        threads.append(th)
    return threads


def _writer_loop(args, sentence_queue: Queue, state: TaggerState, stop_event: Event, error_queue: Queue, progress: Progress,
                 progress_lock: Lock):
    try:
        with (ParquetKeeper(output_dir=Path(args.output)) as keeper):
            last_idx, sc, wc = 0, state.sentences_written, state.words_written
            if state.sentences_written > 0:
                keeper.restore_shard_index(state.output_shard_count)
            while True:
                item = sentence_queue.get()
                try:
                    if item is None:
                        break
                    idx, sentences, err = item
                    if err:
                        raise RuntimeError(f"Worker error: {err}")
                    sl, wl = write_out(keeper, sentences)
                    sc += sl
                    wc += wl
                    with progress_lock:
                        progress.sentences = sc
                        progress.words = wc
                    last_idx = idx
                    if idx % args.checkpoint_every == 0:
                        state.read_rows = last_idx
                        state.sentences_written = sc
                        state.words_written = wc
                        state.output_shard_count = keeper.shard_count
                        _save_state(args.state_file, state)
                finally:
                    sentence_queue.task_done()
            state.read_rows = last_idx
            state.sentences_written = sc
            state.words_written = wc
            state.output_shard_count = keeper.shard_count
            _save_state(args.state_file, state)
            with progress_lock:
                progress.sentences = sc
                progress.words = wc
    except Exception as err:  # noqa: BLE001
        stop_event.set()
        error_queue.put(RuntimeError(f"tagger-writer failed: {err}"))


def _stop_workers(line_queue: Queue, threads: List[Thread]):
    for _ in threads:
        line_queue.put(None)
    line_queue.join()
    for th in threads:
        th.join(timeout=5)


def main():
    args = get_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")

    workers = len(args.tagger_url)
    logging.info(f"Using {workers} tagger URLs")

    total_rows = count_rows(args.input)
    state = _load_state(args.state_file)
    if state:
        logging.info("Last conversion state found: %s", args.state_file)
    else:
        logging.info("No previous conversion state found at %s", args.state_file)
        state = TaggerState(
            input_path=args.input,
            output_path=args.output,
            read_rows=0,
            sentences_written=0,
            words_written=0,
            output_shard_count=0,
            total_rows=total_rows,
        )

    logging.info(f"Tagging parquet text from {args.input}")
    logging.info(f"Output parquet: {args.output}")

    stop_event = Event()
    error_queue: Queue = Queue()
    progress = Progress()
    progress_lock = Lock()

    line_queue: Queue = Queue(maxsize=workers)
    sentence_queue: Queue = Queue(maxsize=workers)
    worker_threads = _start_workers(args.tagger_url, line_queue, sentence_queue, stop_event, error_queue)
    logging.info("Started %d workers with queues", len(worker_threads))

    resume_from_row = state.read_rows

    writer_thread = Thread(
        target=_writer_loop,
        args=(
            args,
            sentence_queue,
            state,
            stop_event,
            error_queue,
            progress,
            progress_lock,
        ),
        name="tagger-writer",
        daemon=False,
    )
    writer_thread.start()
    run_error = None
    try:
        with tqdm(total=total_rows, unit="rows", desc="Tagging rows") as pbar:
            for idx, line in enumerate(iter_text_rows(args.input, args.text_field)):
                pbar.update(1)
                if stop_event.is_set():
                    try:
                        thread_err = error_queue.get_nowait()
                        raise thread_err
                    except Empty:
                        pass
                    raise RuntimeError("Stopping due to worker/writer error")
                if idx <= resume_from_row:
                    continue
                line_queue.put((idx, line))
                with progress_lock:
                    pbar.set_postfix(sentences=progress.sentences, words=progress.words)
    except Exception as err:  # noqa: BLE001
        run_error = err
        stop_event.set()
    finally:
        logging.info("Stopping workers. please wait or parquet files may be left in inconsistent state...")
        _stop_workers(line_queue, worker_threads)
        logging.info("Stopping writer. please wait or parquet files may be left in inconsistent state...")
        sentence_queue.put(None)
        sentence_queue.join()
        writer_thread.join()
        with progress_lock:
            final_sc = progress.sentences
            final_wc = progress.words
        logging.info("Writer totals: sentences=%d words=%d", final_sc, final_wc)
        if run_error is None:
            try:
                run_error = error_queue.get_nowait()
            except Empty:
                run_error = None

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
