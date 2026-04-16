"""
Minimal local web UI for driving MolmoWeb from a browser.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent.actions import SendMsgToUser
from inference import MolmoWeb, Trajectory


ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT_DIR / "inference" / "htmls" / "gui"
HTML_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = HTML_DIR / "history.json"
TEMPLATES_DIR = ROOT_DIR / "inference" / "templates"
STATIC_DIR = ROOT_DIR / "inference" / "static"
TEXTS_PATH = ROOT_DIR / "inference" / "texts.json"

DEFAULT_ENDPOINT = os.environ.get("MOLMOWEB_GUI_ENDPOINT", "http://127.0.0.1:8001")
DEFAULT_MAX_STEPS = int(os.environ.get("MOLMOWEB_GUI_MAX_STEPS", "10"))
HEADLESS = os.environ.get("MOLMOWEB_GUI_HEADLESS", "false").lower() == "true"
LANGS = {"en", "ja"}

with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    TEXT = json.load(f)


def _t(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, key)


def _lang_value(lang: str) -> str:
    return lang if lang in LANGS else "en"


@dataclass
class LiveRun:
    running: bool = False
    prompt: str = ""
    max_steps: int = 0
    step_num: int = 0
    action: str = ""
    page_url: str = ""
    page_title: str = ""
    status: str = "idle"
    error: str | None = None
    answer: str = ""
    started_at: str = ""
    finished_at: str = ""
    trajectory_href: str = ""
    steps_log: list[dict[str, str]] = field(default_factory=list)


@dataclass
class RunRecord:
    id: str
    prompt: str
    answer: str
    status: str
    max_steps: int
    created_at: str
    trajectory_href: str | None = None
    final_url: str | None = None
    final_title: str | None = None
    step_count: int = 0
    error: str | None = None


class AppState:
    def __init__(self):
        self.endpoint: str = DEFAULT_ENDPOINT
        self.max_steps: int = DEFAULT_MAX_STEPS
        self.headless: bool = HEADLESS
        self.viewport_width: int = 1600
        self.viewport_height: int = 1000
        self.client: MolmoWeb | None = None
        self.history: list[RunRecord] = []
        self.last_error: str | None = None
        self.live_run: LiveRun = LiveRun()
        self.worker: threading.Thread | None = None
        self.task_queue: queue.Queue[tuple[str, int] | None] = queue.Queue()
        self.stop_requested: bool = False
        self.lock: threading.Lock = threading.Lock()

    def get_client(self) -> MolmoWeb:
        if self.client is None:
            self.client = MolmoWeb(
                endpoint=self.endpoint,
                headless=self.headless,
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height,
            )
            self.client.step_callback = self.update_live_run
        return self.client

    def reset_browser(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.last_error = None
        self.stop_requested = False
        self.live_run = LiveRun()

    def force_stop(self) -> None:
        self.stop_requested = True
        if self.client is not None:
            self.client.close()
            self.client = None
        self.last_error = None
        self.live_run.running = False
        self.live_run.status = "stopped"
        self.live_run.error = None
        self.live_run.answer = "Force stopped"
        self.live_run.finished_at = datetime.now().strftime("%H:%M:%S")

    def update_live_run(self, event: dict) -> None:
        with self.lock:
            self.live_run.step_num = event.get("step_num", self.live_run.step_num)
            self.live_run.max_steps = event.get("max_steps", self.live_run.max_steps)
            self.live_run.action = event.get("action") or ""
            self.live_run.page_url = event.get("page_url") or ""
            self.live_run.page_title = event.get("page_title") or ""
            self.live_run.error = event.get("error")
            step_num = event.get("step_num")
            action = event.get("action") or ""
            if step_num and action:
                item = {
                    "summary": f"{step_num}. {action}",
                    "screenshot_base64": event.get("screenshot_base64") or "",
                    "page_url": event.get("page_url") or "",
                    "page_title": event.get("page_title") or "",
                }
                if (
                    not self.live_run.steps_log
                    or self.live_run.steps_log[-1]["summary"] != item["summary"]
                ):
                    self.live_run.steps_log.append(item)
            if self.live_run.running:
                self.live_run.status = "running"

    def ensure_worker(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=_worker_loop, daemon=True)
            self.worker.start()


def _load_history() -> list[RunRecord]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text())
        if not isinstance(data, list):
            return []
        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item.setdefault("id", uuid4().hex)
            item.setdefault("trajectory_href", item.get("trajectory_href") or "")
            item.setdefault("error", item.get("error"))
            item.setdefault("step_count", item.get("step_count", 0))
            records.append(RunRecord(**item))
        return records
    except Exception:
        return []


def _save_history() -> None:
    HISTORY_PATH.write_text(
        json.dumps(
            [asdict(record) for record in STATE.history], ensure_ascii=False, indent=2
        )
    )


STATE = AppState()
STATE.history = _load_history()
app = FastAPI(title="MolmoWeb GUI")
app.mount(
    "/artifacts",
    StaticFiles(directory=str(ROOT_DIR / "inference" / "htmls")),
    name="artifacts",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(ROOT_DIR / "assets")), name="assets")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _extract_answer(traj: Trajectory) -> tuple[str, str]:
    for step in reversed(traj.steps):
        if step.prediction and isinstance(step.prediction.action, SendMsgToUser):
            return step.prediction.action.message, "answered"
    if traj.steps and traj.steps[-1].state and traj.steps[-1].state.get("done"):
        return "", "completed"
    return "", "incomplete"


def _save_trajectory(traj: Trajectory, prompt: str) -> str:
    slug = (
        "".join(ch if ch.isalnum() else "_" for ch in prompt).strip("_")[:40] or "task"
    )
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_{slug}_{uuid4().hex[:8]}.html"
    output_path = HTML_DIR / filename
    traj.save_html(output_path=str(output_path), query=prompt)
    return f"/artifacts/gui/{filename}"


def _truncate_text(text: str | None, max_chars: int = 140) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _render_record(record: RunRecord, lang: str) -> str:
    status = (
        _t(lang, record.status)
        if record.status in {"completed", "answered", "incomplete", "stopped"}
        else record.status
    )
    detail_bits = [
        f"{escape(_t(lang, 'steps'))}: {record.step_count}",
        f"max_steps: {record.max_steps}",
    ]
    if record.final_title:
        detail_bits.append(f"title: {escape(record.final_title)}")
    if record.final_url:
        detail_bits.append(f"url: {escape(record.final_url)}")
    if record.trajectory_href:
        detail_bits.append(
            f'<a href="{escape(record.trajectory_href)}" target="_blank" rel="noreferrer">{escape(_t(lang, "trajectory"))}</a>'
        )
    error_html = (
        f'<p class="meta-line">{escape(_t(lang, "error"))}: {escape(record.error)}</p>'
        if record.error
        else ""
    )
    prompt_preview = escape(_truncate_text(record.prompt, 110))
    answer_preview = escape(_truncate_text(record.answer, 150))
    full_prompt = escape(record.prompt)
    full_answer = escape(record.answer)
    return f"""
    <article class="record" onclick="this.classList.toggle('expanded')">
      <div class="record-header">
        <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
        <div class="record-meta">
          <span>{escape(record.created_at)}</span>
          <span>{escape(status)}</span>
        </div>
        <button class="delete-btn" type="button" data-record-id="{escape(record.id)}" title="{escape(_t(lang, "delete"))}">&#x1F5D1;&#xFE0E;</button>
      </div>
      <p class="record-copy"><strong>{escape(_t(lang, "prompt"))}</strong><br><span class="preview">{prompt_preview}</span><span class="full" hidden>{full_prompt}</span></p>
      <p class="answer"><strong>{escape(_t(lang, "answer"))}</strong><br><span class="preview">{answer_preview}</span><span class="full" hidden>{full_answer}</span></p>
      <p class="meta-line">{" | ".join(detail_bits)}</p>
      {error_html}
    </article>
    """


def _status_payload(lang: str) -> dict:
    live = asdict(STATE.live_run)
    live["browser_status"] = (
        _t(lang, "ready") if STATE.client is not None else _t(lang, "not_started")
    )
    live["localized_status"] = (
        _t(lang, live["status"])
        if live["status"] in {"completed", "answered", "incomplete", "stopped"}
        else (_t(lang, "running") if live["running"] else _t(lang, "idle"))
    )
    live["history_html"] = "".join(
        _render_record(record, lang) for record in reversed(STATE.history)
    ) or (f'<p class="hint">{escape(_t(lang, "no_runs"))}</p>')
    live["trace_html"] = (
        "".join(
            _render_trace_item(item, idx) for idx, item in enumerate(live["steps_log"])
        )
        or f'<p class="hint">{escape(_t(lang, "idle"))}</p>'
    )
    live["last_error"] = STATE.last_error
    steps = live["steps_log"]
    if len(steps) >= 2:
        latest = steps[-1]
        live["latest_step_html"] = _render_latest_step(latest)
    elif len(steps) == 1:
        live["latest_step_html"] = f'<p class="hint">{escape(_t(lang, "idle"))}</p>'
    else:
        live["latest_step_html"] = f'<p class="hint">-</p>'
    return live


def _render_latest_step(item: dict[str, str]) -> str:
    parts = []
    summary = item.get("summary", "")
    if summary:
        parts.append(f'<div class="latest-action">{escape(summary)}</div>')
    meta = []
    if item.get("page_title"):
        meta.append(f"title: {escape(item['page_title'])}")
    if item.get("page_url"):
        meta.append(f"url: {escape(item['page_url'])}")
    if meta:
        parts.append(f'<div class="trace-meta">{" | ".join(meta)}</div>')
    if item.get("screenshot_base64"):
        parts.append(
            f'<img class="trace-shot" alt="step screenshot" '
            f'src="data:image/png;base64,{item["screenshot_base64"]}">'
        )
    return "".join(parts)


def _render_trace_item(item: dict[str, str], idx: int = 0) -> str:
    is_first = idx == 0
    img_html = ""
    if not is_first and item.get("screenshot_base64"):
        img_html = (
            f'<img class="trace-shot" alt="step screenshot" '
            f'src="data:image/png;base64,{item["screenshot_base64"]}">'
        )
    meta = []
    if item.get("page_title"):
        meta.append(f"title: {escape(item['page_title'])}")
    if not is_first and item.get("page_url"):
        meta.append(f"url: {escape(item['page_url'])}")
    meta_html = f'<div class="trace-meta">{" | ".join(meta)}</div>' if meta else ""
    open_attr = "" if is_first else " open"
    return (
        f'<details class="trace-item"{open_attr}>'
        f"<summary>{escape(item.get('summary', ''))}</summary>"
        f"{meta_html}"
        f"{img_html}"
        f"</details>"
    )


def _finalize_run(prompt: str, max_steps: int, traj: Trajectory) -> None:
    answer, status = _extract_answer(traj)
    if STATE.stop_requested:
        answer, status = "Force stopped", "stopped"
    href = _save_trajectory(traj, prompt)
    last_state = traj.steps[-1].state if traj.steps else None
    error = next((step.error for step in reversed(traj.steps) if step.error), None)
    with STATE.lock:
        record = RunRecord(
            id=uuid4().hex[:12],
            prompt=prompt,
            answer=answer,
            status=status,
            max_steps=max_steps,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            trajectory_href=href,
            final_url=last_state.page_url if last_state else None,
            final_title=last_state.page_title if last_state else None,
            step_count=len(traj.steps),
            error=error,
        )
        STATE.history.append(record)
        _save_history()
        STATE.live_run.running = False
        STATE.live_run.status = status
        STATE.live_run.answer = answer
        STATE.live_run.error = error
        STATE.live_run.finished_at = datetime.now().strftime("%H:%M:%S")
        STATE.live_run.trajectory_href = href
        STATE.stop_requested = False


def _worker_loop() -> None:
    while True:
        item = STATE.task_queue.get()
        if item is None:
            break
        prompt, max_steps = item
        with STATE.lock:
            STATE.last_error = None
        client = STATE.get_client()
        try:
            traj = client.run(query=prompt, max_steps=max_steps)
            _finalize_run(prompt, max_steps, traj)
        except Exception as exc:
            with STATE.lock:
                if STATE.stop_requested:
                    STATE.last_error = None
                    STATE.live_run.running = False
                    STATE.live_run.error = None
                    STATE.live_run.status = "stopped"
                    STATE.live_run.answer = "Force stopped"
                    STATE.live_run.finished_at = datetime.now().strftime("%H:%M:%S")
                    STATE.stop_requested = False
                    continue
                STATE.last_error = str(exc)
                STATE.live_run.running = False
                STATE.live_run.error = str(exc)
                STATE.live_run.status = "error"
                STATE.live_run.finished_at = datetime.now().strftime("%H:%M:%S")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, lang: str = "en"):
    lang = _lang_value(lang)
    data = _status_payload(lang)
    current_step = (
        escape(_t(lang, "queued"))
        if data["running"] and not data["step_num"]
        else f"{data['step_num']}/{data['max_steps']}"
        if data["step_num"]
        else "-"
    )
    text_json = json.dumps(
        {
            "runTask": _t(lang, "run_task"),
            "running": _t(lang, "running"),
            "forceStop": _t(lang, "force_stop"),
            "taskFinished": _t(lang, "task_finished"),
            "taskFailed": _t(lang, "task_failed"),
            "enterTask": _t(lang, "enter_task"),
            "idle": _t(lang, "idle"),
            "queued": _t(lang, "queued"),
        }
    )
    return templates.TemplateResponse(
        "gui.html",
        {
            "request": request,
            "lang": lang,
            "data": data,
            "current_step": current_step,
            "text_json": text_json,
            "title": _t(lang, "title"),
            "version_suffix": _t(lang, "version_suffix"),
            "help_button": _t(lang, "help_button"),
            "help_heading": _t(lang, "help_heading"),
            "help_steps": _t(lang, "help_steps"),
            "help_note": _t(lang, "help_note"),
            "endpoint": STATE.endpoint,
            "history_toggle": _t(lang, "history_toggle"),
            "lang_ja": _t(lang, "lang_ja"),
            "lang_en": _t(lang, "lang_en"),
            "browser_session_label": _t(lang, "browser_session"),
            "headless_label": _t(lang, "headless"),
            "headless": STATE.headless,
            "task_label": _t(lang, "task"),
            "task_placeholder": _t(lang, "task_placeholder"),
            "max_steps_label": _t(lang, "max_steps"),
            "max_steps_default": STATE.max_steps,
            "width_label": _t(lang, "width"),
            "height_label": _t(lang, "height"),
            "viewport_width": STATE.viewport_width,
            "viewport_height": STATE.viewport_height,
            "run_task_label": _t(lang, "run_task"),
            "reset_label": _t(lang, "reset"),
            "force_stop_label": _t(lang, "force_stop"),
            "current_run_label": _t(lang, "current_run"),
            "idle_label": _t(lang, "idle"),
            "page_title_label": _t(lang, "page_title"),
            "current_step_label": _t(lang, "current_step"),
            "live_trace_label": _t(lang, "live_trace"),
            "expand_all_label": _t(lang, "expand_all"),
            "history_label": _t(lang, "history"),
            "clear_history_label": _t(lang, "clear_history"),
            "history_close_label": _t(lang, "history_close"),
            "window_size_invalid_text": _t(lang, "window_size_invalid"),
        },
    )


@app.get("/api/status")
def api_status(lang: str = "en") -> JSONResponse:
    with STATE.lock:
        return JSONResponse(_status_payload(_lang_value(lang)))


@app.post("/api/run")
def api_run(
    prompt: str = Form(...),
    max_steps: int = Form(DEFAULT_MAX_STEPS),
    lang: str = Form("en"),
    window_width: int = Form(1600),
    window_height: int = Form(1000),
) -> JSONResponse:
    lang = _lang_value(lang)
    prompt = prompt.strip()
    if not prompt:
        return JSONResponse({"detail": _t(lang, "enter_task")}, status_code=400)
    width = int(window_width)
    height = int(window_height)
    if width < 640 or height < 480:
        return JSONResponse(
            {"detail": _t(lang, "window_size_invalid")}, status_code=400
        )

    with STATE.lock:
        if STATE.live_run.running:
            return JSONResponse({"detail": _t(lang, "running")}, status_code=409)
        if STATE.viewport_width != width or STATE.viewport_height != height:
            STATE.viewport_width = width
            STATE.viewport_height = height
            if STATE.client is not None:
                STATE.client.close()
                STATE.client = None
        STATE.last_error = None
        STATE.stop_requested = False
        STATE.live_run = LiveRun(
            running=True,
            prompt=prompt,
            max_steps=max_steps,
            status="running",
            started_at=datetime.now().strftime("%H:%M:%S"),
            steps_log=[],
        )
        STATE.ensure_worker()
        STATE.task_queue.put((prompt, max_steps))
        return JSONResponse(_status_payload(lang))


@app.post("/reset")
def reset(lang: str = Form("en")) -> RedirectResponse:
    with STATE.lock:
        if not STATE.live_run.running:
            STATE.reset_browser()
    return RedirectResponse(f"/?lang={_lang_value(lang)}", status_code=303)


@app.post("/api/stop")
def api_stop(lang: str = Form("en")) -> JSONResponse:
    lang = _lang_value(lang)
    with STATE.lock:
        if not STATE.live_run.running:
            return JSONResponse(_status_payload(lang))
        STATE.force_stop()
        return JSONResponse(_status_payload(lang))


@app.post("/api/history/{record_id}/delete")
def delete_history_record(record_id: str, lang: str = Form("en")) -> JSONResponse:
    with STATE.lock:
        STATE.history = [record for record in STATE.history if record.id != record_id]
        _save_history()
        return JSONResponse(_status_payload(_lang_value(lang)))


@app.post("/api/history/clear")
def clear_history(lang: str = Form("en")) -> JSONResponse:
    with STATE.lock:
        STATE.history = []
        _save_history()
        return JSONResponse(_status_payload(_lang_value(lang)))
