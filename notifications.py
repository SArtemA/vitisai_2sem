"""Telegram notifications sent to the developer, configured through environment variables."""
from __future__ import annotations
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
STATE_PATH = Path(os.getenv("RETRAIN_NOTIFICATION_STATE", str(PROJECT_DIR / "databases" / "retraining_state.json")))


def send_telegram_message(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        logger.warning("Telegram disabled: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
    request = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not result.get("ok"):
            logger.error("Telegram API error: %s", result)
            return False
        return True
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        logger.exception("Telegram notification could not be sent")
        return False


def _load_state() -> dict:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return {"consecutive_no_improvement": max(0, int(data.get("consecutive_no_improvement", 0)))}
    except (OSError, ValueError, TypeError):
        return {"consecutive_no_improvement": 0}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, STATE_PATH)


def _score(value) -> str:
    return "не рассчитано" if value is None else f"{float(value):.4f} ({float(value) * 100:.2f}%)"


def process_retraining_result(result: dict) -> None:
    state = _load_state()
    status = result.get("status", "error")
    failures = state["consecutive_no_improvement"]

    if status == "success":
        state["consecutive_no_improvement"] = 0
        _save_state(state)
        old, new = result.get("old_f1"), result.get("new_f1")
        if old is None:
            details = "Предыдущая модель отсутствовала; новая установлена как начальная."
        else:
            details = (f"Предыдущая модель: {_score(old)}\nНовая модель: {_score(new)}\n"
                       f"Улучшение: {(new - old) * 100:+.2f} п.п.")
        send_telegram_message("✅ Переобучение завершено успешно\n\n" + details +
                              f"\nНеудачных попыток до успеха: {failures}\nНовая модель сохранена.")
        return

    if status == "no_improvement":
        failures += 1
        state["consecutive_no_improvement"] = failures
        _save_state(state)
        try:
            threshold = max(1, int(os.getenv("RETRAIN_NOTIFY_AFTER_FAILURES", "4")))
        except ValueError:
            threshold = 4
        if failures >= threshold and (failures - threshold) % threshold == 0:
            send_telegram_message("⚠️ Не удаётся улучшить качество модели\n\n"
                                  f"Последовательных неудачных попыток: {failures}\n"
                                  f"Текущее качество: {_score(result.get('old_f1'))}\n"
                                  f"Качество кандидата: {_score(result.get('new_f1'))}\n"
                                  "Рабочая модель оставлена без изменений.")
        return

    # Technical errors are reported immediately but are not counted as quality failures.
    _save_state(state)
    send_telegram_message("❌ Техническая ошибка переобучения\n\n"
                          f"Причина: {result.get('message', 'Неизвестная ошибка')}\n"
                          f"Неудачных попыток улучшить качество: {failures}\nРабочая модель не заменялась.")
