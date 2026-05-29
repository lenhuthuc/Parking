"""
Alert service — sends notifications via Email and/or Telegram
when a violation is recorded.
"""
from __future__ import annotations
import logging
import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Optional

log = logging.getLogger("parking.alerts")


def _send_email(subject: str, body: str,
                snapshot_path: Optional[str] = None) -> None:
    from config import cfg
    ec = cfg.alerts.email
    if not ec.enabled:
        return

    password = os.environ.get("ALERT_EMAIL_PASSWORD", ec.password)
    if not password:
        log.warning("Email alerts enabled but no password set.")
        return

    msg = MIMEMultipart()
    msg["From"]    = ec.sender
    msg["To"]      = ", ".join(ec.recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path, "rb") as f:
            img = MIMEImage(f.read(), name=os.path.basename(snapshot_path))
        msg.attach(img)

    try:
        with smtplib.SMTP(ec.smtp_host, ec.smtp_port) as smtp:
            smtp.starttls()
            smtp.login(ec.sender, password)
            smtp.sendmail(ec.sender, ec.recipients, msg.as_string())
        log.info("Alert email sent: %s", subject)
    except Exception as exc:
        log.error("Failed to send alert email: %s", exc)


def _send_telegram(message: str,
                   snapshot_path: Optional[str] = None) -> None:
    from config import cfg
    tc = cfg.alerts.telegram
    if not tc.enabled:
        return

    token   = os.environ.get("TELEGRAM_BOT_TOKEN", tc.bot_token)
    chat_id = tc.chat_id
    if not token or not chat_id:
        log.warning("Telegram alerts enabled but token/chat_id missing.")
        return

    try:
        import urllib.request, urllib.parse, json

        if snapshot_path and os.path.exists(snapshot_path):
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            with open(snapshot_path, "rb") as f:
                photo_data = f.read()
            import multipart  # optional; fall back to text if unavailable
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": message,
            }).encode()
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
            log.info("Telegram alert sent.")
    except Exception as exc:
        log.error("Failed to send Telegram alert: %s", exc)


def notify_violation(zone_id: str, confidence: float,
                     camera_name: str,
                     snapshot_path: Optional[str] = None) -> None:
    """
    Fire-and-forget: sends alerts in a daemon thread so the caller
    (detection worker) is not blocked.
    """
    subject = f"[VI PHAM] Zone {zone_id} — camera {camera_name}"
    body = (
        f"Phát hiện xe vi phạm trong vùng cấm!\n\n"
        f"Camera : {camera_name}\n"
        f"Vùng   : {zone_id}\n"
        f"Độ tin cậy: {confidence:.0%}\n"
    )

    def _send():
        _send_email(subject, body, snapshot_path)
        _send_telegram(f"{subject}\n{body}", snapshot_path)

    t = threading.Thread(target=_send, daemon=True)
    t.start()
