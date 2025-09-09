from __future__ import annotations
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


class EmailSender:
    def __init__(self, host: str, port: int, username: str, password: str, from_addr: str) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_addr = from_addr

    def send_text(self, to_addr: str, subject: str, body: str) -> None:
        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = subject
        msg["From"] = formataddr(("Компания Миранда", self.from_addr))
        msg["To"] = to_addr

        if self.port == 465:
            with smtplib.SMTP_SSL(self.host, self.port) as server:
                server.login(self.username, self.password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg) 