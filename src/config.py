import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    bot_token: str
    bot_username: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_from: str
    manager_phone: str = "+74997450142"
    excel_path: str = "иннпочты.xlsx"
    storage_dir: str = "data"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    replicate_api_token: str = ""
    replicate_seg_model: str = "jagilley/grounded-sam"
    openrouter_api_key: str = ""
    openrouter_model: str = "flux-pro"  # пример: flux-pro / sdxl
    inventory_xml_path: str = "inventory.xml"
    inventory_import_xml_path: str = "1cbitrix/import.xml"
    order_to_email: str = "7450142@mail.ru"
    yandex_api_key: str = ""
    yandex_catalog_id: str = ""



def _sanitize_openai_model(raw: str) -> str:
    m = (raw or "").strip()
    if not m:
        return "gpt-4o-mini"
    # Заменим любые типы тире на обычный '-'
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "–", "—"):
        m = m.replace(ch, "-")
    m = m.lower().strip()
    # Уберём лишние пробелы, заменив их на '-'
    m = re.sub(r"\s+", "-", m)
    # Сведём повторные дефисы к одному
    m = re.sub(r"-+", "-", m)
    # Нормализуем известные алиасы
    aliases = {
        "gpt4o": "gpt-4o",
        "gpt-4-o": "gpt-4o",
        "gpt-4o": "gpt-4o",
        "gpt4o-mini": "gpt-4o-mini",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o-mini": "gpt-4o-mini",
    }
    return aliases.get(m, m)


def get_settings() -> Settings:
    model_env = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return Settings(
        bot_token=os.getenv("BOT_TOKEN", ""),
        bot_username=os.getenv("BOT_USERNAME", ""),
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=int(os.getenv("SMTP_PORT", "465")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        smtp_from=os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "")),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=_sanitize_openai_model(model_env),
        replicate_api_token=os.getenv("REPLICATE_API_TOKEN", ""),
        replicate_seg_model=os.getenv("REPLICATE_SEG_MODEL", "jagilley/grounded-sam"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "flux-pro"),
        inventory_xml_path=os.getenv("INVENTORY_XML_PATH", "inventory.xml"),
        inventory_import_xml_path=os.getenv("INVENTORY_IMPORT_XML_PATH", "1cbitrix/import.xml"),
        order_to_email=os.getenv("ORDER_TO_EMAIL", "7450142@mail.ru"),
        yandex_api_key=os.getenv("YANDEX_API_KEY", ""),
        yandex_catalog_id=os.getenv("YANDEX_CATALOG_ID", ""),
    ) 