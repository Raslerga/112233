from __future__ import annotations
from typing import List, Dict, Optional
from openai import OpenAI
import base64
import os


SYSTEM_PROMPT = (
    "Ты — Миранда, профессиональный текстильный декоратор компании \"Компания Миранда\". "
    "Отвечай дружелюбно и по делу, на русском языке. "
    "Даёшь практические советы по текстилю, шторам, карнизам, тканям, монтажу и обслуживанию, "
    "а также помогаешь дизайнерам и декораторам в их задачах."
)

# Профессиональные стили штор (ключи — внутренние идентификаторы)
STYLE_DESCRIPTIONS: Dict[str, str] = {
    "римские": "элегантные римские шторы с горизонтальными плиссированными складками, подъемный механизм, современный минималистичный дизайн",
    "классическая": "классические портьеры на всю ширину проёма, чёткая вертикальная драпировка, традиционный интерьер",
    "ламбрекен": "портьеры с декоративным ламбрекеном, сложная верхняя драпировка, выразительный декор",
    "две_по_бокам": "двухстворчатые портьеры, симметрично расположенные по бокам окна, возможны подхваты",
}


class AIClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, history: List[Dict[str, str]], user_message: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
            {"role": "user", "content": user_message}
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""

    def _describe_image(self, image_url: str, instruction: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()

    def describe_room_for_prompt(self, room_image_url: str) -> str:
        instruction = (
            "Опиши эту комнату для генерации изображения: стиль интерьера, цветовая гамма, освещение, направление света, "
            "ракурс/точка обзора, тип окна (размер, створки), наличие карниза, визуальная высота потолка, "
            "наличие откосов и подоконника, элементы вокруг окна. Кратко 2–4 предложения."
        )
        return self._describe_image(room_image_url, instruction)

    def describe_fabric(self, fabric_image_url: str) -> str:
        instruction = (
            "Опиши образец ткани: основные цвета, характер рисунка/раппорта (полоса, геометрия, растительный и т.п.), масштаб, "
            "фактура (матовая/глянцевая, бархат/лён/сатин и т.д.), плотность/прозрачность, направление рисунка. "
            "Кратко 1–2 предложения."
        )
        return self._describe_image(fabric_image_url, instruction) 

    def analyze_fabric_sample(self, fabric_image_path: str) -> str:
        """Анализирует локальный файл образца ткани и возвращает детальное описание."""
        if not os.path.exists(fabric_image_path):
            return "Файл образца не найден"
        with open(fabric_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"
        instruction = (
            "Проанализируй образец ткани максимально подробно для использования в генерации изображения. "
            "Укажи точные оттенки, тип рисунка (если есть) и его масштаб, фактуру (блеск/матовость, ворс), "
            "плотность/прозрачность, тип переплетения при наличии визуальных признаков, особенности текстуры. "
            "Пиши кратко, профессионально, без воды."
        )
        return self._describe_image(image_url, instruction)

    def generate_curtain_visualization(
        self,
        curtain_style: str,
        room_description: str,
        fabric_description: str,
        article: str,
        color: str,
    ) -> Optional[bytes]:
        """Генерирует изображение штор через DALL·E 2 и возвращает bytes (PNG/JPG)."""
        style_desc = STYLE_DESCRIPTIONS.get(curtain_style, "элегантные шторы")

        def _trim(txt: str, max_len: int) -> str:
            t = (txt or "").strip()
            if len(t) <= max_len:
                return t
            # обрезаем по слову
            cut = t[: max_len].rsplit(" ", 1)[0]
            return (cut or t[: max_len]).strip()

        # Базовые лимиты для секций
        room_part = _trim(room_description, 350)
        fabric_part = _trim(fabric_description, 350)
        fixed_part = f"Коллекция NUR, артикул {article}, цвет {color}."
        intro_part = f"Фотореалистичный интерьер: {room_part}. На окне — {style_desc}. Ткань: {fabric_part}. "
        prompt = intro_part + fixed_part + " Естественное освещение, реалистичные тени, детализированная драпировка."

        # DALL·E 2 ограничение ~1000 символов: перестрахуемся до 950
        MAX_PROMPT = 950
        if len(prompt) > MAX_PROMPT:
            # Сжимаем описания поэтапно
            room_part = _trim(room_part, 250)
            fabric_part = _trim(fabric_part, 300)
            prompt = (
                f"Фотореалистичный интерьер: {room_part}. На окне — {style_desc}. "
                f"Ткань: {fabric_part}. {fixed_part} Естественное освещение, реалистичные тени."
            )
        if len(prompt) > MAX_PROMPT:
            room_part = _trim(room_part, 180)
            fabric_part = _trim(fabric_part, 220)
            prompt = (
                f"Интерьер: {room_part}. {style_desc}. Ткань: {fabric_part}. {fixed_part}"
            )
        if len(prompt) > MAX_PROMPT:
            # Финальная гарантированная обрезка
            prompt = prompt[:MAX_PROMPT].rsplit(" ", 1)[0]

        try:
            # Предпочтительно получить base64, чтобы не ходить по URL
            res = self.client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="1024x1024",
                response_format="b64_json",
            )
            b64 = getattr(res.data[0], "b64_json", None)  # type: ignore[attr-defined]
            if b64:
                import base64 as _b64
                return _b64.b64decode(b64)
            # Фолбэк на URL, если провайдер вернул ссылку
            url = getattr(res.data[0], "url", None)
            if url:
                import requests as _r
                rr = _r.get(url, timeout=180)
                rr.raise_for_status()
                return rr.content
            return None
        except Exception as e:
            print(f"Ошибка DALL·E 2: {e}")
            return None
