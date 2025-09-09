from __future__ import annotations
from typing import Optional
import replicate
import requests
from openai import OpenAI
from openai import PermissionDeniedError  # type: ignore
from io import BytesIO
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import logging
import numpy as np
import cv2 as cv
import os


logger = logging.getLogger("miranda-image")

# Используем одну из моделей image-to-image/virtual staging.
DEFAULT_MODEL = "black-forest-labs/flux-fill"


def _prepare_png_upto_4mb(src_bytes: bytes, max_side: int = 1024) -> bytes:
    """Конвертирует изображение в PNG и уменьшает до тех пор, пока размер файла не станет <4 МБ."""
    img = Image.open(BytesIO(src_bytes)).convert("RGBA")
    current_max = max(img.size)
    target = min(current_max, max_side)

    while True:
        if max(img.size) > target:
            scale = target / float(max(img.size))
            new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
            img = img.resize(new_size, Image.LANCZOS)
        out = BytesIO()
        out.name = "image.png"
        img.save(out, format="PNG", optimize=True)
        data = out.getvalue()
        if len(data) < 4 * 1024 * 1024:
            return data
        # Слишком большой PNG — уменьшаем сильнее
        if target <= 512:
            return data  # уже достаточно уменьшили; вернём как есть
        target = int(target * 0.8)


def _resize_mask_to_image(mask_png_bytes: bytes, image_png_bytes: bytes) -> bytes:
    """Подгоняет маску (PNG) к размеру изображения (PNG)."""
    img = Image.open(BytesIO(image_png_bytes)).convert("RGBA")
    mask = Image.open(BytesIO(mask_png_bytes)).convert("RGBA")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.LANCZOS)
    out = BytesIO()
    out.name = "mask.png"
    mask.save(out, format="PNG")
    return out.getvalue()


def apply_texture_with_mask(room_bytes: bytes, fabric_bytes: bytes, mask_png_bytes: bytes) -> Optional[bytes]:
    try:
        room = Image.open(BytesIO(room_bytes)).convert("RGB")
        fabric = Image.open(BytesIO(fabric_bytes)).convert("RGB")
        # Маска: берём альфа‑канал и инвертируем (в нашей маске alpha=0 — редактируемая зона)
        m_img = Image.open(BytesIO(mask_png_bytes)).convert("RGBA")
        alpha = m_img.split()[-1]
        from PIL import ImageOps as _ImageOps
        paste_mask = _ImageOps.invert(alpha).convert("L")  # 255 = пастим, 0 = нет
        if fabric.size != room.size:
            fabric = fabric.resize(room.size, Image.LANCZOS)
        # Освещение/тени сохраним через умножение (multiply)
        light = ImageOps.grayscale(room).convert("RGB")
        textured = Image.blend(fabric, light, alpha=0.35)
        result = room.copy()
        result.paste(textured, mask=paste_mask)
        out = BytesIO()
        result.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        logger.exception("Texture fallback error: %s", e)
        return None


class ImageRenderer:
    def __init__(self, api_token: str, model: str = DEFAULT_MODEL) -> None:
        self.client = replicate.Client(api_token=api_token)
        self.model = model

    def render(self, room_image_url: str, fabric_image_url: str, prompt: Optional[str] = None) -> Optional[str]:
        try:
            input_payload = {
                "image": room_image_url,
                "reference_image": fabric_image_url,
                "prompt": prompt or "Apply curtain fabric texture to curtains area realistically, preserve room style, photorealistic.",
            }
            output = self.client.run(f"{self.model}", input=input_payload)
            if isinstance(output, list) and output:
                return str(output[0])
            if isinstance(output, str):
                return output
            return None
        except Exception as e:
            logger.exception("Replicate render error: %s", e)
            return None


# OpenAI-редакция изображений (gpt-image-1 edits)
class OpenAIImageEditor:
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def _do_edit(self, model: str, base_image_bytes: bytes, mask_png_bytes: Optional[bytes], prompt: str) -> Optional[bytes]:
        png_bytes = _prepare_png_upto_4mb(base_image_bytes)
        img_file = BytesIO(png_bytes)
        img_file.name = "image.png"

        params = {
            "model": model,
            "image": img_file,
            "prompt": prompt,
            "size": "1024x1024",
            # response_format не поддерживается в images API всех моделей — уберём
        }
        if mask_png_bytes:
            # Маска должна совпадать по размеру с изображением
            mask_png_bytes = _resize_mask_to_image(mask_png_bytes, png_bytes)
            mask_file = BytesIO(mask_png_bytes)
            mask_file.name = "mask.png"
            params["mask"] = mask_file

        # Новый SDK: images.edit; фолбэк на images.edits для совместимости
        if hasattr(self.client.images, "edit"):
            result = self.client.images.edit(**params)  # type: ignore[arg-type]
        else:
            result = self.client.images.edits(**params)  # type: ignore[attr-defined]

        b64 = result.data[0].b64_json  # type: ignore[attr-defined]
        import base64
        return base64.b64decode(b64)

    def edit_with_mask(self, base_image_bytes: bytes, mask_png_bytes: Optional[bytes], prompt: str) -> Optional[bytes]:
        try:
            # 1) Пытаемся gpt-image-1
            return self._do_edit("gpt-image-1", base_image_bytes, mask_png_bytes, prompt)
        except PermissionDeniedError as e:  # 403: not verified for gpt-image-1
            logger.warning("gpt-image-1 denied (will fallback to dall-e-2): %s", e)
            try:
                return self._do_edit("dall-e-2", base_image_bytes, mask_png_bytes, prompt)
            except Exception as e2:
                logger.exception("DALL-E-2 edit error: %s", e2)
                return None
        except Exception as e:
            # Любая иная ошибка — попробуем сразу DALL-E-2
            logger.warning("gpt-image-1 edit failed, trying dall-e-2: %s", e)
            try:
                return self._do_edit("dall-e-2", base_image_bytes, mask_png_bytes, prompt)
            except Exception as e2:
                logger.exception("DALL-E-2 edit error: %s", e2)
                return None


class CurtainSegmenter:
    def __init__(self, api_token: str, model: str) -> None:
        self.client = replicate.Client(api_token=api_token)
        self.model = model
        self.alternatives = [
            "cjwbw/grounded-segment-anything",
            "yahoo-inc/grounded-sam",
            "andreasjansson/grounded-sam",
        ]

    def segment_mask(self, image_url: str, text_prompt: str = "curtain") -> Optional[bytes]:
        try_models = [self.model] + [m for m in self.alternatives if m != self.model]
        last_err: Optional[Exception] = None
        for slug in try_models:
            try:
                output = self.client.run(f"{slug}", input={
                    "image": image_url,
                    "text_prompt": text_prompt,
                    "threshold": 0.3,
                    "mask_output": True,
                })
                url = None
                if isinstance(output, list) and output:
                    url = str(output[0])
                elif isinstance(output, str):
                    url = output
                if not url:
                    continue
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                img = Image.open(BytesIO(r.content)).convert("L")
                bw = img.point(lambda x: 255 if x > 127 else 0, mode='1').convert("L")
                alpha = bw.point(lambda x: 0 if x == 255 else 255)
                rgba = Image.new("RGBA", img.size, (255, 255, 255, 255))
                rgba.putalpha(alpha)
                out = BytesIO()
                rgba.save(out, format="PNG")
                return out.getvalue()
            except Exception as e:
                last_err = e
                logger.warning("Grounded-SAM '%s' failed: %s", slug, e)
                continue
        if last_err:
            logger.exception("Replicate mask error: %s", last_err)
        return None


def download_file(url: str, dest_path: str) -> None:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def to_png_mask_from_telegram(file_url: str) -> Optional[bytes]:
    try:
        r = requests.get(file_url, timeout=120)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("L")
        # Белое = область штор (редактировать), чёрное = фиксировать
        bw = img.point(lambda x: 255 if x > 127 else 0, mode='1').convert("L")
        # Для OpenAI mask: прозрачные области заменяются. Сделаем белое -> прозрачное.
        alpha = bw.point(lambda x: 0 if x == 255 else 255)
        rgba = Image.new("RGBA", img.size, (255, 255, 255, 255))
        rgba.putalpha(alpha)
        out = BytesIO()
        rgba.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        logger.exception("Mask convert error: %s", e)
        return None


class OpenRouterImageGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/images"

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Optional[bytes]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "HTTP-Referer": "miranda-bot",
                "X-Title": "Miranda Curtains Bot",
            }
            payload = {
                "model": self.model,
                "prompt": prompt,
                "size": f"{width}x{height}",
            }
            r = requests.post(self.base_url, json=payload, headers=headers, timeout=180)
            # Если сервер вернул изображение напрямую
            ctype = r.headers.get("Content-Type", "").lower()
            if ctype.startswith("image/") or ctype.startswith("application/octet-stream"):
                if r.status_code == 200 and r.content:
                    return r.content
            # Проверим статус и попробуем JSON
            r.raise_for_status()
            try:
                data = r.json()
            except Exception:
                # Попробуем альтернативный эндпоинт generations
                alt = requests.post(
                    "https://openrouter.ai/api/v1/images/generations",
                    json=payload,
                    headers=headers,
                    timeout=180,
                )
                ctype2 = alt.headers.get("Content-Type", "").lower()
                if ctype2.startswith("image/") or ctype2.startswith("application/octet-stream"):
                    if alt.status_code == 200 and alt.content:
                        return alt.content
                alt.raise_for_status()
                data = alt.json()
            # Ожидаем data -> list -> {b64_json|url}
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list) and data["data"]:
                    item = data["data"][0]
                    if isinstance(item, dict):
                        if item.get("b64_json"):
                            import base64
                            return base64.b64decode(item["b64_json"])  # type: ignore[arg-type]
                        if item.get("url"):
                            img = requests.get(item["url"], timeout=180)
                            img.raise_for_status()
                            return img.content
                if data.get("image"):
                    img = requests.get(data["image"], timeout=180)
                    img.raise_for_status()
                    return img.content
            return None
        except Exception as e:
            logger.exception("OpenRouter generate error: %s", e)
            return None


class OpenAIImageGenerator:
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        import base64
        try:
            # Пытаемся gpt-image-1
            res = self.client.images.generate(model="gpt-image-1", prompt=prompt, size=size, response_format="b64_json")
            b64 = getattr(res.data[0], "b64_json", None)  # type: ignore[attr-defined]
            if b64:
                return base64.b64decode(b64)
            # Фолбэк на URL, если провайдер вернул ссылку
            url = getattr(res.data[0], "url", None)
            if url:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                return r.content
            return None
        except PermissionDeniedError as e:
            logger.warning("gpt-image-1 denied (fallback to dall-e-2): %s", e)
            try:
                res = self.client.images.generate(model="dall-e-2", prompt=prompt, size=size, response_format="b64_json")
                b64 = getattr(res.data[0], "b64_json", None)  # type: ignore[attr-defined]
                if b64:
                    return base64.b64decode(b64)
                url = getattr(res.data[0], "url", None)
                if url:
                    r = requests.get(url, timeout=180)
                    r.raise_for_status()
                    return r.content
                return None
            except Exception as e2:
                logger.exception("DALL-E-2 generate error: %s", e2)
                return None
        except Exception as e:
            # Общая ошибка — попробуем DALL-E-2
            logger.warning("gpt-image-1 generate failed, trying dall-e-2: %s", e)
            try:
                res = self.client.images.generate(model="dall-e-2", prompt=prompt, size=size, response_format="b64_json")
                b64 = getattr(res.data[0], "b64_json", None)  # type: ignore[attr-defined]
                if b64:
                    return base64.b64decode(b64)
                url = getattr(res.data[0], "url", None)
                if url:
                    r = requests.get(url, timeout=180)
                    r.raise_for_status()
                    return r.content
                return None
            except Exception as e2:
                logger.exception("DALL-E-2 generate error: %s", e2)
                return None 


class YandexArtGenerator:
    def __init__(self, api_key: str, catalog_id: str) -> None:
        self.api_key = api_key
        self.catalog_id = catalog_id
        # Эндпоинт генерации изображений в Foundation Models (image generation)
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/imageGeneration"

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Optional[bytes]:
        try:
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "x-folder-id": self.catalog_id,
                "Content-Type": "application/json",
            }
            payload = {
                "modelUri": f"art://{self.catalog_id}/yandex-art/latest",
                "generationOptions": {"mimeType": "image/png", "seed": 0},
                "messages": [
                    {"weight": 1, "text": prompt},
                ],
                "width": width,
                "height": height,
            }
            r = requests.post(self.url, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            # Ответ может содержать base64 в поле "image" или массив данных
            # Нормализуем
            img_b64 = None
            if isinstance(data, dict):
                if data.get("result") and isinstance(data["result"], dict):
                    img_b64 = data["result"].get("image")
                if not img_b64 and data.get("image"):
                    img_b64 = data.get("image")
                if not img_b64 and data.get("images"):
                    arr = data["images"]
                    if isinstance(arr, list) and arr:
                        img_b64 = arr[0]
            if not img_b64:
                return None
            import base64
            return base64.b64decode(img_b64)
        except Exception as e:
            logger.exception("YandexART generate error: %s", e)
            return None 


def refine_curtain_mask(base_image_bytes: bytes, raw_mask_png_bytes: bytes) -> Optional[bytes]:
    try:
        base = Image.open(BytesIO(base_image_bytes)).convert("RGBA")
        raw = Image.open(BytesIO(raw_mask_png_bytes)).convert("L")
        # Бинаризуем и получаем bbox исходной области
        bw = raw.point(lambda x: 255 if x > 127 else 0).convert("L")
        bbox = bw.getbbox()
        if not bbox:
            return raw_mask_png_bytes
        x0, y0, x1, y1 = bbox
        W, H = base.size
        # Расширим по ширине и вниз до пола/подоконника (прибл.)
        pad_x = int(0.1 * (x1 - x0))
        x0 = max(0, x0 - pad_x)
        x1 = min(W, x1 + pad_x)
        # Если исходная маска короткая, опустим низ почти до пола
        y1 = max(y1, int(H * 0.92))
        # Слегка поднимем верх (карниз)
        y0 = max(0, y0 - int(0.03 * H))
        # Сформируем новую область в виде двух полотен (лево/право) при широкой зоне
        refined = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(refined)
        width = x1 - x0
        if width > int(0.35 * W):
            mid = (x0 + x1) // 2
            # Два полотна с небольшим нахлёстом по центру
            overlap = max(6, int(0.02 * W))
            draw.rectangle([x0, y0, mid + overlap, y1], fill=255)
            draw.rectangle([mid - overlap, y0, x1, y1], fill=255)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=255)
        # Сгладим края и сузим шум
        refined = refined.filter(ImageFilter.GaussianBlur(radius=2))
        refined = refined.point(lambda x: 255 if x > 160 else 0).convert("L")
        # Для API OpenAI прозрачное = редактируемая зона. Сделаем альфу 0 где белое
        alpha = refined.point(lambda x: 0 if x == 255 else 255)
        rgba = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        rgba.putalpha(alpha)
        out = BytesIO()
        rgba.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        logger.exception("refine_curtain_mask error: %s", e)
        return None 


def build_curtain_mask_cv(base_image_bytes: bytes) -> Optional[bytes]:
    try:
        data = np.frombuffer(base_image_bytes, dtype=np.uint8)
        img = cv.imdecode(data, cv.IMREAD_COLOR)
        if img is None:
            return None
        H, W = img.shape[:2]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(gray, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=int(W*0.25), maxLineGap=20)
        # Найдём вертикальные границы окна по линиям
        x_candidates = []
        y_top = int(H*0.1)
        y_bottom = int(H*0.9)
        if lines is not None:
            for l in lines[:,0,:]:
                x1,y1,x2,y2 = l
                if abs(x1-x2) < 8 and abs(y2-y1) > H*0.2:
                    x_candidates.append(int((x1+x2)//2))
        if len(x_candidates) >= 2:
            x_candidates.sort()
            x0 = x_candidates[0]
            x1 = x_candidates[-1]
            if x1 - x0 < int(W*0.2):
                x0, x1 = int(W*0.3), int(W*0.7)
        else:
            x0, x1 = int(W*0.25), int(W*0.75)
        # Верх по горизонтальным линиям/эвристике
        y0 = int(H*0.12)
        # Низ — почти до пола
        y1p = int(H*0.92)
        # Маска двух полотен
        mask = np.zeros((H, W), dtype=np.uint8)
        overlap = max(8, int(0.02*W))
        mid = (x0+x1)//2
        cv.rectangle(mask, (x0, y0), (mid+overlap, y1p), 255, -1)
        cv.rectangle(mask, (mid-overlap, y0), (x1, y1p), 255, -1)
        # Имитация вертикальных складок: эрозия узким вертикальным ядром и дилатация
        k = cv.getStructuringElement(cv.MORPH_RECT, (3, 31))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
        # Сгладим края
        mask = cv.GaussianBlur(mask, (5,5), 0)
        _, mask = cv.threshold(mask, 120, 255, cv.THRESH_BINARY)
        # Преобразуем в RGBA (прозрачное = редактируемая зона)
        alpha = np.where(mask==255, 0, 255).astype(np.uint8)
        rgba = np.dstack([np.full_like(mask, 255), np.full_like(mask, 255), np.full_like(mask, 255), alpha])
        ok, buf = cv.imencode('.png', cv.cvtColor(rgba, cv.COLOR_BGRA2RGBA))
        if not ok:
            return None
        return buf.tobytes()
    except Exception as e:
        logger.exception("build_curtain_mask_cv error: %s", e)
        return None 


def _estimate_window_box_cv(bgr: np.ndarray) -> tuple[int, int, int, int]:
    H, W = bgr.shape[:2]
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=int(W*0.25), maxLineGap=20)
    x_candidates = []
    if lines is not None:
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = l
            if abs(x1-x2) < 8 and abs(y2-y1) > H*0.2:
                x_candidates.append(int((x1+x2)//2))
    if len(x_candidates) >= 2:
        x_candidates.sort()
        x0 = x_candidates[0]
        x1 = x_candidates[-1]
        if x1 - x0 < int(W*0.2):
            x0, x1 = int(W*0.3), int(W*0.7)
    else:
        x0, x1 = int(W*0.25), int(W*0.75)
    y0 = int(H*0.12)
    y1 = int(H*0.92)
    return x0, y0, x1, y1


def build_mask_from_template(base_image_bytes: bytes, template_path: str, *, dx: int = 0, dy: int = 0, scale: float = 1.0, white_thr: int = 245, feather_px: int = 3) -> Optional[bytes]:
    try:
        if not os.path.exists(template_path):
            return None
        data = np.frombuffer(base_image_bytes, dtype=np.uint8)
        bgr = cv.imdecode(data, cv.IMREAD_COLOR)
        if bgr is None:
            return None
        H, W = bgr.shape[:2]
        x0, y0, x1, y1 = _estimate_window_box_cv(bgr)
        target_w = max(1, x1 - x0)
        target_h = max(1, y1 - y0)
        # Масштаб
        target_w = max(1, int(target_w * max(0.2, min(3.0, scale))))
        target_h = max(1, int(target_h * max(0.2, min(3.0, scale))))
        tpl = cv.imread(template_path, cv.IMREAD_UNCHANGED)
        if tpl is None:
            return None
        if tpl.shape[2] == 3:
            alpha = np.full((tpl.shape[0], tpl.shape[1]), 255, dtype=np.uint8)
            tpl_rgba = np.dstack([tpl, alpha])
        else:
            tpl_rgba = tpl
        tpl_resized = cv.resize(tpl_rgba, (target_w, target_h), interpolation=cv.INTER_AREA)
        rgb = tpl_resized[:,:,:3]
        a = tpl_resized[:,:,3]
        thr = np.uint8(white_thr)
        white = (rgb[:,:,0] >= thr) & (rgb[:,:,1] >= thr) & (rgb[:,:,2] >= thr) & (a > 10)
        mask_local = np.zeros((target_h, target_w), dtype=np.uint8)
        mask_local[white] = 255
        mask = np.zeros((H, W), dtype=np.uint8)
        # Смещение
        x0c = max(0, min(W - target_w, x0 + dx))
        y0c = max(0, min(H - target_h, y0 + dy))
        mask[y0c:y0c+target_h, x0c:x0c+target_w] = np.maximum(mask[y0c:y0c+target_h, x0c:x0c+target_w], mask_local)
        # Feather
        k = max(1, int(feather_px) * 2 + 1)
        mask = cv.GaussianBlur(mask, (k, k), 0)
        _, mask = cv.threshold(mask, 120, 255, cv.THRESH_BINARY)
        alpha = np.where(mask==255, 0, 255).astype(np.uint8)
        rgba = np.dstack([np.full_like(mask, 255), np.full_like(mask, 255), np.full_like(mask, 255), alpha])
        ok, buf = cv.imencode('.png', cv.cvtColor(rgba, cv.COLOR_BGRA2RGBA))
        if not ok:
            return None
        return buf.tobytes()
    except Exception as e:
        logger.exception("build_mask_from_template error: %s", e)
        return None 