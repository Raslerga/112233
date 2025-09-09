import os
import json
from typing import Dict, List, Optional
from unidecode import unidecode


class FabricCollection:
    """Управляет коллекцией тканей NUR и кешем описаний образцов.

    - Артикулы берутся из каталога products/<article>
    - Галерея цветов — из products/<article>/gallery
    - Описания сохраняются в data/fabric_descriptions.json
    """

    def __init__(self, products_dir: str = "products", descriptions_file: str = "data/fabric_descriptions.json") -> None:
        self.products_dir = products_dir
        self.descriptions_file = descriptions_file
        self.descriptions: Dict[str, str] = self._load_descriptions()

    def _load_descriptions(self) -> Dict[str, str]:
        if os.path.exists(self.descriptions_file):
            try:
                with open(self.descriptions_file, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                    if isinstance(data, dict):
                        return {str(k): str(v) for k, v in data.items()}
            except Exception:
                pass
        return {}

    def _save_descriptions(self) -> None:
        os.makedirs(os.path.dirname(self.descriptions_file) or ".", exist_ok=True)
        with open(self.descriptions_file, "w", encoding="utf-8") as f:
            json.dump(self.descriptions, f, ensure_ascii=False, indent=2)

    def get_articles(self) -> List[str]:
        if not os.path.exists(self.products_dir):
            return []
        articles: List[str] = []
        for item in os.listdir(self.products_dir):
            item_path = os.path.join(self.products_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "gallery")):
                articles.append(item)
        return sorted(articles)

    def get_colors(self, article: str) -> List[str]:
        gallery_path = os.path.join(self.products_dir, article, "gallery")
        if not os.path.exists(gallery_path):
            return []
        colors: List[str] = []
        for file in os.listdir(gallery_path):
            lf = file.lower()
            if lf.endswith(".jpg") or lf.endswith(".jpeg") or lf.endswith(".png"):
                colors.append(os.path.splitext(file)[0])
        return sorted(colors)

    def get_fabric_image_path(self, article: str, color: str) -> Optional[str]:
        gallery_path = os.path.join(self.products_dir, article, "gallery")
        if not os.path.exists(gallery_path):
            return None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(gallery_path, f"{color}{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    def get_fabric_description(self, article: str, color: str) -> Optional[str]:
        return self.descriptions.get(f"{article}_{color}")

    def set_fabric_description(self, article: str, color: str, description: str) -> None:
        self.descriptions[f"{article}_{color}"] = description.strip()
        self._save_descriptions()

    def has_fabric_description(self, article: str, color: str) -> bool:
        return f"{article}_{color}" in self.descriptions

    def _norm(self, s: str) -> str:
        return unidecode((s or "").strip()).lower()

    def search_articles(self, query: str) -> List[str]:
        q = self._norm(query)
        if not q:
            return []
        arts = self.get_articles()
        # точное нормализованное совпадение
        for a in arts:
            if self._norm(a) == q:
                return [a]
        # частичное по нормализованной строке
        res = [a for a in arts if q in self._norm(a)]
        return res[:20]

    def search_colors(self, article: str, query: str) -> List[str]:
        q = self._norm(query)
        if not q:
            return []
        colors = self.get_colors(article)
        for c in colors:
            if self._norm(c) == q:
                return [c]
        return [c for c in colors if q in self._norm(c)][:20]
