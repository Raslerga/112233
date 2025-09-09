from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from lxml import etree
from rapidfuzz import process, fuzz
from unidecode import unidecode


@dataclass
class Item:
    article: str
    color: str
    name: str
    price_per_meter: float
    quantity_m: float


class Inventory:
    def __init__(self, xml_path: str, import_xml_path: Optional[str] = None) -> None:
        self.xml_path = xml_path
        self.import_xml_path = import_xml_path
        self.items: List[Item] = []
        # Ключом служит нормализованный артикул или наименование
        self.by_article: Dict[str, List[Item]] = {}
        self.article_keys: List[str] = []
        self.unique_colors_by_article: Dict[str, List[str]] = {}
        # Маппинг штрихкод -> цвет (из import.xml)
        self._barcode_to_color: Dict[str, str] = {}
        self._load()

    @staticmethod
    def _norm(s: str) -> str:
        s = (s or "").strip()
        return unidecode(s).lower()

    def _get_text(self, node, path_list: List[str]) -> str:
        for p in path_list:
            found = node.findtext(p)
            if found:
                return found.strip()
        return ""

    def _load_import_colors(self) -> None:
        if not self.import_xml_path:
            return
        try:
            tree = etree.parse(self.import_xml_path)
            root = tree.getroot()
            # Ищем товары в import.xml
            goods = root.findall('.//Товар')
            for g in goods:
                barcode = (g.findtext('Штрихкод') or '').strip()
                if not barcode:
                    continue
                # Цвет может быть как человекочитаемым, так и кодом (1,2,3...)
                color_val = ""
                for prop in g.findall('ХарактеристикиТовара/ХарактеристикаТовара'):
                    name = prop.findtext('Наименование') or ''
                    if self._norm(name) == self._norm('Цвет'):
                        color_val = (prop.findtext('Значение') or '').strip()
                        break
                if color_val:
                    self._barcode_to_color[barcode] = color_val
        except Exception:
            # Тихо пропускаем, чтобы не падал весь инвентарь
            pass

    def _extract_color(self, offer) -> str:
        # Сначала пробуем извлечь цвет из предложения (если там уже лежит человекочитаемое значение)
        for prop in offer.findall("ХарактеристикиТовара/ХарактеристикаТовара"):
            name = prop.findtext("Наименование") or ""
            if self._norm(name) == self._norm("Цвет"):
                val = prop.findtext("Значение") or ""
                if val:
                    return val.strip()
        # Альтернативные варианты (редко): ЗначенияСвойств/ЗначенияСвойства
        for prop in offer.findall("ЗначенияСвойств/ЗначенияСвойства"):
            name = prop.findtext("Наименование") or ""
            if self._norm(name) == self._norm("Цвет"):
                val = prop.findtext("Значение") or ""
                if val:
                    return val.strip()
        # Если нет — вернём пусто, позже попробуем по штрихкоду из import.xml
        return ""

    def _load(self) -> None:
        # Предварительно загрузим карту штрихкодов в цвета из import.xml
        self._load_import_colors()

        tree = etree.parse(self.xml_path)
        root = tree.getroot()
        # КоммерческаяИнформация/ПакетПредложений/Предложения/Предложение
        offers = root.findall(".//Предложения/Предложение")
        for offer in offers:
            article = self._get_text(offer, ["Артикул", "Article", "art"]) or ""
            name = self._get_text(offer, ["Наименование", "Name"]) or ""
            # Цена — берём первую <Цены>/<Цена>/<ЦенаЗаЕдиницу>
            price_str = ""
            prices = offer.find("Цены")
            if prices is not None:
                price_el = prices.find("Цена")
                if price_el is not None:
                    price_str = self._get_text(price_el, ["ЦенаЗаЕдиницу", "Цена", "PricePerMeter", "price"]) or "0"
            qty_str = self._get_text(offer, ["Количество", "Остаток", "Quantity", "qty"]) or "0"
            color = self._extract_color(offer)
            # Если цвета нет в offers — пытаемся по штрихкоду найти в import.xml
            if not color:
                barcode = (offer.findtext('Штрихкод') or '').strip()
                if barcode and barcode in self._barcode_to_color:
                    color = self._barcode_to_color[barcode]
            try:
                price_v = float(str(price_str).replace(",", "."))
            except Exception:
                price_v = 0.0
            try:
                qty_v = float(str(qty_str).replace(",", "."))
            except Exception:
                qty_v = 0.0
            if not (article or name):
                continue
            item = Item(article=article, color=color, name=name, price_per_meter=price_v, quantity_m=qty_v)
            self.items.append(item)
        # Индексация по артикулу и по наименованию
        for it in self.items:
            keys = set()
            if it.article:
                keys.add(self._norm(it.article))
            if it.name:
                keys.add(self._norm(it.name))
            for key in keys:
                self.by_article.setdefault(key, []).append(it)
        self.article_keys = list(self.by_article.keys())
        for k, lst in self.by_article.items():
            colors = sorted({it.color for it in lst if it.color})
            self.unique_colors_by_article[k] = colors

    def search_article(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        q = self._norm(query)
        res = process.extract(q, self.article_keys, scorer=fuzz.WRatio, limit=limit)
        return [(cand, score) for cand, score, _ in res]

    def get_colors(self, article_key: str) -> List[str]:
        return self.unique_colors_by_article.get(article_key, [])

    def find_item(self, article_key: str, color_query: str) -> Optional[Item]:
        # Если цветов нет, вернём лучшее совпадение по ключу
        candidates = self.by_article.get(article_key, [])
        if not candidates:
            return None
        colors = self.get_colors(article_key)
        if not colors:
            # Вернём позицию с максимальным остатком
            return max(candidates, key=lambda x: x.quantity_m)
        cq = self._norm(color_query)
        match = process.extractOne(cq, [self._norm(c) for c in colors], scorer=fuzz.WRatio)
        if not match:
            return None
        norm_color = match[0]
        for it in candidates:
            if self._norm(it.color) == norm_color:
                return it
        return None 