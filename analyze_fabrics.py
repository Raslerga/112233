import os
import json
import time
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def analyze_fabric_sample(image_path: str, api_key: str) -> str | None:
    if not os.path.exists(image_path):
        return None
    try:
        client = OpenAI(api_key=api_key)
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — профессиональный текстильный дизайнер. "
                        "Опиши образец ткани максимально подробно для создания штор."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Проанализируй этот образец ткани: точные цвета, тип узора, фактура материала, "
                                "плотность, прозрачность. Используй профессиональную терминологию. "
                                "Результат должен быть готов для промпта генерации изображения."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка API: {e}")
        return None


def main() -> None:
    print("Запускаю анализ образцов ткани...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip().lower() == "your_openai_api_key_here":
        print("Не найден OpenAI API ключ в файле .env")
        print("Добавьте OPENAI_API_KEY=ВАШ_КЛЮЧ в .env")
        return
    
    os.makedirs("data", exist_ok=True)
    descriptions_file = "data/fabric_descriptions.json"
    
    descriptions: dict[str, str] = {}
    if os.path.exists(descriptions_file):
        try:
            with open(descriptions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    descriptions = {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
    
    products_dir = "products"
    analyzed_count = 0
    total_count = 0
    errors: list[str] = []
    
    articles = []
    if os.path.exists(products_dir):
        for item in os.listdir(products_dir):
            item_path = os.path.join(products_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "gallery")):
                articles.append(item)
    articles.sort()

    for article in articles:
        article_path = os.path.join(products_dir, article)
        gallery_path = os.path.join(article_path, "gallery")
        if not os.path.exists(gallery_path):
            continue
        
        print(f"Артикул: {article}")
        
        color_files = [
            f for f in os.listdir(gallery_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        color_files.sort()
        
        for color_file in color_files:
            color_name = os.path.splitext(color_file)[0]
            key = f"{article}_{color_name}"
            
            if key in descriptions:
                print(f"  Пропуск: {color_name} (уже проанализирован)")
                continue
            
            total_count += 1
            print(f"  Анализ: {color_name}")
            
            image_path = os.path.join(gallery_path, color_file)
            description = analyze_fabric_sample(image_path, api_key)
            
            if description:
                descriptions[key] = description
                analyzed_count += 1
                print(f"    Сохранено ({len(description)} символов)")
                try:
                    with open(descriptions_file, "w", encoding="utf-8") as f:
                    json.dump(descriptions, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"    Ошибка сохранения: {e}")
            else:
                errors.append(key)
                print("    Ошибка анализа")
            
            time.sleep(2)
    
    print("Итоги:")
    print(f"  Проанализировано: {analyzed_count}")
    print(f"  Новых образцов: {total_count}")
    print(f"  Ошибок: {len(errors)}")
    if errors:
        print("Первые ошибки:")
        for err in errors[:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
