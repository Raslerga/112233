# Бот ИИ «Компания Миранда» — авторизация по ИНН

## Быстрый старт (Windows PowerShell)
1. Установите Python 3.10+
2. Создайте и активируйте окружение и установите зависимости:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python -m pip install --upgrade pip
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```
3. Создайте файл `.env` по образцу `.env.example` и заполните переменные:
   - BOT_TOKEN — токен Telegram-бота
   - BOT_USERNAME — имя пользователя бота без `@` (например, MyMirandaBot)
   - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM — параметры почты Beget
4. Убедитесь, что файл `иннпочты.xlsx` лежит в корне проекта.
5. Запуск бота:
   ```powershell
   .\.venv\Scripts\python src\bot.py
   ```

## Что делает бот сейчас
- Приветствует пользователя и просит ввести ИНН.
- Проверяет ИНН по `иннпочты.xlsx`.
- Если найден — отправляет на email ссылку подтверждения вида `https://t.me/<BOT_USERNAME>?start=CONFIRM_<TOKEN>`.
- По переходу по ссылке подтверждает пользователя и приветствует.

## Файлы
- `src/bot.py` — логика бота
- `src/excel_loader.py` — загрузка Excel и поиск email по ИНН
- `src/email_sender.py` — отправка писем через SMTP (Beget)
- `src/storage.py` — простое файловое хранилище токенов/пользователей
- `src/config.py` — загрузка настроек из `.env` 