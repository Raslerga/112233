import logging
from typing import Optional
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InputFile, BotCommand, MenuButtonCommands, MenuButtonDefault, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
# from telegram.ext import JobQueue
from telegram.constants import ChatAction
from telegram.error import BadRequest
from unidecode import unidecode

from config import get_settings
from excel_loader import find_email_by_inn
from email_sender import EmailSender
from storage import Storage
from ai_client import AIClient
# Удалены импорты, связанные с визуализацией
from inventory import Inventory
from fabric_collection import FabricCollection

from io import BytesIO
from PIL import Image
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("miranda-bot")
# Снизим шум от сторонних библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)


async def _set_commands_for_authorized(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    await context.bot.set_my_commands(
        [
            BotCommand("start", "Запустить бота"),
            BotCommand("ai", "Поговорить с ИИ"),
            BotCommand("inventory", "Наличие и цена"),
            BotCommand("order", "Оформить заказ"),
            BotCommand("exit", "Назад в меню"),
            BotCommand("help", "Помощь"),
        ],
        scope=None,
        language_code=None,
    )
    try:
        await context.bot.set_chat_menu_button(chat_id=chat_id, menu_button=MenuButtonCommands())
    except Exception:
        pass


async def _clear_commands_for_unauthorized(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    await context.bot.set_my_commands(commands=[])
    try:
        await context.bot.set_chat_menu_button(chat_id=chat_id, menu_button=MenuButtonDefault())
    except Exception:
        pass


async def _inc_sent(context: ContextTypes.DEFAULT_TYPE, n: int = 1) -> None:
    stats = context.application.bot_data.setdefault("stats", {"received": 0, "sent": 0})
    stats["sent"] = int(stats.get("sent", 0)) + n


# Хелперы-обёртки
async def _reply_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs):
    m = await update.effective_message.reply_text(text, **kwargs)
    await _inc_sent(context)
    return m

async def _send_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs):
    m = await context.bot.send_message(chat_id=update.effective_chat.id, text=text, **kwargs)
    await _inc_sent(context)
    return m

async def _reply_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
    m = await update.effective_message.reply_photo(*args, **kwargs)
    await _inc_sent(context)
    return m


def _to_telegram_jpeg(img_bytes: bytes) -> BytesIO:
    img = Image.open(BytesIO(img_bytes))
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    bio = BytesIO()
    bio.name = "result.jpg"
    img.save(bio, format="JPEG", quality=90, optimize=True)
    bio.seek(0)
    return bio


async def _mark_active(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return
    app = context.application
    active: dict = app.bot_data.setdefault("active_users", {})
    active[user.id] = {"username": user.username or "", "first_name": user.first_name or "", "last": time.time()}
    # Учёт статистики полученных сообщений
    stats = app.bot_data.setdefault("stats", {"received": 0, "sent": 0})
    stats["received"] = int(stats.get("received", 0)) + 1
    # Троттлинг логирования: не чаще раза в 60 секунд
    now = time.time()
    last_log = app.bot_data.get("active_last_log", 0.0)
    if now - last_log >= 60.0:
        window = 180.0  # 3 минуты
        online = [uid for uid, info in active.items() if now - info.get("last", 0) <= window]
        online_count = len(online)
        sent = int(stats.get("sent", 0))
        recv = int(stats.get("received", 0))
        logger.info("status=online active_3m=%d received=%d sent=%d", online_count, recv, sent)
        app.bot_data["active_last_log"] = now


INTRO_TEXT = (
    "Здравствуйте! Я — Искусственный интеллект ООО \"Компания Миранда\".\n"
    "Создан быть помощником для текстильных декораторов и дизайнеров — клиентов компании."
)

MAIN_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("Поговорить с ИИ"), KeyboardButton("Наличие и цена")], [KeyboardButton("Оформить заказ")]], resize_keyboard=True
)

UNAUTH_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("Поговорить с ИИ"), KeyboardButton("Авторизация")]], resize_keyboard=True
)

BACK_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("Назад в меню")]], resize_keyboard=True
)

ADD_OR_SEND_KB = ReplyKeyboardMarkup(
    [[KeyboardButton("Добавить ещё"), KeyboardButton("Отправить заказ")], [KeyboardButton("Назад в меню")]], resize_keyboard=True
)

VIZ_START_KB = ReplyKeyboardMarkup(
    [[KeyboardButton("Начать визуализацию")], [KeyboardButton("Назад в меню")]], resize_keyboard=True
)

VIZ_STYLE_KB = ReplyKeyboardMarkup(
    [[KeyboardButton("Римские"), KeyboardButton("Классическая 1 штора")], [KeyboardButton("С ламбрекеном"), KeyboardButton("Две шторы по бокам")], [KeyboardButton("Назад в меню")]], resize_keyboard=True
)


async def cmd_exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Сброс любых режимов и возвращение в главное меню
    context.user_data.clear()
    storage: Storage = context.bot_data["storage"]
    if storage.is_authorized(update.effective_user.id):
        await _reply_text(update, context, "Вы вышли в главное меню.", reply_markup=ReplyKeyboardRemove())
    else:
        await update.effective_message.reply_text("Вы в меню.", reply_markup=UNAUTH_MENU)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = get_settings()
    user = update.effective_user
    args = context.args or []

    # Обработка подтверждения по deep-link: /start CONFIRM_<TOKEN>
    if args and args[0].startswith("CONFIRM_"):
        token = args[0].split("CONFIRM_", 1)[1]
        storage: Storage = context.bot_data["storage"]
        pending = storage.pop_pending(token)
        if pending is None:
            await update.effective_message.reply_text(
                "Ссылка недействительна или уже была использована. Если проблема повторяется, свяжитесь с менеджером."
            )
            return
        if pending.user_id != user.id:
            await update.effective_message.reply_text(
                "Ссылка подтверждения привязана к другому пользователю Telegram. Пожалуйста, запросите новую."
            )
            return

        storage.mark_authorized(user.id, pending.inn, pending.email)
        # Включаем команды и кнопку «Меню» после успешной авторизации
        await _set_commands_for_authorized(context, update.effective_chat.id)
        await update.effective_message.reply_text(
            "Авторизация подтверждена! Добро пожаловать. Можете пробовать функции бота.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return

    # Обычный старт
    storage: Storage = context.bot_data["storage"]
    if storage.is_authorized(user.id):
        await _set_commands_for_authorized(context, update.effective_chat.id)
        await update.effective_message.reply_text(
            "Вы уже авторизованы. Можете пробовать функции бота.", reply_markup=ReplyKeyboardRemove()
        )
        return

    # Неавторизован: очистим команды и покажем два варианта
    await _clear_commands_for_unauthorized(context, update.effective_chat.id)
    await update.effective_message.reply_text(
        INTRO_TEXT + "\n\nВыберите действие:", reply_markup=UNAUTH_MENU
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _mark_active(update, context)
    settings = get_settings()
    storage: Storage = context.bot_data["storage"]

    # Ранняя обработка команд из кнопки «Меню»
    text_in = (update.effective_message.text or "").strip()
    low = text_in.lower()
    if low in {"/ai", "/inventory", "/order", "/exit", "/help"}:
        if low == "/exit":
            await cmd_exit(update, context)
            return
        if low == "/help":
            await update.effective_message.reply_text(
                "Доступные команды:\n/ai — чат с ИИ\n/inventory — наличие и цена\n/order — оформить заказ\n/exit — назад в меню"
            )
            return
        if low == "/ai":
            ai: Optional[AIClient] = context.bot_data.get("ai_client")
            if ai is None:
                await update.effective_message.reply_text("ИИ недоступен.")
                return
            context.user_data["chat_with_ai"] = True
            await update.effective_message.reply_text(
                "Режим чата с Мирандой. Напишите свой вопрос. Для выхода — /exit или кнопка \"Назад в меню\".",
                reply_markup=BACK_MENU,
            )
            return
        # inventory/order только для авторизованных
        if not storage.is_authorized(update.effective_user.id):
            await update.effective_message.reply_text(
                "Эта команда доступна после авторизации. Нажмите \"Авторизация\" в меню.",
                reply_markup=UNAUTH_MENU,
            )
            return
        if low == "/inventory":
            if context.bot_data.get("inventory") is None:
                try:
                    context.bot_data["inventory"] = Inventory(settings.inventory_xml_path, settings.inventory_import_xml_path)
                except Exception as e:
                    logger.exception("Inventory load error: %s", e)
                    await update.effective_message.reply_text("Не удалось загрузить складскую выгрузку.")
                    return
            context.user_data["inv_mode"] = True
            context.user_data["inv_step"] = "await_article"
            await update.effective_message.reply_text(
                "Раздел: Наличие и цена. Напишите артикул товара (можно приблизительно).",
                reply_markup=BACK_MENU,
            )
            return
        if low == "/order":
            context.user_data["order_mode"] = True
            context.user_data["order_step"] = "await_article"
            context.user_data.setdefault("order_items", [])
            await update.effective_message.reply_text(
                "Раздел: Оформить заказ. Введите артикул товара.", reply_markup=BACK_MENU
            )
            return

    # Режим чата с ИИ (разрешён всем)
    if context.user_data.get("chat_with_ai"):
        text = text_in
        if text.lower() in {"/exit", "назад", "в меню", "назад в меню"}:
            context.user_data["chat_with_ai"] = False
            if storage.is_authorized(update.effective_user.id):
                await update.effective_message.reply_text("Вы в главном меню.", reply_markup=ReplyKeyboardRemove())
            else:
                await update.effective_message.reply_text("Вы в меню.", reply_markup=UNAUTH_MENU)
            return
        history = storage.get_history(update.effective_user.id, limit=20)
        ai: AIClient = context.bot_data["ai_client"]
        storage.append_history(update.effective_user.id, "user", text)
        thinking = None
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            thinking = await update.effective_message.reply_text("Думаю…", reply_markup=BACK_MENU)
            reply = ai.generate(history, text)
            try:
                await thinking.edit_text(reply)
            except BadRequest:
                try:
                    await thinking.delete()
                except Exception:
                    pass
                await update.effective_message.reply_text(reply, reply_markup=BACK_MENU)
        except Exception as e:
            logger.exception("AI error: %s", e)
            reply = "Извините, не удалось получить ответ от ИИ. Попробуйте позже."
            if thinking is not None:
                try:
                    await thinking.delete()
                except Exception:
                    pass
            await update.effective_message.reply_text(reply, reply_markup=BACK_MENU)
        storage.append_history(update.effective_user.id, "assistant", reply)
        return

    # Режим «Наличие и цена»
    if context.user_data.get("inv_mode"):
        inv: Inventory = context.bot_data.get("inventory")
        if inv is None:
            await update.effective_message.reply_text("Склад временно недоступен.")
            context.user_data.clear()
            return
        user_text = text_in
        if user_text.lower() in {"/exit", "назад", "в меню", "назад в меню"}:
            context.user_data.clear()
            await update.effective_message.reply_text("Вы в главном меню.", reply_markup=ReplyKeyboardRemove())
            return
        step = context.user_data.get("inv_step")
        if step == "await_article":
            q = user_text
            # 1) Точное совпадение по ключу (нормализованно)
            q_key = unidecode((q or "").strip()).lower()
            if q_key in inv.by_article:
                context.user_data["inv_article_key"] = q_key
                await update.effective_message.reply_text(
                    "Какой цвет вас интересует? Укажите название или числовое обозначение цвета.",
                    reply_markup=BACK_MENU,
                )
                context.user_data["inv_step"] = "await_color"
                return
            # 2) Нечёткий поиск
            matches = inv.search_article(q, limit=20)
            if not matches:
                await update.effective_message.reply_text("Не нашёл похожих артикулов. Повторите запрос.", reply_markup=BACK_MENU)
                return
            # Уберём возможные дубликаты ключей, сохраняя порядок
            seen = set()
            dedup_matches = []
            for cand, score in matches:
                if cand in seen:
                    continue
                seen.add(cand)
                dedup_matches.append((cand, score))
            # Если один явный вариант или очень высокое совпадение — берём сразу
            if len(dedup_matches) == 1 or (dedup_matches[0][1] >= 95 and (len(dedup_matches) == 1 or (len(dedup_matches) > 1 and dedup_matches[1][1] <= dedup_matches[0][1] - 10))):
                best_key = dedup_matches[0][0]
                context.user_data["inv_article_key"] = best_key
                await update.effective_message.reply_text(
                    "Какой цвет вас интересует? Укажите название или числовое обозначение цвета.",
                    reply_markup=BACK_MENU,
                )
                context.user_data["inv_step"] = "await_color"
                return
            # Иначе предложим выбор
            choices = [m[0] for m in dedup_matches]
            context.user_data["inv_article_choices"] = choices
            lines = []
            for i, key in enumerate(choices):
                items = inv.by_article.get(key, [])
                label = ""
                if items:
                    label = items[0].article or items[0].name or key
                else:
                    label = key
                lines.append(f"{i+1}. {label}")
            await update.effective_message.reply_text(
                "Нашлось несколько вариантов. Выберите номер артикула:\n" + "\n".join(lines),
                reply_markup=BACK_MENU,
            )
            context.user_data["inv_step"] = "choose_article"
            return
        if step == "choose_article":
            choices = context.user_data.get("inv_article_choices", [])
            if not choices:
                context.user_data["inv_step"] = "await_article"
                await update.effective_message.reply_text("Введите артикул товара ещё раз.", reply_markup=BACK_MENU)
                return
            sel_key = None
            user_text = (update.effective_message.text or "").strip()
            if user_text.isdigit():
                idx = int(user_text) - 1
                if 0 <= idx < len(choices):
                    sel_key = choices[idx]
            if sel_key is None:
                try:
                    from rapidfuzz import process, fuzz
                except Exception:
                    process = None
                labels = []
                for key in choices:
                    items = inv.by_article.get(key, [])
                    label = items[0].article or items[0].name if items else key
                    labels.append(label)
                if process is not None:
                    match = process.extractOne(user_text, labels, scorer=fuzz.WRatio)
                    if match:
                        sel_idx = labels.index(match[0])
                        if 0 <= sel_idx < len(choices):
                            sel_key = choices[sel_idx]
            if sel_key is None:
                await update.effective_message.reply_text("Не понял выбор. Укажите номер из списка.", reply_markup=BACK_MENU)
                return
            context.user_data["inv_article_key"] = sel_key
            context.user_data.pop("inv_article_choices", None)
            await update.effective_message.reply_text(
                "Какой цвет вас интересует? Укажите название или числовое обозначение цвета.",
                reply_markup=BACK_MENU,
            )
            context.user_data["inv_step"] = "await_color"
            return
        if step == "await_color":
            best_key = context.user_data.get("inv_article_key")
            color_in = user_text
            item = inv.find_item(best_key, color_in)
            if not item:
                await update.effective_message.reply_text(
                    "Такого цвета для этого артикула не найдено. Проверьте написание или введите другой цвет (слово или число).",
                    reply_markup=BACK_MENU,
                )
                return
            await update.effective_message.reply_text(
                f"Артикул: {item.article}\nНаименование: {item.name}\nЦвет: {item.color or '-'}\n" \
                f"Цена за метр: {item.price_per_meter:.2f}\nОстаток: {item.quantity_m:.2f} м"
            )
            context.user_data.clear()
            await update.effective_message.reply_text("Готово. Вы в главном меню.", reply_markup=ReplyKeyboardRemove())
            return

    # Режим «Оформить заказ»
    if context.user_data.get("order_mode"):
        user_text = text_in
        if user_text.lower() in {"/exit", "назад", "в меню", "назад в меню"}:
            context.user_data.clear()
            await update.effective_message.reply_text("Вы в главном меню.", reply_markup=ReplyKeyboardRemove())
            return

        step = context.user_data.get("order_step")
        storage: Storage = context.bot_data["storage"]
        user_info = storage.get_user(update.effective_user.id) or {}
        order_email = settings.order_to_email
        sender: EmailSender = context.bot_data["email_sender"]

        context.user_data.setdefault("order_items", [])

        if step == "await_article":
            context.user_data["order_article"] = user_text
            context.user_data["order_step"] = "await_color"
            await update.effective_message.reply_text("Введите цвет (слово или цифра).", reply_markup=BACK_MENU)
            return

        if step == "await_color":
            context.user_data["order_color"] = user_text
            context.user_data["order_step"] = "await_qty"
            await update.effective_message.reply_text("Введите количество (в метрах).", reply_markup=BACK_MENU)
            return

        if step == "await_qty":
            context.user_data["order_qty"] = user_text
            context.user_data["order_step"] = "await_panels"
            await update.effective_message.reply_text(
                "Разбивка полотен: укажите, например, 'одно полотно' или 'две панели по 1,5 м'.",
                reply_markup=BACK_MENU,
            )
            return

        if step == "await_panels":
            context.user_data["order_panels"] = user_text
            item = {
                "article": context.user_data.get("order_article", "-"),
                "color": context.user_data.get("order_color", "-"),
                "qty": context.user_data.get("order_qty", "-"),
                "panels": context.user_data.get("order_panels", "одно полотно"),
            }
            context.user_data["order_items"].append(item)
            for k in ("order_article", "order_color", "order_qty", "order_panels"):
                context.user_data.pop(k, None)
            context.user_data["order_step"] = "await_add_or_send"
            lines = [
                f"{i+1}) {it['article']} | цвет: {it['color']} | м: {it['qty']} | полотна: {it.get('panels','-')}"
                for i, it in enumerate(context.user_data["order_items"])
            ]
            await update.effective_message.reply_text(
                "Позиция добавлена. Текущий заказ:\n" + "\n".join(lines) + "\n\nДобавить ещё позицию или отправить заказ?",
                reply_markup=ADD_OR_SEND_KB,
            )
            return

        if step == "await_add_or_send":
            low = user_text.lower()
            if low in {"добавить ещё", "добавить еще"}:
                context.user_data["order_step"] = "await_article"
                await update.effective_message.reply_text("Введите артикул следующей позиции.", reply_markup=BACK_MENU)
                return
            if low in {"отправить заказ", "отправить"}:
                if not context.user_data.get("order_transport"):
                    context.user_data["order_step"] = "await_transport_final"
                    await update.effective_message.reply_text(
                        "Укажите транспорт: название ТК (Деловые линии, ПЭК, СДЭК), или напишите 'самовывоз' или 'доставка'.",
                        reply_markup=BACK_MENU,
                    )
                    return
                # если уже указан транспорт — сразу отправляем
                low = "готово"
                context.user_data["order_step"] = "await_transport_final"

        if step == "await_transport_final":
            if user_text.strip():
                # сохранить при первом вводе
                if not context.user_data.get("order_transport") and user_text.lower() not in {"готово"}:
                    context.user_data["order_transport"] = user_text.strip()
            items = context.user_data.get("order_items", [])
            if not items:
                await update.effective_message.reply_text("Нет позиций для отправки. Добавьте хотя бы одну.", reply_markup=BACK_MENU)
                context.user_data["order_step"] = "await_article"
                return
            inn = user_info.get("inn", "-")
            email = user_info.get("email", "-")
            subject = "Новый заказ из Telegram-бота"
            body_lines = [
                "Поступил новый заказ из бота.",
                "",
                f"ИНН клиента: {inn}",
                f"Email клиента: {email}",
                f"Доставка/ТК: {context.user_data.get('order_transport','-')}",
                "",
                "Позиции:",
            ]
            for i, it in enumerate(items, 1):
                body_lines.append(
                    f"{i}) Артикул: {it['article']} | Цвет: {it['color']} | Кол-во (м): {it['qty']} | полотна: {it.get('panels','одно полотно')}"
                )
            body = "\n".join(body_lines)
            try:
                sender.send_text(order_email, subject, body)
                await update.effective_message.reply_text("Заказ отправлен менеджеру. Мы свяжемся с вами.")
            except Exception as e:
                await update.effective_message.reply_text(
                    "Не удалось отправить заказ. Проверьте настройки почты (SMTP хост/порт/логин/пароль).\n"
                    f"Техническая причина: {type(e).__name__}: {e}")
            context.user_data.clear()
            await update.effective_message.reply_text("Вы в главном меню.", reply_markup=ReplyKeyboardRemove())
            return


    # Неавторизованный пользователь: кнопки «Поговорить с ИИ» и «Авторизация»
    if not storage.is_authorized(update.effective_user.id):
        txt = text_in
        if txt == "Поговорить с ИИ" or txt.lower() == "/ai":
            ai: Optional[AIClient] = context.bot_data.get("ai_client")
            if ai is None:
                await update.effective_message.reply_text(
                    "Раздел чата с ИИ временно недоступен: не настроен ключ OpenAI.",
                    reply_markup=UNAUTH_MENU,
                )
                return
            context.user_data["chat_with_ai"] = True
            await update.effective_message.reply_text(
                "Режим чата с Мирандой. Напишите свой вопрос. Для выхода нажмите \"Назад в меню\" или /exit.",
                reply_markup=BACK_MENU,
            )
            return
        if txt == "Авторизация":
            context.user_data["awaiting_inn"] = True
            await update.effective_message.reply_text(
                "Для авторизации отправьте ИНН вашей компании.", reply_markup=BACK_MENU
            )
            return
        if not context.user_data.get("awaiting_inn"):
            await update.effective_message.reply_text("Выберите действие.", reply_markup=UNAUTH_MENU)
            return

    # Ещё не авторизован — ждём ИНН
    if not context.user_data.get("awaiting_inn"):
        await update.effective_message.reply_text("Нажмите /start, чтобы начать авторизацию.")
        return

    # Пользователь прислал ИНН
    inn_raw = (update.effective_message.text or "").strip()
    inn = "".join(ch for ch in inn_raw if ch.isdigit())
    if not inn:
        await update.effective_message.reply_text("Похоже, это не ИНН. Пришлите, пожалуйста, цифры ИНН.")
        return

    email = find_email_by_inn(settings.excel_path, inn)
    if not email:
        await update.effective_message.reply_text(
            f"ИНН не найден. Пожалуйста, обратитесь к вашему менеджеру по номеру {settings.manager_phone}."
        )
        return

    pending = storage.create_pending(update.effective_user.id, inn, email)
    link = f"https://t.me/{settings.bot_username}?start=CONFIRM_{pending.token}"
    tg_link = f"tg://resolve?domain={settings.bot_username}&start=CONFIRM_{pending.token}"
    code_line = f"/start CONFIRM_{pending.token}"

    sender: EmailSender = context.bot_data["email_sender"]
    subject = "Подтверждение авторизации в боте Компании Миранда"
    body = (
        "Здравствуйте!\n\n"
        "Вы запросили подтверждение авторизации в Telegram-боте ИИ \"Компания Миранда\".\n"
        "Вариант 1 — открыть приложение Telegram по ссылке:\n"
        f"{tg_link}\n\n"
        "Вариант 2 — если ссылка открывается в браузере, используйте:\n"
        f"{link}\n\n"
        "Или отправьте боту это сообщение (скопируйте и вставьте):\n"
        f"{code_line}\n\n"
        "Если вы не запрашивали авторизацию, просто игнорируйте это письмо."
    )
    try:
        sender.send_text(email, subject, body)
        await update.effective_message.reply_text(
            "Мы отправили письмо со ссылкой-подтверждением на вашу почту. Проверьте, пожалуйста, email и перейдите по ссылке.",
            reply_markup=UNAUTH_MENU,
        )
    except Exception as e:
        logger.exception("Ошибка отправки письма: %s", e)
        await update.effective_message.reply_text(
            "Не удалось отправить письмо на ваш email. Пожалуйста, свяжитесь с менеджером.",
            reply_markup=UNAUTH_MENU,
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _mark_active(update, context)
    # Фото комнаты для визуализации
    if context.user_data.get("viz_mode") and context.user_data.get("viz_step") == "await_room_photo":
        ai: Optional[AIClient] = context.bot_data.get("ai_client")
        if ai is None:
            await update.effective_message.reply_text("ИИ недоступен для анализа фото комнаты.")
            return
        # Берём максимально крупное фото
        ph = update.effective_message.photo[-1]
        file = await context.bot.get_file(ph.file_id)
        await update.effective_message.reply_text("Анализирую фото комнаты...")
        room_desc = ai.describe_room_for_prompt(file.file_path)
        context.user_data["viz_room_desc"] = room_desc
        context.user_data["viz_step"] = "await_article"
        await update.effective_message.reply_text("Шаг 2. Введите артикул из коллекции NUR:", reply_markup=BACK_MENU)
        return
    return


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _mark_active(update, context)
    return


def main() -> None:
    settings = get_settings()

    if not settings.bot_token:
        raise SystemExit("Не задан BOT_TOKEN в .env")
    if not settings.bot_username:
        raise SystemExit("Не задан BOT_USERNAME в .env")

    storage = Storage(settings.storage_dir)
    email_sender = EmailSender(
        settings.smtp_host,
        settings.smtp_port,
        settings.smtp_user,
        settings.smtp_password,
        settings.smtp_from,
    )

    ai_client = None
    if settings.openai_api_key:
        ai_client = AIClient(settings.openai_api_key, settings.openai_model)

    app = Application.builder().token(settings.bot_token).build()
    app.bot_data["storage"] = storage
    app.bot_data["email_sender"] = email_sender
    app.bot_data["ai_client"] = ai_client
    # Коллекция тканей NUR для визуализации
    app.bot_data["fabric_collection"] = FabricCollection()
    app.bot_data["active_users"] = {}
    app.bot_data["active_last_log"] = 0.0

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("exit", cmd_exit))
    # Общий обработчик команд, кроме /start и /exit → в handle_text
    app.add_handler(MessageHandler(filters.COMMAND & ~filters.Regex(r"^/(start|exit)(?:\s|$)"), handle_text))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Бот запущен")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main() 