from dataclasses import dataclass, asdict
import json
import os
import subprocess # для запуска внешних процессов
from typing import Sequence
import uuid # для генерации уникальных идентификаторов

from dotenv import find_dotenv, load_dotenv
# интерфейс для языковых моделей
from langchain_core.language_models import LanguageModelLike
# конфигурация для запускаемых объектов
from langchain_core.runnables import RunnableConfig
# базовый класс и декоратор для инструментов
from langchain_core.tools import BaseTool, tool
from langchain_gigachat.chat_models import GigaChat
# создание агента
from langgraph.prebuilt import create_react_agent
# сохранение состояния в памяти
from langgraph.checkpoint.memory import InMemorySaver
from mail import fetch_recent_emails


load_dotenv(find_dotenv())
# PATH_FILE = "ООО_реквизиты_Маркетсистемс.docx"


@dataclass
class Bank:
    """Банковские реквизиты заказчика"""
    name: str  # наименование банка
    current_account: str  # расчётный счёт
    corporate_account: str  # корреспондентский счёт
    BIC: str  # БИК банка


@dataclass
class Customer:
    """Заказчик"""
    name: str  # полное название юридического лица, наприемер, ООО «Рога и копыта»
    INN: str  # ИНН
    KPP: str  # ОГРН или ОГРНИП
    address: str  # юридический адрес
    signatory: str  # подписант
    bank: Bank  # банковские реквизиты заказчика


@dataclass
class Job:
    task: str  # выполненная задача
    count: int  # количество
    unit: str  # единица измерения
    price: int  # цена за задачу
    price_total: int  # общая стоимость
    price_nds: int  # стоимость с НДС
    price_nds_18: int  # стоимость с НДС 18%
    price_nds_10: int  # стоимость с НДС 10%


@tool
def generate_pdf_act(customer: Customer, jobs: list[Job]) -> None:
    """
    Генерирует PDF-акт, в котором заполнены данные
    клиента, его банковские реквизиты, а также выполненные задачи

    Args:
        customer (Customer): данные клиента
        jobs (list[Job]): список выполненных задач для внесения в акт

    Returns:
        None
    """
    # Преобразуем объекты в словари
    act_json = {
        "customer": asdict(customer),
        "jobs": list(map(
            lambda j: asdict(j), jobs
        ))
    }
    # Сохраняем данные в JSON файл  
    with open(os.path.join("C:/Users/smoly/LangChain/LangChain-and-GigaChat/typst", "act.json"), "w", encoding="utf-8") as f:
        json.dump(act_json, f, ensure_ascii=False)
    command = ["typst", "compile", "--root", "./typst", "typst/act.typ"]
    try:
        subprocess.run(command,
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       text=True,
                       encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print(e.stderr)


@tool
def generate_pdf_invoice(customer: Customer, jobs: list[Job]) -> None:
    """
    Генерирует PDF-счёт, в котором заполнены данные
    клиента, а также выполненные задачи

    Args:
        customer (Customer): данные клиента
        jobs (list[Job]): список выполненных задач для внесения в акт

    Returns:
        None
    """
    invoice_json = {
        "customer": asdict(customer),
        "jobs": list(map(
            lambda j: asdict(j), jobs
        ))
    }
    with open(os.path.join("C:/Users/smoly/LangChain/LangChain-and-GigaChat/typst", "invoice.json"), "w", encoding="utf-8") as f:
        json.dump(invoice_json, f, ensure_ascii=False)
    command = ["typst", "compile", "--root", "./typst", "typst/invoice.typ"]
    try:
        subprocess.run(command,
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr)


# Класс LLM-агента
class LLMAgent:
    def __init__(self, model: LanguageModelLike, tools: Sequence[BaseTool]):
        self._model = model
        self._agent = create_react_agent(
            model,
            tools=tools,
            checkpointer=InMemorySaver())
        self._config: RunnableConfig = {
                "configurable": {"thread_id": uuid.uuid4().hex}} # уникальный ID сессии

    def upload_file(self, file):
        """Загрузка файла в GigaChat"""
        file_uploaded_id = self._model.upload_file(file).id_  # type: ignore
        return file_uploaded_id

    def invoke(
        self,
        content: str,
        attachments: list[str] | None = None,
        temperature: float = 0.1
    ) -> str:
        """Отправка сообщения агенту"""
        message: dict = {
            "role": "user",
            "content": content,
            **({"attachments": attachments} if attachments else {}) 
        }
        return self._agent.invoke(
            {
                "messages": [message],
                "temperature": temperature
            },
            config=self._config)["messages"][-1].content


def print_agent_response(llm_response: str) -> None:
    print(f"\033[35m{llm_response}\033[0m")


def get_user_prompt() -> str:
    return input("\nТы: ")


def main():
    system_prompt = (
        "Твоя задача найти файл,  содержащий реквизиты компании в одном из"
        "писем на почте. Найди имя этого файла и выведи его ответом. В ответе"
        "только имя файла или слово нет, если не нашёл"
    )

    model = GigaChat(
        model="GigaChat-2-Max",
        verify_ssl_certs=False,
    )

    @tool
    def upload_email_attacnment_file_to_llm(filename: str) -> None:
        """Загружает конкретный файл-аттачмент конкретного писыма в LLM"""
        print(f"upload {filename} to LLM")
        agent.upload_file(open(filename, "rb"))

    agent = LLMAgent(model, tools=[fetch_recent_emails])
    filename = agent.invoke(content=system_prompt)
    if filename == "нет":
        exit("ничего не получилось")


    agent = LLMAgent(model, tools=[generate_pdf_act,
                                   generate_pdf_invoice])
    file_uploaded_id = agent.upload_file(open(f"attachments/{filename}", "rb"))

    system_prompt = (
        "Твоя задача спросить у пользователя, что он хочет сгенерировать — акт или счёт или оба документа. "
        "Затем нужно сгенерировать акт или счёт, для этого тебе надо взять реквизиты "
        "контрагента из приложенного файла, а также запроси работы для включения в "
        "акт (наименования задач и их стоимость), работ может быть несколько. "
        "Если пользователь указывает в качестве работы аренда жилья, то для документов берём одну работу, в точности такую "
        "Аренда жилого помещения «Гостевой дом Kalina», стоимостью 5 тыс руб. за сутки и спроси на сколько суток снимают если не указали"
        "Никакие данные не придумывай, всё необходимое строго запроси у "
        "пользователя. Мои реквизиты заказчика не запрашивай, они есть в моём коде. "
        "Имя и отчество подписанта сокращаем до одной первой буквы, "
        "например, Иванов А.Е. "
        "Название компании оборачиваем в кавычки ёлочкой, например, "
        "ООО «Компания Сбер», то есть до названия компании ставим « и после названия "
        "ставим »."
    )

    agent_response = agent.invoke(content=system_prompt, attachments=[file_uploaded_id])

    while (True):
        # ... передача сообщения агенту ...
        agent_response = agent.invoke(get_user_prompt())
        print_agent_response(agent_response)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("/nПока, буду ждать ещё!")



































# ====================== версия одного агента без захода на почту =======================
# from dataclasses import dataclass, asdict
# import json
# import os
# import subprocess
# from typing import Sequence
# import uuid

# from dotenv import find_dotenv, load_dotenv
# from langchain_core.language_models import LanguageModelLike
# from langchain_core.runnables import RunnableConfig
# from langchain_core.tools import BaseTool, tool
# from langchain_gigachat.chat_models import GigaChat
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import InMemorySaver


# load_dotenv(find_dotenv())
# PATH_FILE = "C:/Users/smoly/LangChain/LangChain-and-GigaChat/ООО_реквизиты_Маркетсистемс.docx"

# @dataclass
# class Bank:
#     """Банковские реквизиты заказчика"""
#     name: str  # наименование банка
#     current_account: str  # расчётный счёт
#     corporate_account: str  # корреспондентский счёт
#     BIC: str  # БИК банка
# @dataclass
# class Customer:
#     """Заказчик"""
#     name: str  # полное название юридического лица, наприемер, ООО «Рога и копыта»
#     INN: str  # ИНН
#     KPP: str  # ОГРН или ОГРНИП
#     address: str  # юридический адрес
#     signatory: str  # подписант
#     bank: Bank  # банковские реквизиты заказчика

# @dataclass
# class Job:
#     task: str  # выполненная задача
#     count: int  # количество
#     unit: str  # единица измерения
#     price: int  # цена за задачу
#     price_total: int  # общая стоимость
#     price_nds: int  # стоимость с НДС
#     price_nds_18: int  # стоимость с НДС 18%
#     price_nds_10: int  # стоимость с НДС 10%

# @tool
# def generate_pdf_act(customer: Customer, jobs: list[Job]) -> None:
#     """
#     Генерирует PDF-акт, в котором заполнены данные
#     клиента, его банковские реквизиты, а также выполненные задачи

#     Args:
#         customer (Customer): данные клиента
#         jobs (list[Job]): список выполненных задач для внесения в акт

#     Returns:
#         None
#     """
#     # print(f"generate_pdf_act({asdict(customer)}, {list(map(lambda j: asdict(j), jobs))})")
#     act_json = {
#         "customer": asdict(customer),
#         "jobs": list(map(
#             lambda j: asdict(j), jobs
#         ))
#     }
#     with open(os.path.join("C:/Users/smoly/LangChain/LangChain-and-GigaChat/typst", "act.json"), "w", encoding="utf-8") as f:
#         json.dump(act_json, f, ensure_ascii=False)
#     command = ["typst", "compile", "--root", "./typst", "typst/act.typ"]
#     try:
#         subprocess.run(command,
#                check=True,
#                stdout=subprocess.PIPE,
#                stderr=subprocess.PIPE,
#                text=True,
#                encoding="utf-8")
#     except subprocess.CalledProcessError as e:
#         print(e.stderr)


# class LLMAgent:
#     def __init__(self, model: LanguageModelLike, tools: Sequence[BaseTool]):
#         self._model = model
#         self._agent = create_react_agent(
#             model,
#             tools=tools,
#             checkpointer=InMemorySaver())
#         self._config: RunnableConfig = {
#                 "configurable": {"thread_id": uuid.uuid4().hex}}

#     def upload_file(self, file):
#         file_uploaded_id = self._model.upload_file(file).id_  # type: ignore
#         return file_uploaded_id

#     def invoke(
#         self,
#         content: str,
#         attachments: list[str]|None=None,
#         temperature: float=0.1
#     ) -> str:
#         """Отправляет сообщение в чат"""
#         message: dict = {
#             "role": "user",
#             "content": content,
#             **({"attachments": attachments} if attachments else {}) 
#         }
#         return self._agent.invoke(
#             {
#                 "messages": [message],
#                 "temperature": temperature
#             },
#             config=self._config)["messages"][-1].content
    
# def main():

#     # более простой промпт для запуска (пробного)
#     system_prompt = (
#         "Твоя задача сгенерировать акт, для этого тебе надо будет взять реквизиты"
#         "контрагента из приложенного файла, и ещё запроси работы для включения в акт"
#         "(наименование задач и их стоимость) работ может быть несколько будь готов к "
#         "любому варианту генерировать — акт. Никакие данные придумывать не надо, всё "
#         "необходимое запроси у пользователя. Имя и Отчество подписанта сокращай до "
#         "одной буквы, например Иванов И.И. "
#         "Название компании оборачиваем в кавычки ёлочкой, например, ООО «Рога и копыта»,"
#         "то есть до названия компании ставим « и после названия ставим »."
#     )


#     model = GigaChat(
#     model="GigaChat-Pro",
#     verify_ssl_certs=False,
#     )
#     agent = LLMAgent(model, tools=[generate_pdf_act])
#     file_uploaded_id= agent.upload_file(open(PATH_FILE, "rb"))
#     agent_response = agent.invoke(content=system_prompt, attachments=[file_uploaded_id])

#     print("Agent response: ", agent_response)
#     while(True):
#         prompt = input("Prompt: ")
#         llm_response = agent.invoke(content=prompt)
#         print("Agent response: ", llm_response)

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("/nПока, буду ждать ещё!")
