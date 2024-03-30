"""This module instantiate prompt templates and parsers."""

from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from src.constants import PROMPT_TEMPLATES_PATH


conversation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template_file(
            PROMPT_TEMPLATES_PATH + "conversation.txt", []
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

detect_conversation_change_parser = BooleanOutputParser()
detect_conversation_change_prompt = PromptTemplate.from_file(
    PROMPT_TEMPLATES_PATH + "detect_conversation_change.txt"
)

summarize_conversation_prompt = PromptTemplate.from_file(
    PROMPT_TEMPLATES_PATH + "summarize_conversation.txt"
)
