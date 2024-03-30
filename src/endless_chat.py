""""This module includes the core functions of the endless chat."""

import json
import uuid

from operator import itemgetter

from typing import Optional

from langchain.memory import RedisChatMessageHistory
from langchain.memory.buffer import ConversationBufferMemory
from langchain.schema.runnable import (
    RunnableLambda,
    RunnablePassthrough,
)

from langchain_community.utilities.redis import get_client
from langchain_community.vectorstores import FAISS

from src.constants import (
    CONVERSATION_SIMILARITY_THRESHOLD,
    REDIS_PREFIX_MESSAGES,
    REDIS_PREFIX_SUMMARIES,
    REDIS_URL,
)
from src.prompt_templates import (
    conversation_prompt,
    detect_conversation_change_parser,
    detect_conversation_change_prompt,
    summarize_conversation_prompt,
)


class EndlessChat:
    """This class implements the core functions of the endless chat."""

    def __init__(self, user_id: str, llm, embeddings_model):
        self.user_id = user_id
        self.conversation_id: str = ""
        self.conversation_changed: bool = False

        # Interfaces*
        self.llm = llm
        self.embeddings_model = embeddings_model

        # Redis client
        self.redis_client = get_client(redis_url=REDIS_URL)

        # Setup a new conversation
        self._setup_conversation()

    def _setup_conversation(self, conversation_id: Optional[str] = None) -> None:
        if conversation_id is None:
            self.conversation_id = str(uuid.uuid4())
            print("INFO: New conversation created")
        else:
            self.conversation_id = conversation_id
            print("INFO: Previous conversation attached")
        print("DEBUG: Conversation ID:", self.conversation_id)

        self.memory_backend = RedisChatMessageHistory(
            session_id=f"{self.user_id}:{self.conversation_id}",
            url=REDIS_URL,
            key_prefix=f"{REDIS_PREFIX_MESSAGES}:",
        )

        self.conversation_memory = ConversationBufferMemory(
            chat_memory=self.memory_backend, return_messages=True
        )

        self.conversation_chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self.conversation_memory.load_memory_variables)
                | itemgetter("history")
            )
            | conversation_prompt
            | self.llm
        )

        self.conversation_change_chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self.conversation_memory.load_memory_variables)
                | itemgetter("history")
            )
            | detect_conversation_change_prompt
            | self.llm
            | detect_conversation_change_parser
        )

        self.summarize_conversation_chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self.conversation_memory.load_memory_variables)
                | itemgetter("history")
            )
            | summarize_conversation_prompt
            | self.llm
        )

    def _search_similar_conversations(self, user_message: str) -> Optional[str]:
        # Get all conversation summaries for the user
        conversations = []
        conversation_keys = self.redis_client.keys(
            pattern=f"{REDIS_PREFIX_SUMMARIES}:{self.user_id}:*"
        )
        conversation_keys = [k.decode() for k in conversation_keys]
        for key in conversation_keys:
            conversation = json.loads(self.redis_client.get(key).decode("utf-8"))
            conversations.append((conversation["text"], conversation["embedding"]))

        # Create in-memory db for comparing summary embeddings
        summaries_faiss = FAISS.from_embeddings(
            conversations,
            self.embeddings_model,
            metadatas=[{"conversation_id": k.split(":")[2]} for k in conversation_keys],
        )

        # Get the nearest conversations for the user message
        nearest_conversations = summaries_faiss.similarity_search_with_score(user_message)

        # Validate similarity threshold
        nearest_conversation: Optional[str] = None
        if isinstance(nearest_conversations, list) and (len(nearest_conversations) > 0):
            print(
                "DEBUG: Similarity score to the nearest conversation:",
                nearest_conversations[0][1],
            )
            if nearest_conversations[0][1] < CONVERSATION_SIMILARITY_THRESHOLD:
                nearest_conversation = nearest_conversations[0][0].metadata["conversation_id"]
        else:
            print("DEBUG: No conversations to follow up")

        return nearest_conversation

    def _summarize_current_conversation(self) -> None:
        if len(self.conversation_memory.load_memory_variables({})["history"]) > 0:
            # Summarize current conversation
            conversation_summary = self.summarize_conversation_chain.invoke({}).content

            # Ebed summary
            summary_embeded = self.embeddings_model.embed_query(conversation_summary)

            # Persist summary text+embedding
            self.redis_client.set(
                f"{REDIS_PREFIX_SUMMARIES}:{self.user_id}:{self.conversation_id}",
                json.dumps(
                    {
                        "text": conversation_summary,
                        "embedding": summary_embeded,
                    }
                ),
            )

    def detect_conversation_change(self, user_message: str):
        """Detect a conversation change based on chat history and last user
        message.

        Args:
            user_message (str): User message

        Returns:
            bool: Flag for conversation change

        """

        if len(self.conversation_memory.load_memory_variables({})["history"]) > 0:
            # Validate if conversation changed by calling the LLM
            self.conversation_changed = self.conversation_change_chain.invoke(
                {"input": user_message}
            )

            if self.conversation_changed:
                print("INFO: Conversation change detected")

                # Summarize current conversation
                self._summarize_current_conversation()

                # Search for most similar conversations
                nearest_conversation: Optional[str] = self._search_similar_conversations(
                    user_message
                )

                # Setup a previous conversation
                self._setup_conversation(nearest_conversation)

    def chat(self, user_message: str) -> str:
        """Call the assistant and save user and assistant messages.

        Args:
            user_message (str): User message

        Returns:
            str: Assistant message

        """

        # Call the chain and get the assistant message
        assistant_message = self.conversation_chain.invoke({"input": user_message}).content

        # Persist messages on memory
        self.conversation_memory.save_context(
            {"input": user_message}, {"output": assistant_message}
        )

        return assistant_message
