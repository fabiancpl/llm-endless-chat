# Endless chat with a LLM using LangChain

This repository introduces a feature that enables extended conversations with a LLM, regardless of the context window limitations inherent to the specific model. This development is motivated by the requirements of a buy assistant use case for an e-commerce platform.

The feature, implemented in LangChain, involves storing messages on memory, segmented by topic (conversation). Whenever a user requests a product recommendation, an initial call to the LLM determines whether there is a change in topic from the user's last message relative to the message history of the current conversation. If a topic change is detected, several complementary processes are initiated:

1) Summarize the message history for the current conversation, then embed and store it in a vector database. This summarization process requires a second call to the LLM.
2) Attempt to match the user’s new message to a previous conversation by performing a similarity search between the embedding of the message and the embeddings of all summaries stored in the database.
3) If the similarity to the closest summary falls below a threshold, the message history from that conversation is retrieved to update the current message history. If not, the message history is cleared, and a new conversation is initiated.

In the default case where no topic change is detected, the message history remains unchanged.

Subsequently, the final call is made to the LLM to generate the response to the user.

You can read a more comprehensive explanation [here](https://medium.com/@fabiancpl91/endless-chat-with-a-llm-using-langchain-98342ee7ec07).