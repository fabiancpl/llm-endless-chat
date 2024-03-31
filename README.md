# Endless chat with a LLM using LangChain

This repository implements a functionality to allow extensive conversations with a LLM independently of the context window the particular model can offer. The motivation is derived from a buy assistant use case for an e-commerce platform.

The functionality is implemented in LangChain and consists on storing messages on memory segmented by topic (conversation). Whenever a user makes a request about a product recommendation, a first call to the LLM infers whether there is a change in the topic of the user's last message compared to the message history for the current conversation. If a topic change is detected, some complementary processes are done:

1) Summarize, embed and store on a vector database the message history for the current conversation.
2) Try to match the new topic with a previous conversation performing a similarity search between the user message embedding and summary embeddings.
3) If the similarity with the closest conversation summary is under a threshold, the respective conversation message history is retrieved and replaced by the current message history. Else, the current message history is just cleared and a new conversation is take over.

For the default case when no topic change is detected, the message history (conversation) remains the same.

After that, the final call to the LLM is made to generate the response to the user.