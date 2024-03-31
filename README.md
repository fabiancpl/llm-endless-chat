# Endless chat with a LLM using LangChain

This repository implements a functionality to allow extensive conversations with a LLM independently of the context window the particular model can offer. The motivation is derived from a buy assistant use case for an e-commerce platform.

The functionality is implemented in LangChain and consists on storing messages on memory segmented by topic (conversation). Whenever a user makes a request about a product recommendation, a first call to the LLM infers if there is a change in the topic of the user's last message compared to the message history for the current conversation. If a topic change is detected, some complementary processes are carried out:

1) Summarize, embed and store on a vector database the message history for the current conversation. The summarization implies a second call to the LLM.
2) Try to match the user's new message with a previous conversation performing a similarity search between the message embedding and all the summary embeddings stored on the database.
3) If the similarity with the closest summary is under a threshold, the message history of the respective conversation is retrieved and replaced by the current message history. Otherwise, the current message history is simply deleted and a new conversation is started.

For the default case when no topic change is detected, the message history remains the same.

After that, the final call to the LLM is made to generate the response to the user.

You can read a more comprehensive explanation [here](https://medium.com/@fabiancpl91/endless-chat-with-a-llm-using-langchain-98342ee7ec07).