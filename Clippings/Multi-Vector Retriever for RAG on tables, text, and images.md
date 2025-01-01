---
title: "Multi-Vector Retriever for RAG on tables, text, and images"
source: "https://blog.langchain.dev/semi-structured-multi-modal-rag/"
author:
  - "[[LangChain]]"
published: 2023-10-20
created: 2025-01-01
description: "SummarySeamless question-answering across diverse data types (images, text, tables) is one of the holy grails of RAG. We’re releasing three new cookbooks that showcase the multi-vector retriever for RAG on documents that contain a mixture of content types. These cookbooks as also present a few ideas for pairing"
tags:
  - "clippings"
---
### Summary

Seamless question-answering across diverse data types (images, text, tables) is one of the holy grails of [[RAG]]. We’re releasing three new cookbooks that showcase the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector?ref=blog.langchain.dev) for RAG on documents that contain a mixture of content types. These cookbooks as also present a few ideas for pairing [multimodal LLMs](https://huyenchip.com/2023/10/10/multimodal.html?ref=blog.langchain.dev) with the multi-vector retriever to unlock RAG on images.

- [Cookbook for semi-structured (tables + text) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev)
- [Cookbook for multi-modal (text + tables + images) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb?ref=blog.langchain.dev)
- [Cookbook for private multi-modal (text + tables + images) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb?ref=blog.langchain.dev)

### Context

LLMs can acquire new information in at least two ways: (1) weight updates (e.g., [fine-tuning](https://blog.langchain.dev/using-langsmith-to-support-fine-tuning-of-open-source-llms/)) and (2) [RAG](https://docs.google.com/presentation/d/1exjoapZ4EB_2xSQ7BSfSdPsZT30ap_6N3bi37F3cG0E/edit?ref=blog.langchain.dev#slide=id.g23a039de743_0_40) (retrieval augmented generation), which passes relevant context to the LLM via prompt. RAG has particular promise [for](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb?ref=blog.langchain.dev) [factual](https://www.youtube.com/watch?v=hhiLw5Q_UFg&ref=blog.langchain.dev) [recall](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts?ref=blog.langchain.dev) because it marries the reasoning capability of LLMs with the content of external data sources, which is [particularly powerful](https://www.glean.com/blog/lessons-and-learnings-from-building-an-enterprise-ready-ai-assistant?ref=blog.langchain.dev) for enterprise data.

![](https://blog.langchain.dev/content/images/2023/10/image-15.png)

### Ways To Improve RAG

A number of techniques to improve RAG have been developed:

| **Idea** | **Example** | **Sources** |
| --- | --- | --- |
| **Base case** **RAG** | Top K retrieval on embedded document chunks, return doc chunks for LLM context window | [LangChain vectorstores](https://python.langchain.com/docs/modules/data_connection/vectorstores/?ref=blog.langchain.dev), [embedding models](https://python.langchain.com/docs/modules/data_connection/text_embedding/?ref=blog.langchain.dev) |
| **Summary** **embedding** | Top K retrieval on embedded document summaries, but return full doc for LLM context window | [LangChain Multi Vector Retriever](https://twitter.com/hwchase17/status/1695078249788027071?s=20&ref=blog.langchain.dev) |
| **Windowing** | Top K retrieval on embedded chunks or sentences, but return expanded window or full doc | [LangChain Parent Document Retriever](https://twitter.com/hwchase17/status/1691179199594364928?s=20&ref=blog.langchain.dev) |
| **Metadata filtering** | Top K retrieval with chunks filtered by metadata | [Self-query retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query?ref=blog.langchain.dev) |
| **Fine-tune RAG embeddings** | Fine-tune embedding model on your data | [LangChain fine-tuning guide](https://blog.langchain.dev/using-langsmith-to-support-fine-tuning-of-open-source-llms/) |
| **2-stage** **RAG** | First stage keyword search followed by second stage semantic Top K retrieval | [Cohere re-rank](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker?ref=blog.langchain.dev) |

### Applying RAG to Diverse Data Types

Yet, RAG on documents that contain semi-structured data (structured tables with unstructured text) and multiple modalities (images) has remained a challenge. With the emergence of [several](https://www.adept.ai/blog/fuyu-8b?ref=blog.langchain.dev) [multimodal](https://llava.hliu.cc/?ref=blog.langchain.dev) [models](https://openai.com/research/gpt-4v-system-card?ref=blog.langchain.dev), it is now worth considering unified strategies to enable RAG across modalities and semi-structured data.

### Multi-Vector Retriever

Back in August, we released the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector?ref=blog.langchain.dev). It uses a simple, powerful idea for RAG: decouple documents, which we want to use for answer synthesis, from a reference, which we want to use for retriever. As a simple example, we can create a summary of a verbose document optimized to vector-based similarity search, but still pass the full document into the LLM to ensure no context is lost during answer synthesis. Here, we show that this approach useful beyond raw text, and can be applied generally to either tables or images to support RAG.

![](https://blog.langchain.dev/content/images/2023/10/mvr_overview.png)

**Document Loading**

Of course, to enable this approach we first need the ability to partition a document into its various types. [Unstructured](https://unstructured.io/?ref=blog.langchain.dev) is a great ELT tool well-suited for this because it can extract elements (tables, images, text) from numerous file types.

For example, Unstructured will [partition PDF files](https://unstructured-io.github.io/unstructured/bricks/partition.html?ref=blog.langchain.dev#partition-pdf) by first removing all embedded image blocks. Then it will use a layout model ([YOLOX](https://arxiv.org/abs/2107.08430?ref=blog.langchain.dev)) to get bounding boxes (for tables) as well as `titles`, which are candidate sub-sections of the document (e.g., Introduction, etc). It will then perform post processing to aggregate text that falls under each `title` and perform further chunking into text blocks for downstream processing based on user-specific flags (e.g., min chunk size, etc).

**Semi-Structured Data**

The combination of Unstructured file parsing and multi-vector retriever can support RAG on semi-structured data, which is a challenge for naive chunking strategies that may spit tables. We generate [summaries](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector?ref=blog.langchain.dev#summary) of table elements, which is better suited to natural language retrieval. If a table summary is retrieved via semantic similarity to a user question, the *raw table* is passed to the LLM for answer synthesis as described above. See the below cookbook and diagram:

- [Cookbook for semi-structure RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev)

![](https://blog.langchain.dev/content/images/2023/10/image-16.png)

**Multi-Modal Data**

We can take this one step further and consider images, which is quickly becoming enabled by the release of multi-modal LLMs such as [GPT4-V](https://openai.com/research/gpt-4v-system-card?ref=blog.langchain.dev) and open source models such as [LLaVA](https://llava.hliu.cc/?ref=blog.langchain.dev) and [Fuyu-8b](https://www.adept.ai/blog/fuyu-8b?ref=blog.langchain.dev). There are at least three ways to approach the problem, which utilize the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector?ref=blog.langchain.dev#summary) framework as discussed above:

`Option 1:` Use multimodal embeddings (such as [CLIP](https://openai.com/research/clip?ref=blog.langchain.dev)) to embed images and text together. Retrieve either using similarity search, but simply link to images in a docstore. Pass raw images and text chunks to a multimodal LLM for synthesis.

`Option 2:` Use a multimodal LLM (such as [GPT4-V](https://openai.com/research/gpt-4v-system-card?ref=blog.langchain.dev), [LLaVA](https://llava.hliu.cc/?ref=blog.langchain.dev), or [FUYU-8b](https://www.adept.ai/blog/fuyu-8b?ref=blog.langchain.dev)) to produce text summaries from images. Embed and retrieve text summaries using a text embedding model. And, again, reference raw text chunks or tables from a docstore for answer synthesis by a LLM; in this case, we exclude images from the docstore (e.g., because can't feasibility use a multi-modal LLM for synthesis).

`Option 3:` Use a multimodal LLM (such as [GPT4-V](https://openai.com/research/gpt-4v-system-card?ref=blog.langchain.dev), [LLaVA](https://llava.hliu.cc/?ref=blog.langchain.dev), or [FUYU-8b](https://www.adept.ai/blog/fuyu-8b?ref=blog.langchain.dev)) to produce text summaries from images. Embed and retrieve image summaries with a reference to the raw image, as we did above in option 1. And, again, pass raw images and text chunks to a multimodal LLM for answer synthesis. This option is sensible if we don't want to use multimodal embeddings.

![](https://blog.langchain.dev/content/images/2023/10/image-22.png)

We tested `option 2` using the [7b parameter LLaVA model](https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main?ref=blog.langchain.dev) (weights available [here](https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main?ref=blog.langchain.dev)) to generate image image summaries. LLaVA recently added to [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3436?ref=blog.langchain.dev), which allows it run on consumer laptops (Mac M2 max, 32gb ~45 token / sec) and produces reasonable image summaries. For example, for the image below it captures the humor: `The image features a close-up of a tray filled with various pieces of fried chicken. The chicken pieces are arranged in a way that resembles a map of the world, with some pieces placed in the shape of continents and others as countries. The arrangement of the chicken pieces creates a visually appealing and playful representation of the world.`

![figure-8-1.jpg](https://blog.langchain.dev/content/images/2023/10/data-src-image-f9f6a819-aed0-425f-b752-b1cec011ffc1.jpeg)

We store these in the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector?ref=blog.langchain.dev#summary) along with table and text summaries.

- [Cookbook for multimodal (text + tables + images) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb?ref=blog.langchain.dev)

If data privacy is a concern, this RAG pipeline can be run locally using open source components on a consumer laptop with [LLaVA](https://github.com/haotian-liu/LLaVA/?ref=blog.langchain.dev) 7b for image summarization, [Chroma](https://www.trychroma.com/?ref=blog.langchain.dev) vectorstore, open source embeddings ([Nomic’s GPT4All](https://python.langchain.com/docs/integrations/text_embedding/gpt4all?ref=blog.langchain.dev)), the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector?ref=blog.langchain.dev#summary), and [LLaMA2-13b-chat](https://python.langchain.com/docs/integrations/chat/ollama?ref=blog.langchain.dev) via [Ollama.ai](http://ollama.ai/?ref=blog.langchain.dev) for answer generation.

- [Cookbook for private multi-modal (text + tables + images) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb?ref=blog.langchain.dev)

### Conclusion

We show that the multi-vector retriever can be used to support semi-structured RAG as well as semi-structured RAG with multi-modal data. We also show that this full pipeline can be run locally on a consumer laptops using open source components. Finally, we present three general approaches for multimodal RAG that utilize the multi-vector retriever concept to be features in future cookbooks.

### Join our newsletter

Updates from the LangChain team and community

Enter your email

Processing your application...

Success! Please check your inbox and click the link to confirm your subscription.

Sorry, something went wrong. Please try again.