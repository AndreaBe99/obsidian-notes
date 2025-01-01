---
title: "Multimodality RAG (MRAG): Extract, Store and Retrieve visual data, diagrams, images from document"
source: "https://medium.com/@shivamarora1/multimodality-rag-mrag-extract-store-and-retrieve-visual-data-diagrams-images-from-document-dd47b1892dc8"
author:
  - "[[Shivam]]"
published: 2024-08-23
created: 2025-01-01
description: "If a RAG application (Retrieval Augmented Generation) is able to understand and extract information from wide variety of sources then its utility would be incrementally increased. RAG can extract…"
tags:
  - "clippings"
---
![](https://miro.medium.com/v2/resize:fit:875/1*kLu5QFWv-pEYKiM0JVzI0Q.png)

Multimodality RAG (MRAG): Extract, Store and Retrieve visual data from document

**Summary**

If a [[RAG]] application (Retrieval Augmented Generation) is able to understand and extract information from wide variety of sources then its utility would be incrementally increased. RAG can extract information from different sources like image, graph, charts apart from traditional text document. This requires framework to understand and generate final response by coherently interpreting textual and visual information.

**MRAG (**Multimodality Retrieval Augment Generation**)**

Enterprise data is usually stored in different folders. These folders contain files that are of different modality. Text and Image modality are commonly found. RAG application should be able to extract and understand information stored in different modality documents. Multimodality files are unstructured and usually have format different from each other. Multimodality files makes RAG pipeline complex because different approach and method would be used to extract information.

*TLDR. Here is repo with all required code*

**Challenges with documents that contains image**

Normally we use embedding models to convert textual data into vectors. These textual embeddings models only convert textual data and not suitable for images. There are various other embedding models available to convert images into vector. We need to utilise both embedding models if we want to convert textual document that contains images or diagrams.

![](https://miro.medium.com/v2/resize:fit:875/1*PsZx4jxX_9WKqiYWviLQwQ.jpeg)

Figure 1: Document with image and text modality

Similar to above, document and report may contain information-dense images like charts and diagrams. These elements have additional context and many points of interest. RAG pipeline must effectively capture these visual elements along with textual data. You must make sure semantic representation of charts and diagrams aligns with semantic meaning of text.

**Approach for Multimodal RAG**

With key challenges understood, Let’s discuss architecture to tackle these challenges:

One approach to solve this problem is to ground all modalities (*text, image, charts*) to one primary modality and encode that modality into vector space. In this case,

- **Processing Phase**: process text normally but for images, first create text description, summary or metadata in processing step. Actual image is also stored for later retrieval.
- **Retrieval Phase:** retrieval part primarily extracts text chunks and image metadata. Final answer is generated with LLMs along with image (chart) that ground the generated response.

**Visual Language Model (VLM)**

Large Language Models (LLMs) are designed to understand, interpret, and generate text-based information. LLMs are trained on vast amounts of textual data, LLMs can perform a range of natural language processing tasks, such as text generation, summarisation, question-answering, and more.

Visual Language Models (VLMs) takes both images and text as inputs and can answer questions about images with detail and context, VLMs can perform deeper analysis of images and provide useful insights, such as captioning for images, object detection, and reading text embedded within images.

In this example following models and tools are used to build RAG pipeline:

- **LLM**: OpenAI, Claude, Llama 3 for general reasoning and question answering.
- **VLM:** Microsoft Kosmos 2, Paligemma for visual question answering and image description.
- **DePlot:** Subset of VLM to comprehend charts and plots
- **Embedding model**: Encoding data into vectors
- **Vector database**: Storing and retrieval of vectors

**Extracting multimodal data and creating a vector database**

The first step for building a RAG pipeline is to preprocess your data and store it as vectors in a vector store so that you can then retrieve relevant vectors based on a user’s query. With images present in the data, here is a generic RAG preprocessing workflow to work through (*Figure 2*).

![](https://miro.medium.com/v2/resize:fit:759/1*kRpN1Yus4_s-6u_xnei8nA.jpeg)

Figure 2: Generic MRAG pipeline with image and text modality

The document may contains several images, bar charts like (*Figure 3)* or textual data. To interpret these bar charts, use [Google’s DePlot](https://huggingface.co/google/deplot), a visual-language model capable of comprehending charts and plots when coupled with an LLM.

![](https://miro.medium.com/v2/resize:fit:875/1*4ufcOFGzuti1xRkeaBnKww.png)

Figure 3: DePlot: Plot-to-text translation

Document can have other images . [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) VLM is capable of performing various tasks like: *comprehending*, *extracting objects*, *answer* *questions* and *extracting* deeper information about images. Paligemma generally works for most types of images but if you want to use it for specific precision use case like medical, construction images etc. consider to fine tune it.

![](https://miro.medium.com/v2/resize:fit:610/1*k25vvlj1Mbgx4tsIkNoVVA.png)

Figure 4: Visual Question Answer using PaliGemma VLM

With knowledge of working of **Paligemma** and **DePlot**, Let’s expand the preprocessing pipeline that we discuss in *Figure 2* to dive deeper into handling each modality in the pipeline that leverages textual data extractors, VLM and LLM to create vectors.

![](https://miro.medium.com/v2/resize:fit:875/1*_aCmfB84um3nefiELp6GiA.jpeg)

Figure 5: DePlot convert chart to textual table, Paligemma convert image to text description

**MRAG workflow**

The goal is to ground images to text modality. Start by extracting and classifying your data into images and text. Then handle these two modalities separately to eventually store them into vector store.

Paligemma VLM can generate image description and can be used to classify images into categories whether they are graphs or not. Based on classification, use DePlot to convert graphs and charts to linearised tabular text. Since this tabular text is different from regular text, this text is summarised using LLM so that it can be retrieved from vector store easily in subsequent retrieval phase. Similarly normal images context is captured in description and that is also stored in vector storage.

Regular text is extracted from document using traditional OCR, Extracted text is [chunked](https://shivamreloaded.com/ai/2023/10/27/document-splitting.html) and vectors are stored in vector storage.

**Talking to your application**

Following above steps, you have captured all the multimodal information present in the PDF. Here’s how RAG pipeline will work when user asks a question.

When a user asks the system with a question, MRAG pipeline converts question into an embedding and performs semantic search to retrieve relevant chunks from store. Since these chunks are also retrieved from images or chart descriptions, you need to take additional steps before sending these chunks to LLM for final response.

![](https://miro.medium.com/v2/resize:fit:875/1*FyUsFWXi-tBro5gJhA-w4w.jpeg)

Figure 6: Relevant is chunks are fetched from vector storage using cosine similarity

Here are further steps that MRAG pipeline takes after retrieving relevant chunks from the vector store:

1. If chunk is retrieved from the description generated by Paligemma then stored image is simply send along with user’s question to Paligemma (VLM) to generate the answer. This is nothing but VQA task. VLM is capable to understand image semantics and objects inside the image. Generated answer is sent as context to LLM
2. If chunk is retrieved from chart or plot, then linearized table data stored as summarized text is appended as context.

3\. The chunk coming from regular text are used as it is.

All these chunks, along with the user question, are now ready to generate a final answer. From document in *Figure 1* LLM referred images and text generated the final response.

![](https://miro.medium.com/v2/resize:fit:875/1*PZwx36VuqFA4y319wXZ22w.png)

Response from LLM with appropriate image

You can find live demo [here](https://chat-diagram-extractor.streamlit.app/) and code in link below: