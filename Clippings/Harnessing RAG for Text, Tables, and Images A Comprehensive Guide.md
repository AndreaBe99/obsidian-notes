---
title: "Harnessing RAG for Text, Tables, and Images: A Comprehensive Guide"
source: https://medium.com/@kbouziane.ai/harnessing-rag-for-text-tables-and-images-a-comprehensive-guide-ca4d2d420219
author:
  - "[[khalid bouziane]]"
published: 2023-11-28
created: 2025-01-01
description: In the realm of information retrieval, Retrieval Augmented Generation (RAG) has emerged as a powerful tool for extracting knowledge from vast amounts of text data. This versatile technique leverages…
tags:
  - clippings
---


![](https://miro.medium.com/v2/resize:fit:875/0*_9o9DiGrZ66TXvHf)

In the realm of information retrieval, Retrieval Augmented Generation ([[RAG]]) has emerged as a powerful tool for extracting knowledge from vast amounts of text data. This versatile technique leverages a combination of retrieval and generation strategies to effectively summarize and synthesize information from relevant documents. However, while RAG has gained considerable traction, its application to a broader range of content types, encompassing text, tables, and images, remains relatively unexplored.

**The Challenge of Multimodal Content**

Most real-world documents contain a rich tapestry of information, often combining text, tables, and images to convey complex ideas and insights. While traditional RAG models excel at processing text, they struggle to effectively integrate and comprehend multimodal content. This limitation hinders the ability of RAG to fully capture the essence of these documents, potentially leading to incomplete or inaccurate representations.

In this post we will explore how to create a Multi-modal RAG that can handle these types of documents.  
Here is a graph that we will use as a guide to process such documents.

![](https://miro.medium.com/v2/resize:fit:875/1*SDpKs-GxgosuQsjQrwBjhg.png)

Multi-Modal RAG

**Step 1 : Split the file to raw elements.**

First, let’s import all necessary libraries to our environment

```
import os  
import openai  
import io  
import uuid  
import base64  
import time  
from base64 import b64decode  
import numpy as np  
from PIL import Image  
  
from unstructured.partition.pdf import partition_pdf  
  
from langchain.chat_models import ChatOpenAI  
from langchain.schema.messages import HumanMessage, SystemMessage  
from langchain.vectorstores import Chroma  
from langchain.storage import InMemoryStore  
from langchain.schema.document import Document  
from langchain.embeddings import OpenAIEmbeddings  
from langchain.retrievers.multi_vector import MultiVectorRetriever  
from langchain.chat_models import ChatOpenAI  
from langchain.prompts import ChatPromptTemplate  
from langchain.schema.output_parser import StrOutputParser  
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda  
  
from operator import itemgetter
```

We will use [Unstructured](https://unstructured.io/) to parse images, texts, and tables from documents (PDFs), you can run this code directly in this [google colab](https://colab.research.google.com/drive/1I9-JMGL76XXzUel7XTjO8pWK7jV-V8WD?usp=sharing), or download the pdf file from [here](https://sgp.fas.org/crs/misc/IF10244.pdf) ,upload it to session storage. and follow steps below.

> ( Refer to the installation instructions in google colab to setup your venv before excuting the code )

```
# load the pdf file to drive  
# split the file to text, table and images  
def doc_partition(path,file_name):  
	raw_pdf_elements = partition_pdf(  
		filename=path + file_name,  
		extract_images_in_pdf=True,  
		infer_table_structure=True,  
		chunking_strategy="by_title",  
		max_characters=4000,  
		new_after_n_chars=3800,  
		combine_text_under_n_chars=2000,  
		image_output_dir_path=path)  
  
	return raw_pdf_elements  
	
path = "/content/"  
file_name = "wildfire_stats.pdf"  
raw_pdf_elements = doc_partition(path,file_name)
```

Once you run the code above, all images included in the file will be extracted and saved in your path. in our case (path = “/content/”)

Next we will append each raw element to its category, (text to texts ,table to tables, for images, *unstructed* has taken care of that already..).

```
def data_category(raw_pdf_elements): # we may use decorator here  
	tables = []  
	texts = []  
	for element in raw_pdf_elements:  
		if "unstructured.documents.elements.Table" in str(type(element)):  
			tables.append(str(element))  
		elif "unstructured.documents.elements.CompositeElement" in str(type(element)):  
		texts.append(str(element))  
	data_category = [texts,tables]  
	return data_category  
	
texts = data_category(raw_pdf_elements)[0]  
tables = data_category(raw_pdf_elements)[1]
```

**Step 2 : Image captioning and table summarizing**

For summarizing tables, we will use [[Langchain]] and GPT-4. For generating image captions, we will use GPT-4-Vision-Preview. This is because it is the only model that can currently handle multiple images together, which is important for our documents that contain multiple images. For text elements, we will leave them as they are before making them into embeddings.

Get your OpenAI API key ready

```
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxx'openai.api_key = os.environ["OPENAI_API_KEY"]
```
```
def tables_summarize(data_category):    prompt_text = """You are an assistant tasked with summarizing tables. \                    Give a concise summary of the table. Table chunk: {element} """    prompt = ChatPromptTemplate.from_template(prompt_text)    model = ChatOpenAI(temperature=0, model="gpt-4")    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})        return table_summariestable_summaries = tables_summarize(data_category)text_summaries = texts
```

For images we need to encode them to base64 format before feeding them to our model for captioning

```
def encode_image(image_path):    ''' Getting the base64 string '''    with open(image_path, "rb") as image_file:        return base64.b64encode(image_file.read()).decode('utf-8')def image_captioning(img_base64,prompt):    ''' Image summary '''    chat = ChatOpenAI(model="gpt-4-vision-preview",                      max_tokens=1024)    msg = chat.invoke(        [            HumanMessage(                content=[                    {"type": "text", "text":prompt},                    {                        "type": "image_url",                        "image_url": {                            "url": f"data:image/jpeg;base64,{img_base64}"                        },                    },                ]            )        ]    )    return msg.content
```

Now we can append our list of images\_base64 and summarize them, then we split base64-encoded images and their associated texts.

while running the code below you may get this `RateLimitError` with error code `429`. This error occurs when you’ve exceeded the rate limit for requests per minute (RPM) for the `gpt-4-vision-preview` in your organization, in my case, I have a usage limit of 3 RPM, so after each image captioning, I set 60s as a safety mesure and wait for the rate limit to reset.

```
img_base64_list = []image_summaries = []prompt = "Describe the image in detail. Be specific about graphs, such as bar plots."for img_file in sorted(os.listdir(path)):    if img_file.endswith('.jpg'):        img_path = os.path.join(path, img_file)        base64_image = encode_image(img_path)        img_base64_list.append(base64_image)        img_capt = image_captioning(base64_image,prompt)        time.sleep(60)        image_summaries.append(image_captioning(img_capt,prompt))
```
```
def split_image_text_types(docs):    ''' Split base64-encoded images and texts '''    b64 = []    text = []    for doc in docs:        try:            b64decode(doc)            b64.append(doc)        except Exception as e:            text.append(doc)    return {        "images": b64,        "texts": text    }
```

**Step 3** : **Create a Multi-vector retriever and store texts, tables, images and their indexes in a Vectore Base**

We have completed the first part, wich include document partitioning to raw elements, summarizing tables and images, now we are ready for the second part, where we will create a multi-vector retriever and store our outputs from part one in chromadb along with their ids.

> 1\. we create a vectorestore to index the child chunks (summary\_texts ,summary\_tables, summary\_img), and use OpenAIEmbeddings() for embeddings,
> 
> 2 .A docstore for the parent documents to store (doc\_ids, texts), (table\_ids, tables) and (img\_ids, img\_base64\_list)

```
vectorstore = Chroma(collection_name="multi_modal_rag",                     embedding_function=OpenAIEmbeddings())store = InMemoryStore()id_key = "doc_id"retriever = MultiVectorRetriever(    vectorstore=vectorstore,    docstore=store,    id_key=id_key,)doc_ids = [str(uuid.uuid4()) for _ in texts]summary_texts = [    Document(page_content=s, metadata={id_key: doc_ids[i]})    for i, s in enumerate(text_summaries)]retriever.vectorstore.add_documents(summary_texts)retriever.docstore.mset(list(zip(doc_ids, texts)))table_ids = [str(uuid.uuid4()) for _ in tables]summary_tables = [    Document(page_content=s, metadata={id_key: table_ids[i]})    for i, s in enumerate(table_summaries)]retriever.vectorstore.add_documents(summary_tables)retriever.docstore.mset(list(zip(table_ids, tables)))img_ids = [str(uuid.uuid4()) for _ in img_base64_list]summary_img = [    Document(page_content=s, metadata={id_key: img_ids[i]})    for i, s in enumerate(image_summaries)]retriever.vectorstore.add_documents(summary_img)retriever.docstore.mset(list(zip(img_ids, img_base64_list)))
```

**Step 4 : Wrap all the above using langchain RunnableLambda**

- We first compute the context (both “texts” and “images” in this case) and the question (just a RunnablePassthrough here)
- Then we pass this into our prompt template, which is a custom function that formats the message for the gpt-4-vision-preview model.
- And finally we parse the output as a string.

```
from operator import itemgetterfrom langchain.schema.runnable import RunnablePassthrough, RunnableLambdadef prompt_func(dict):    format_texts = "\n".join(dict["context"]["texts"])    return [        HumanMessage(            content=[                {"type": "text", "text": f"""Answer the question based only on the following context, which can include text, tables, and the below image:Question: {dict["question"]}Text and tables:{format_texts}"""},                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dict['context']['images'][0]}"}},            ]        )    ]model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)chain = (    {"context": retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}    | RunnableLambda(prompt_func)    | model    | StrOutputParser()      )
```

Now, we are ready to test our Multi-retrieval Rag

```
chain.invoke(    "What is the change in wild fires from 1993 to 2022?")
```

here is the answer :

> Based on the provided chart, the number of wildfires has increased from 1993 to 2022. The chart shows a line graph with the number of fires in thousands, which appears to start at a lower point in 1993 and ends at a higher point in 2022. The exact numbers for 1993 are not provided in the text or visible on the chart, but the visual trend indicates an increase.
> 
> Similarly, the acres burned, represented by the shaded area in the chart, also show an increase from 1993 to 2022. The starting point of the shaded area in 1993 is lower than the ending point in 2022, suggesting that more acres have been burned in 2022 compared to 1993. Again, the specific figures for 1993 are not provided, but the visual trend on the chart indicates an increase in the acres burned over this time period.to do

**References** :

[https://python.langchain.com/docs/modules/data\_connection/retrievers/multi\_vector](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)

[https://blog.langchain.dev/semi-structured-multi-modal-rag/](https://blog.langchain.dev/semi-structured-multi-modal-rag/)