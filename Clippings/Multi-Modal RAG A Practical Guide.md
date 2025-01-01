---
title: "Multi-Modal RAG: A Practical Guide"
source: "https://gautam75.medium.com/multi-modal-rag-a-practical-guide-99b0178c4fbb"
author:
  - "[[Gautam Chutani]]"
published: 2024-09-17
created: 2025-01-01
description: "MultiModal RAG is an advanced approach to information retrieval and generation that combines the strengths of multiple content types, primarily text and images. Unlike traditional RAG systems that…"
tags:
  - "clippings"
---
## Using vLLM to serve models for Multimodal Text Summarization, Table Processing, and Answer Synthesis

## What is MultiModal RAG?

**MultiModal [[RAG]]** is an advanced approach to information retrieval and generation that combines the strengths of multiple content types, primarily text and images. Unlike traditional RAG systems that focus solely on text, MultiModal RAG harnesses the power of both textual and visual information, providing a more comprehensive and context-rich foundation for generating responses.

The importance of MultiModal RAG cannot be overstated in our increasingly visual world. Many documents, from research papers to business reports, contain a mixture of text, images, charts, and tables. By incorporating visual elements into the retrieval and generation process, MultiModal RAG systems can:

1\. Capture nuances lost in text-only analysis  
2\. Provide more accurate and contextually relevant responses  
3\. Enhance understanding of complex concepts through visual aids  
4\. Improve the overall quality and depth of generated content

## Strategies for Implementing Multimodal RAG

There are several ways to develop a MultiModal RAG pipeline, each with its own strengths and considerations:

## Joint Embedding and Retrieval

- Utilize models like [**CLIP**](https://github.com/openai/CLIP) (Contrastive Language-Image Pre-training) or [**ALIGN**](https://huggingface.co/docs/transformers/en/model_doc/align) (A Large-scale ImaGe and Noisy-text embedding) to create unified embeddings for both text and images.
- Implement approximate nearest neighbor search using libraries like [**FAISS**](https://github.com/facebookresearch/faiss) or [**Annoy**](https://github.com/spotify/annoy) for efficient retrieval.
- Feed retrieved multimodal content (raw images and text chunks) to a multimodal LLM such as LLaVa, Pixtral 12B, GPT-4V, Qwen-VL for answer generation.

## Image-to-Text Conversion

- Generate Summaries from images using models like [**LLaVA**](https://huggingface.co/docs/transformers/en/model_doc/llava), or [**FUYU-8b**](https://huggingface.co/adept/fuyu-8b).
- Use text-based embedding models like [**Sentence-BERT**](https://arxiv.org/pdf/1908.10084) to create embeddings for both original text and image captions.
- Pass the text chunks to an LLM for final answer synthesis.

## Hybrid Retrieval with Raw Image Access

- Employ a multimodal LLM to produce text summaries from images.
- Embed and retrieve these summaries with references to raw images, alongside other textual chunks. This can be achieved via Multi-Vector Retriever with [[vector databases]] like Chroma, Milvus to store raw text and images along with their summaries for retrieval.
- For final answer generation, use multimodal models like [**Pixtral 12B**](https://huggingface.co/mistralai/Pixtral-12B-2409), [**LLaVa**](https://github.com/haotian-liu/LLaVA), [**GPT-4V**](https://platform.openai.com/docs/guides/vision), [**Qwen-VL**](https://github.com/QwenLM/Qwen-VL) that can process both text and raw image inputs simultaneously.

## Practical Implementation

In this article, we’ll explore the third approach, leveraging a powerful combination of cutting-edge tools to create an efficient and effective MultiModal RAG system:

1. [**Unstructured**](https://github.com/Unstructured-IO/unstructured): For parsing images, text, and tables from various document formats, including PDFs.
2. [**LLaVa**](https://huggingface.co/llava-hf/llava-1.5-7b-hf) via [**vLLM**](https://github.com/vllm-project/vllm): Powered by the vLLM serving engine, this setup uses Vision Language Model (VLM) named LLaVA (`llava-hf/llava-1.5-7b-hf`) to handle text/table summarization and multimodal tasks like image summarization and answer generation from integrated textual and visual inputs. While not the most advanced model, LLaVA is highly efficient and computationally inexpensive. Thanks to vLLM, it can be seamlessly deployed on a CPU, making it an ideal, cost-effective solution for those looking to balance performance with resource efficiency.
3. [**Chroma DB**](https://www.trychroma.com/): As our vector database to store text chunks, table summaries, and image summaries alongside their raw images. Coupled with its [MultiVector Retriever](https://python.langchain.com/docs/how_to/multi_vector/) feature, it provides a robust storage and retrieval system for our multimodal system.
4. [**LangChain**](https://python.langchain.com/docs/introduction/): As the orchestration tool to seamlessly integrate these components together.

By combining these tools, we’ll demonstrate how to build a robust MultiModal RAG system that can process diverse document types, generate high-quality summaries, and produce comprehensive answers that leverage both textual and visual information.

## Download Data

We will use this [blog post](https://www.onlycfo.io/p/2024-gtm-benchmarks) as our document source, as it contains valuable information presented through charts and tables in the form of images.

```
import osimport ioimport reimport uuidimport base64import shutilimport requestsfrom tqdm import tqdmfrom PIL import Imageimport matplotlib.pyplot as pltfrom IPython.display import HTML, displayfrom unstructured.partition.pdf import partition_pdffrom langchain_core.documents import Documentfrom langchain_text_splitters import CharacterTextSplitterfrom langchain.storage import InMemoryStorefrom langchain_chroma import Chromafrom langchain.chains.llm import LLMChain, PromptTemplatefrom langchain_core.messages import HumanMessage, SystemMessagefrom langchain_core.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)from langchain_core.output_parsers import StrOutputParserfrom langchain_core.runnables import RunnableLambda, RunnablePassthroughfrom langchain.retrievers.multi_vector import MultiVectorRetrieverfrom openai import OpenAI as OpenAI_vLLMfrom langchain_community.llms.vllm import VLLMOpenAIfrom langchain.embeddings import HuggingFaceEmbeddingsembeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')
```
```
os.mkdir("data")shutil.move("gtm_benchmarks_2024.pdf", "data")
```

## Extract Text, Tables and Images from PDF document

Once we have the PDF downloaded, we will utilize `unstructured.io` library to process our document and extract the contents.

```
def extract_pdf_elements(path, fname):    """    Extract images, tables, and chunk text from a PDF file.    path: File path, which is used to dump images (.jpg)    fname: File name    """    return partition_pdf(        filename=path + fname,        extract_images_in_pdf=True,        infer_table_structure=True,        chunking_strategy="by_title",        max_characters=4000,        new_after_n_chars=3800,        combine_text_under_n_chars=2000,        image_output_dir_path=path,    )def categorize_elements(raw_pdf_elements):    """    Categorize extracted elements from a PDF into tables and texts.    raw_pdf_elements: List of unstructured.documents.elements    """    tables = []    texts = []    for element in raw_pdf_elements:        if "unstructured.documents.elements.Table" in str(type(element)):            tables.append(str(element))        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):            texts.append(str(element))    return texts, tables
```
```
folder_path = "./data/"file_name = "gtm_benchmarks_2024.pdf"raw_pdf_elements = extract_pdf_elements(folder_path, file_name)texts, tables = categorize_elements(raw_pdf_elements)text_splitter = CharacterTextSplitter.from_tiktoken_encoder(    chunk_size = 1000, chunk_overlap = 0)joined_texts = " ".join(texts)texts_token = text_splitter.split_text(joined_texts)print("No of Textual Chunks:", len(texts))print("No of Table Elements:", len(tables))print("No of Text Chunks after Tokenization:", len(texts_token))
```

## Generate Table summaries

We will use the vLLM engine, running on a CPU machine, to serve the 7B parameter LLaVA model (`llava-hf/llava-1.5-7b-hf`) for generating table summaries. We can use text-based LLMs also as we typically do in any RAG system but for now, we will make use of LLaVa model itself which can process both text and images.

We generate table summaries in order to enhance the natural language retrieval. These summaries are essential for retrieving raw tables and text chunks efficiently.

> To setup vLLM serving engine, you can refer [here](https://medium.com/@gautam75/vllm-efficient-serving-with-scalable-performance-cb72c155b89e).

```
llm_client = VLLMOpenAI(    base_url = "http://localhost:8000/v1",    api_key = "dummy",    model_name = "llava-hf/llava-1.5-7b-hf",    temperature = 1.0,    max_tokens = 300)
```
```
def generate_text_summaries(texts, tables, summarize_texts=False):    """    Summarize text elements    texts: List of str    tables: List of str    summarize_texts: Bool to summarize texts    """        prompt_text = """You are an assistant tasked with summarizing tables for retrieval. \    Give a concise summary of the table that is well optimized for retrieval. Make sure to capture all the details. \    Input: {element} """    prompt = ChatPromptTemplate.from_template(prompt_text)        summarize_chain = {"element": lambda x: x} | prompt | llm_client | StrOutputParser()        text_summaries = []    table_summaries = []        if texts and summarize_texts:        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})    elif texts:        text_summaries = texts            if tables:        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 3})    return text_summaries, table_summariestext_summaries, table_summaries = generate_text_summaries(    texts_token, tables, summarize_texts=False)print("No of Text Summaries:", len(text_summaries))print("No of Table Summaries:", len(table_summaries))
```

## Generate Image summaries

Now, we will make use of our Vision Language Model (VLM) to generate the image summaries.

*Note: Images can be made available to the model in two main ways: by passing a link to the image or by passing the base64 encoded image directly in the request.*

```
api_key = "dummy"base_url = "http://localhost:8000/v1"vlm_client = OpenAI_vLLM(    api_key = api_key,    base_url = base_url,)
```
```
def encode_image(image_path):    """Getting the base64 string"""    with open(image_path, "rb") as image_file:        return base64.b64encode(image_file.read()).decode("utf-8")def image_summarize(img_base64, prompt):    """Make image summary"""    chat_response = vlm_client.chat.completions.create(        model="llava-hf/llava-1.5-7b-hf",        max_tokens=1024,        messages=[{            "role": "user",            "content": [                {"type": "text", "text": prompt},                {                    "type": "image_url",                    "image_url": {                        "url": f"data:image/jpeg;base64,{img_base64}",                    },                },            ],        }],        stream=False    )    return chat_response.choices[0].message.content.strip()def generate_img_summaries(path):    """    Generate summaries and base64 encoded strings for images    path: Path to list of .jpg files extracted by Unstructured    """        img_base64_list = []        image_summaries = []        prompt = """You are an assistant tasked with summarizing images for optimal retrieval. \    These summaries will be embedded and used to retrieve the raw image.    Write a clear and concise summary that captures all the important information, including any statistics or key points present in the image."""        for img_file in tqdm(sorted(os.listdir(path))):        if img_file.endswith(".jpg"):            img_path = os.path.join(path, img_file)            base64_image = encode_image(img_path)            img_base64_list.append(base64_image)                        generated_summary = image_summarize(base64_image, prompt)            print(generated_summary)            image_summaries.append(generated_summary)    return img_base64_list, image_summariesimg_base64_list, image_summaries = generate_img_summaries(folder_path)assert len(img_base64_list) == len(image_summaries)
```

## Store and Index Document Summaries

To configure Multi-Vector Retriever, we will store the raw documents including texts, tables, and images in the docstore, while indexing their summaries in the vectorstore to enhance semantic retrieval efficiency.

```
def create_multi_vector_retriever(    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):    """    Create retriever that indexes summaries, but returns raw images or texts    """        store = InMemoryStore()    id_key = "doc_id"        retriever = MultiVectorRetriever(        vectorstore=vectorstore,        docstore=store,        id_key=id_key,    )        def add_documents(retriever, doc_summaries, doc_contents):        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]        summary_docs = [            Document(page_content=s, metadata={id_key: doc_ids[i]})            for i, s in enumerate(doc_summaries)        ]        retriever.vectorstore.add_documents(summary_docs)        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))            if text_summaries:        add_documents(retriever, text_summaries, texts)        if table_summaries:        add_documents(retriever, table_summaries, tables)        if image_summaries:        add_documents(retriever, image_summaries, images)    return retrievervectorstore = Chroma(    collection_name="mm_rag_vectorstore", embedding_function=embeddings, persist_directory="./chroma_db" )retriever_multi_vector_img = create_multi_vector_retriever(    vectorstore,    text_summaries,    texts,    table_summaries,    tables,    image_summaries,    img_base64_list,)
```

## Setup for Multi-Vector Retrieval

Next we define functions and configurations for handling and processing text data and base64-encoded images, including resizing images and formatting prompts for the model. It sets up a multi-modal retrieval and generation (RAG) context chain to integrate and analyze both text and image data for answering user queries.

Since we are serving our vision language model with vLLM’s HTTP server that is compatible with [OpenAI Vision API](https://platform.openai.com/docs/guides/vision) (Chat Completion API), so to setup the context for the model, we are following the specific chat template that can be found [here](https://github.com/vllm-project/vllm/blob/main/examples/template_llava.jinja).

```
def plt_img_base64(img_base64):    """Disply base64 encoded string as image"""        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'        display(HTML(image_html))def looks_like_base64(sb):    """Check if the string looks like base64"""    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not Nonedef is_image_data(b64data):    """    Check if the base64 data is an image by looking at the start of the data    """    image_signatures = {        b"\xff\xd8\xff": "jpg",        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",        b"\x47\x49\x46\x38": "gif",        b"\x52\x49\x46\x46": "webp",    }    try:        header = base64.b64decode(b64data)[:8]          for sig, format in image_signatures.items():            if header.startswith(sig):                return True        return False    except Exception:        return Falsedef resize_base64_image(base64_string, size=(64, 64)):    """    Resize an image encoded as a Base64 string    """        img_data = base64.b64decode(base64_string)    img = Image.open(io.BytesIO(img_data))        resized_img = img.resize(size, Image.LANCZOS)        buffered = io.BytesIO()    resized_img.save(buffered, format=img.format)        return base64.b64encode(buffered.getvalue()).decode("utf-8")def split_image_text_types(docs):    """    Split base64-encoded images and texts    """    b64_images = []    texts = []    for doc in docs:                if isinstance(doc, Document):            doc = doc.page_content        if looks_like_base64(doc) and is_image_data(doc):            doc = resize_base64_image(doc, size=(64, 64))            b64_images.append(doc)        else:            texts.append(doc)    return {"images": b64_images, "texts": texts}def img_prompt_func(data_dict):    """    Join the context into a single string    """    formatted_texts = "\n".join(data_dict["context"]["texts"])    messages = []        text_message = {        "type": "text",        "text": (            "You are an AI assistant with expertise in finance and business metrics.\n"            "You will be given information that may include text, tables, and charts related to business performance and industry trends.\n"            "Your task is to analyze this information and provide a clear, concise answer to the user's question.\n"            "Focus on the most relevant data points and insights that directly address the user's query.\n"            f"User's question: {data_dict['question']}\n\n"            "Information provided:\n"            f"{formatted_texts}"        ),    }    messages.append(text_message)        if data_dict["context"]["images"]:        for image in data_dict["context"]["images"]:            image_message = {                "type": "image_url",                "image_url": {"url": f"data:image/jpeg;base64,{image}"},            }            messages.append(image_message)    return [HumanMessage(content=messages)]def multi_modal_rag_context_chain(retriever):    """Multi-modal RAG context chain"""    chain = (        {            "context": retriever | RunnableLambda(split_image_text_types),            "question": RunnablePassthrough(),        }        | RunnableLambda(img_prompt_func)    )    return chainchain_multimodal_context = multi_modal_rag_context_chain(retriever_multi_vector_img)
```

## Examine Retrieval

```
query = "How has the median YoY ARR growth rate for public SaaS companies changed from 2020 to 2024?"docs = retriever_multi_vector_img.invoke(query)
```
```
plt_img_base64(docs[0])
```
![](https://miro.medium.com/v2/resize:fit:700/1*F1Uw74aOG7YklRw7LLS8JA.png)

Retrieved image

## Run RAG Pipeline for Answer Generation

Since our current model does not support very long context and multiple multi-modal items per text prompt, so we will modify the retrieved context and test the final answer synthesis part.

```
context = chain_multimodal_context.invoke(query)[0].content
```
```
context = [    {        'type': 'text',        'text': "You are an AI assistant with expertise in finance and business metrics.\nYou will be given information that may include text, tables, and charts related to business performance and industry trends.\nYour task is to analyze this information and provide a clear, concise answer to the user's question.\nFocus on the most relevant data points and insights that directly address the user's query.\nUser's question: How has the median YoY ARR growth rate for public SaaS companies changed from 2020 to 2024?"    },    {        'type': 'image_url',        'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3wRoQOP1o8pPT9abFKX3Axuu1toJH3vce1SbvY0AN8pPT9axf7Zn+0zQDQr/MblRIQAjjdjcDnp0P0zW5u9jUVxCtxCY2MqAkHdG20j8RQBiW2uXM0sUcvh7UIfMkCbiAVQZI3Me3Az3re8pPT9aZbwi2iEatK4BJzI248+9S7vY0AN8pPT9arXcy20lqnlFvPl8rO7G35Sc+/TH41b3exqKWJJWV2D5UHGGIH5U1bqJ3toLHMj7gA3yNtOVI5x29arpq9hI1uqXSObgssW3kOV+8AfaoLxhLc2ssDXDyQSMhjiYKmSv/AC0J7fnyaZDY3SpEPNsLZUZjtt7fkA/3STgHpk45pDLFpq1tdRW7MHt5JywjhnXa7bevH61cWVHUMjBlPQryDWYbG5hlhZJ4bsRlvmu1/eLn+66jj8qr2TPZzWdnElvZRIJDLbM5cv3Bjbv6n69KAN3cPf8AI0bh7/karWWoW2oWkdzA+Y5ASu4bTwcdDVncv94UAG4e/wCRpCwIPXp6Uu5fUUhYEHkdKAESRGBCupKna2D0PoadkeopqKgBKquSctgd/enYHoKADI9RUU8ENygWVQcHKnOCp9QexqXA9BRgegoAxZLWeG6tp50N60CuqXKHa8efVBw31Hp0qXT724lhtPKkiv4WV/OuVYKVPVRt/TtWrgegqtPp1pczpPLCDMgIWRSVYA+457UAV4NTnn+yg6dcRmeN2bfj90R0DfXtSQXGpyS2wntYIY3iYzjzdzI+eAvqP8aeukQARhpryRUUrh7lyD06889Kkt9MsbIA21pDEyIUDKg3BeOM9ew/KgCxHDGm4qigudzEDqfWn7V9BVKS33TJJFcCNScyLyd/AHrx0pi2so8vdfBsZ3/Ljd+vFAGhtX0FG1fQVnrayjZuvw2M7vlxu/XihbWYeXuvw20Hf8uN3p34oA0Nq+go2r6Cs9bWYbN1+DgHd8v3vTvxQtrMNmb8HCkN8uNx7HrxjigDQ2r6CkKgA8DpVAWsw2ZvwcKQ3y9Tzg9eMcflTo7eVHjZr0OFB3rtxuPPPXjt+VAH/9k='}    }]
```
```
chat_response = vlm_client.chat.completions.create(    model="llava-hf/llava-1.5-7b-hf",    messages=[{        "role": "user",        "content": context,    }],    stream=True)for chunk in chat_response:    if chunk.choices[0].delta.content:        print(chunk.choices[0].delta.content, end="", flush=True)
```

Based on the provided user question and retrieved text snippet and image, the model will now start streaming its response.

## Considerations

1. To demonstrate image retrieval, larger (4k token) text chunks were produced initially and then summarized. However, this is not always the case, and other methods may be required to ensure accurate and efficient chunking and indexing.
2. The summarization and answer quality appears to be sensitive to image size and resolution.
3. The VLM used currently supports Single-image input only.
4. The model may struggle to understand certain visual elements such as graphs or complex flow-charts.
5. The model may generate incorrect descriptions or captions in certain scenarios. For eg, may give false information while asking statistical questions.
6. If text within image is unclear or blur, the model will do its best to interpret it. However, the results may be less accurate.

## Future Scope

1. Test accuracy with VLMs such as Pixtral-12B that have longer context window and also support passing multiple images per message and/or pass multi-turn conversations.
2. Enable more sophisticated interaction between text and image modalities, where the model can dynamically prioritize visual or textual information based on the question.
3. Introduce more refined summarization techniques for visual content to generate better semantic representations of images.
4. Since we are using vLLM to serve the model, it would be interesting to see how the performance varies if we run the same model on CPU and GPU with different optimizations.
5. And last but not the least, use better chunking and retrieval mechanism.

## Conclusion

In conclusion, MultiModal RAG systems represent a significant advancement in information retrieval and processing. This technology opens doors to enhanced decision-making processes across various sectors, from healthcare and finance to education and autonomous systems.

Looking ahead, we anticipate further integration of diverse data types and improved real-time processing capabilities. As these systems evolve, they promise to revolutionize how we interact with and leverage information, paving the way for more intuitive and powerful AI solutions. The future of MultiModal RAG systems is not just about technological advancement; it’s about transforming how we understand and utilize the vast array of information available to us, ultimately leading to more informed decisions and innovative solutions across industries.

## References

1. [https://blog.langchain.dev/semi-structured-multi-modal-rag/](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
2. [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
3. [https://python.langchain.com/docs/how\_to/multi\_vector/](https://python.langchain.com/docs/how_to/multi_vector/)
4. [https://docs.vllm.ai/en/latest/models/vlm.html](https://docs.vllm.ai/en/latest/models/vlm.html)
5. [https://platform.openai.com/docs/guides/vision](https://platform.openai.com/docs/guides/vision)
6. [https://cs.stanford.edu/~myasu/blog/racm3/](https://cs.stanford.edu/~myasu/blog/racm3/)
7. [https://medium.com/@bijit211987/multimodal-retrieval-augmented-generation-mm-rag-2e8f6dc59f11](https://medium.com/@bijit211987/multimodal-retrieval-augmented-generation-mm-rag-2e8f6dc59f11)
8. [https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117](https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117)