# LLM-Motes

# Lecture 1: Introduction to LLMs

**Historical Context of LLMs**
- Early chatbots like ELIZA, highlighting significant advancements in natural language processing .

**Open Source vs. Closed Source Models**
<br>
<img src="/images/closed-source-vs-open-weight-models.jpg"/>
- The differences between open-source and closed-source models, noting that open-source models like Facebook's LLaMA 3.1 are becoming increasingly competitive with closed-source models like GPT-4 .

**Generative AI and Its Applications**
- Generative AI encompasses various fields, including text, video, and audio. The lecture talks about the capabilities of generative AI, emphasizing its transformative potential .

**Job Market Insights**
<br>
<img src="/images/Generative AI Market Size.jpg" />
- The global generative AI job market is projected to grow significantly, making skills in LLMs increasingly valuable.

Book we're following :  Building Large Language Model by Sebastian Raschka

# Lecture 2: What really is LLMs?

**1. Introduction to Large Language Models (LLMs)**
- LLMs are neural networks designed to understand, generate, and respond to human-like text .
- They are trained on massive datasets and can perform a variety of natural language processing tasks.

**2. Definition and Characteristics of LLMs**
<br>
- An LLM is fundamentally a deep neural network that processes text .
- The term "large" refers to the vast number of parameters (often billions or even trillions) that these models contain, which enhances their performance .

**3. Comparison with Earlier NLP Models**
- Earlier NLP models were task-specific (e.g., translation or sentiment analysis), while LLMs are versatile and can handle multiple tasks .
- Modern LLMs, like ChatGPT, can perform complex tasks such as drafting emails, which were challenging for earlier models .

**4. The Secret Sauce: Transformer Architecture**
- The key to LLMs' effectiveness lies in the Transformer architecture, introduced in the influential paper "Attention is All You Need" .
- This architecture allows for efficient processing of input data through mechanisms like multi-head attention and positional encoding .

**5. Parameter of GPTs**
<img src="/images/image1" />
<img src="/images/image2" />


**5. Terminology Clarification**
- **Artificial Intelligence (AI)**: The broadest category, encompassing any machine that mimics human intelligence .
- **Machine Learning (ML)**: A subset of AI that involves systems learning from data .
- **Deep Learning (DL)**: A further subset of ML that primarily uses neural networks .
- **Generative AI**: Combines LLMs with deep learning to create various forms of media, not limited to text .

**6. Applications of LLMs**
- **Content Creation**: LLMs can generate new text, such as poems or articles .
- **Chatbots**: They can serve as virtual assistants, automating customer service interactions .
- **Machine Translation**: LLMs can translate text accurately into different languages .
- **Sentiment Analysis**: They can analyze text to determine sentiment, useful for monitoring social media .

# Lecture 3 ~ Pretraining LLMs vs Finetuning LLMs

**Introduction to LLMs**
+ Creating LLMs = Pre-Training + Fine-Tuning

- Large Language Models (LLMs) are advanced AI systems designed to understand and generate human-like text. The lecture series covers the stages of building LLMs, starting from foundational concepts to practical applications .

**Key Terminologies**
- **LLM**: Refers to Large Language Models, derived from earlier NLP models.
- **Gen AI**: Generative AI, a subset of AI focused on creating content.

**Stages of Building LLMs**
1. **Pre-Training**:
   - Involves training on a large and diverse dataset to learn language patterns and structures .
   - Example: GPT-3 was trained on 300 billion tokens from various sources, including Common Crawl, web text, books, and Wikipedia .

2. **Fine-Tuning**:
   - A refinement process where the pre-trained model is trained on a narrower dataset specific to a particular application .
   - Necessary for applications requiring specialized knowledge, such as customer service chatbots or legal tools .

**Pre-Training Details**
- Pre-training allows LLMs to perform various tasks like translation, question answering, and sentiment analysis without specific training for each task .
- The training process is computationally intensive, costing around $4.6 million for GPT-3 .

**Fine-Tuning Examples**
- **SK Telecom**: Fine-tuned an LLM for telecom-related customer service, resulting in improved performance metrics .
- **Harvey**: An AI tool for attorneys, fine-tuned on legal case history to provide accurate legal assistance .
- **JP Morgan**: Developed a fine-tuned LLM for internal research analysis, leveraging proprietary data .

**Conclusion of Stages**
- The process of building an LLM consists of collecting data, pre-training on a large corpus, and fine-tuning for specific applications. Understanding the difference between pre-training (unsupervised learning) and fine-tuning (supervised learning with labeled data) is crucial for effective model deployment .

**Visual Representation**
- A schematic illustrating the stages of LLM development emphasizes the importance of data collection, computational power, and the distinction between foundational and fine-tuned models .

+ Two types of fine-tuning
a. Labeled dataset consisting of instruction-answer pairs, text translation, and airline customer support.
b. Labeled dataset consisting of text and associated labels. eg: emails -> spam or non-spam

This summary encapsulates the essential concepts and processes involved in building large language models, highlighting the significance of both pre-training and fine-tuning stages.

# L4 ~ Introduction of Transformers and Large Language Models (LLMs)

- **Transformers**: Transformers are a deep neural network architecture introduced in the 2017 paper "Attention is All You Need." This architecture is fundamental to modern LLMs, enabling significant advancements in tasks like machine translation .

+ This paper was initially introduced for translation task. Text completion was not in consideration itself.

**Key Concepts**

- **Architecture**: The Transformer architecture consists of two main components: the **encoder** and the **decoder**. The encoder converts input text into vector embeddings, while the decoder generates output text from these embeddings and partial outputs .

- **Self-Attention Mechanism**: This mechanism allows the model to weigh the importance of different words relative to each other, capturing long-range dependencies in text. It enables the model to consider context from previous words when predicting the next word .

**Steps in Transformer Architecture**

1. **Input Text**: The process begins with input text that needs translation .
2. **Pre-processing**: The text is tokenized, breaking it down into individual tokens and assigning unique IDs . There are mutliple popular tokenizers.
3. **Encoder**: The token IDs are passed to the encoder, which converts them into vector embeddings that capture semantic meanings. Can we represent tokens in such a way that it captures the semantic meaning of the text? Vector Embedding! & We'll project these tokens into higher dimension
4. **Decoder**: The decoder receives the vector embeddings and partial output text, generating the final output word by word(this is important) by arriving at the final output layer.

+ 97 times attention repeated in attention paper:)
 
 **Self attention mechanism**
- Allows models to weigh importance of different words/tokens relative to each other.
- Enables model to capture long-range dependencies which was lacking for RNNs(1980) & LSTMs(1997)  

**Differences Between Models**

- **BERT vs. GPT**: BERT (Bidirectional Encoder Representations from Transformers) predicts masked words in a sentence, looking at context from both directions, making it effective for tasks like sentiment analysis. In contrast, GPT (Generative Pre-trained Transformers) generates text one word at a time, focusing only on left context .

+ GPT model only has the decoder & doesn't have encoder.
+ Wheras BERT model only has the encoder.

- **Transformers vs. LLMs**: Not all Transformers are LLMs; Transformers can also be applied to other tasks like computer vision(Vision Transformer). Similarly, not all LLMs are based on Transformers; they can also utilize recurrent or convolutional architectures .

**Conclusion of Lecture**: The lecture emphasizes understanding the foundational concepts of Transformers and LLMs, including their architecture, mechanisms, and differences.
