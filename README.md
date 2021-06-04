# semantic-search-with-sbert

# Semantic Search

Created: May 7, 2021 5:06 PM
Created By: Subir Verma
Git: https://github.com/nlpsbr/search-api
Last Edited By: Subir Verma
Last Edited Time: May 20, 2021 6:12 PM
Stakeholders: Subir Verma
Status: In Progress 🙌
Type: Architecture Overview

# Overview

Building In-house Semantic Search Engine - Fast and Accurate

Semantic search seeks to improve search accuracy by understanding the content of the search query. In contrast to traditional search engines, that only finds documents based on lexical matches, semantic search can also find synonyms.

The idea behind semantic search is to embedd all entries in your corpus, which can be sentences, paragraphs, or documents, into a vector space.

At search time, the query is embedded into the same vector space and the closest embedding from your corpus are found. These entries should have a high semantic overlap with the query.

### Problem Statement

- Currently, users on platform find difficult to look for courses/blogs of their needs
- Building something which is Fast, Scalable and most importantly relevant to students/users is challenging.

### Proposed Solution

- For Scalability and Speed we use Fast API and FAISS indexing techniques.
- Fine-tuning transformer model on our dataset.

# Success Criteria

- Convergence Metric: Students purchasing new courses and spending more time on platform

# Semantic Architecture

- pass all passages in our collection through a trained T5 model, which generates potential queries from users.
- use these (query, passage) pairs to train a SentenceTransformer bi-encoder.

    ![Semantic%20Search%20af53aff787ca4b9fb3101ced00bb1875/Screenshot_2021-05-17_at_12.38.40_PM.png](Semantic%20Search%20af53aff787ca4b9fb3101ced00bb1875/Screenshot_2021-05-17_at_12.38.40_PM.png)

    # Step 1 - Query Generation

    ```python
    fromtransformersimport T5Tokenizer, T5ForConditionalGeneration
    importtorchtokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    model.eval()

    para = "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects."

    input_ids = tokenizer.encode(para, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=3)

    print("Paragraph:")
    print(para)

    print("\nGenerated Queries:")
    for iin range(len(outputs)):
        query = tokenizer.decode(outputs[i], skip_special_tokens=True)
        print(f'{i + 1}:{query}')
    ```

    # Step 2 - Bi-Encoder Training

    ## Model Architecture

    - SentenceTransformers was designed in such way that fine-tuning your own sentence / text embeddings models is easy. It provides most of the building blocks that we can stick together to tune embeddings for our specific task.
    - Note: there is no single training strategy that works for all use-cases. Instead, which training strategy to use greatly depends on our available data and on our target task.
    - For sentence / text embeddings, we want to map a variable length input text to a fixed sized dense vector. The most basic network architecture we can use is the following

    ![Semantic%20Search%20af53aff787ca4b9fb3101ced00bb1875/Screenshot_2021-05-18_at_2.42.57_PM.png](Semantic%20Search%20af53aff787ca4b9fb3101ced00bb1875/Screenshot_2021-05-18_at_2.42.57_PM.png)

    - we can create the networks architectures from scratch by defining the individual layers. For example, the following code would create the depicted network architecture:

    ```python
    fromsentence_transformersimport SentenceTransformer, models

    word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    ```

    - We can also construct more complex models:

    ```python
    fromsentence_transformersimport SentenceTransformer, models
    fromtorchimport nn

    word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    ```

    ## Loss Function

    ![Semantic%20Search%20af53aff787ca4b9fb3101ced00bb1875/Screenshot_2021-05-20_at_6.12.21_PM.png](Semantic%20Search%20af53aff787ca4b9fb3101ced00bb1875/Screenshot_2021-05-20_at_6.12.21_PM.png)

    - With the generated queries, we can then train a bi-encoder using the use MultipleNegativesRankingLoss.
    - MultipleNegativesRankingLoss is a great loss function if you only have positive pairs, for example, only pairs of similar texts like pairs of paraphrases, pairs of duplicate questions, pairs of (query, response), or pairs of (source_language, target_language).
    - This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)…, (a_n, p_n) where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
    - The performance usually increases with increasing batch sizes.
    - You can also provide one or multiple hard negatives per anchor-positive pair by structuring the data like this: (a_1, p_1, n_1), (a_2, p_2, n_2)

    ```python
    from sentence_transformersimport SentenceTransformer,  SentencesDataset, LoggingHandler, losses
    from sentence_transformers.readersimport InputExample

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
        InputExample(texts=['Anchor 2', 'Positive 2'])]
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    ```

    ## Result Comparison

    model1: fine-tuned model

    model2: pre-trained model

    ![Semantic%20Search%20af53aff787ca4b9fb3101ced00bb1875/Screenshot_2021-05-18_at_3.49.15_PM.png](Semantic%20Search%20af53aff787ca4b9fb3101ced00bb1875/Screenshot_2021-05-18_at_3.49.15_PM.png)
