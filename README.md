# Fine tunning Large-Language-Models

## Supervised fine tuning (in 5 steps)
1. Choose a fine-tuning task
2. Prepare training dataset
3. choose a base model
4. Fine-tune model via supervised learning
5. Evaluate model performance

   
Fine-tuning is the process of taking a pre-trained model and adjusting at least one internal parameter (i.e., weights) during training. In the context of Large Language Models (LLMs), fine-tuning transforms a general-purpose base model (e.g., GPT-3) into a specialized model tailored for a specific use case (e.g., ChatGPT).

The primary advantage of fine-tuning is that it allows models to achieve improved performance with fewer manually labeled examples compared to models trained solely via supervised learning.

While base self-supervised models can perform well across various tasks with the aid of prompt engineering, they may still generate completions that are not entirely accurate or helpful. For instance, comparing completions between the base GPT-3 model (davinci) and a fine-tuned model (text-davinci-003) reveals that the fine-tuned model provides more helpful responses. The fine-tuning process used for text-davinci-003, known as alignment tuning, aims to enhance the helpfulness, honesty, and harmlessness of the LLM's responses.

The benefits of fine-tuning include:

1. **Improved Performance:** A smaller, fine-tuned model can often outperform larger, more expensive models on specific tasks for which it was trained.
2. **Efficiency:** Despite being smaller in size, fine-tuned models can achieve better task performance, as demonstrated by OpenAI's InstructGPT models, which outperformed the larger GPT-3 base model.
3. **Domain-specific Knowledge:** Fine-tuning allows models to learn domain-specific information during the training process, mitigating the issue of finite context windows and improving performance on tasks requiring specialized knowledge.
4. **Lower Inference Costs:** Fine-tuned models can reduce the need for additional context in prompts, resulting in lower inference costs.

# 3 Ways to Fine-tune Language Models

Fine-tuning a language model is a crucial step in leveraging its power for specific tasks. There are three primary approaches to fine-tuning: self-supervised learning, supervised learning, and reinforcement learning. These approaches can be used individually or in combination to enhance the performance of the model.

## Self-supervised Learning

Self-supervised learning involves training a model based on the inherent structure of the training data. In the case of Language Model Models (LLMs), this often entails predicting the next word given a sequence of words or tokens. Self-supervised learning can be utilized to develop models that mimic a person’s writing style using example texts.

## Supervised Learning

Supervised learning is a widely used method for fine-tuning language models. It involves training the model on input-output pairs specific to a particular task. For instance, instruction tuning aims to enhance model performance in tasks like question answering or responding to user prompts. Creating question-answer pairs and integrating them into prompt templates is essential for supervised learning.

## Reinforcement Learning

Reinforcement learning (RL) is another approach to fine-tuning language models. RL employs a reward model to guide the training process by scoring model completions based on human preferences. OpenAI's InstructGPT models exemplify how RL can be utilized for model fine-tuning through multiple steps involving supervised learning and RL algorithms like Proximal Policy Optimization (PPO).

### Supervised Fine-tuning Steps (High-level)

For this article's purpose, we will delve into the supervised fine-tuning approach. Below is a high-level outline of the steps involved:

1. **Choose fine-tuning task**: Select the specific task for fine-tuning, such as summarization, question answering, or text classification.
2. **Prepare training dataset**: Create input-output pairs and preprocess the data by tokenizing, truncating, and padding text.
3. **Choose a base model**: Experiment with different pre-trained models and select the one that best suits the task.
4. **Fine-tune model via supervised learning**: Train the model using the prepared dataset and evaluate its performance.

While each step in the fine-tuning process is critical, we will primarily focus on the training procedure in this article. Understanding how to effectively train a fine-tuned model is essential for achieving optimal performance in various natural language processing tasks.

# 3 Options for Parameter Training

When fine-tuning a model with a substantial number of parameters, ranging from ~100M to 100B, it's essential to consider computational costs and the strategy for parameter training. This article explores three generic options for parameter training and discusses their implications.

## Option 1: Retrain All Parameters

The simplest approach is to retrain all internal model parameters, known as full parameter tuning. While straightforward, this method is computationally expensive and may lead to catastrophic forgetting, where the model forgets useful information learned during initial training.

## Option 2: Transfer Learning

Transfer learning (TL) involves preserving the useful representations/features learned by the model from past training when applying it to a new task. TL typically entails replacing the final layers of the neural network with new ones while leaving the majority of parameters unchanged. While TL reduces computational costs, it may not address the issue of catastrophic forgetting.

## Option 3: Parameter Efficient Fine-tuning (PEFT)

Parameter Efficient Fine-tuning (PEFT) is a family of techniques aimed at fine-tuning a base model with a small number of trainable parameters. One popular method within PEFT is Low-Rank Adaptation (LoRA), which modifies a subset of layers' weights using a low-rank matrix decomposition technique. LoRA significantly reduces the number of trainable parameters while maintaining performance comparable to full parameter tuning.

Other popular and more efficinet method is quantized-lora (or qlora). QLoRA is a fine-tuning technique that combines a high-precision computing technique with a low-precision storage method. It is one of the most memory and computationally efficient fine-tuning methods ever created. QLoRA works by transmitting gradient signals through a fixed, 4-bit quantized pre-trained language model into Low Rank Adapters (LoRA). The weights of a pre-trained language model are quantized to 4-bits using the NormalFloat encoding. This produces a 4-bit base model that retains the full model architecture but with weights stored in a compact quantized format.

## Example Code 1: Fine-tuning an LLM using LoRA

We will demonstrate fine-tuning a language model using LoRA for sentiment classification. We'll use the Hugging Face ecosystem to fine-tune the `distilbert-base-uncased` model, a ~70M parameter model based on BERT, for classifying text as 'positive' or 'negative'. By employing transfer learning and LoRA, we can efficiently fine-tune the model to run on personal computers within a reasonable timeframe.
by using this PEFT method, we can reduce training parameters 
>> trainable params: 628,994 || all params: 67,584,004 || trainable%: 0.9306847223789819

## Example code 2: Fine-tuning an LLM using Qlora

we will demonstate fine-tuning a model using qlora for summarizing the conversation. we'll be using hugging face ecosystem to fine-tune the NousResearch/Llama-2-7b-hf model, a ~7B parameter model based of llama 2. for summarized the conversation between person. By employing transfer learning and QLoRA, we can efficiently fine-tune the model to run on personal computers within a reasonable timeframe.
by using this PEFT method, we can reduce training parameters 
>> trainable params: 16777216 || all params: 3517190144 || trainable%: 0.477006226934315

   
--->>>> For detailed code examples and resources, please refer to the GitHub repository and Hugging Face for the final model and dataset.
