# BERT-Base-Uncased Quantized Model for Sentiment Analysis for Stock Market Sentiment

This repository hosts a quantized version of the BERT model, fine-tuned for stock-market-analysis-sentiment-classification tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details

- **Model Architecture:** BERT Base Uncased  
- **Task:** Sentiment Analysis for Stock Market Sentiment 
- **Dataset:** Stanford Sentiment Treebank v2 (SST2)  
- **Quantization:** Float16  
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage

### Installation

```sh
pip install transformers torch
```


### Loading the Model

```python

from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load quantized model
quantized_model_path = "AventIQ-AI/sentiment-analysis-for-stock-market-sentiment"
quantized_model = BertForSequenceClassification.from_pretrained(quantized_model_path)
quantized_model.eval()  # Set to evaluation mode
quantized_model.half()  # Convert model to FP16

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define a test sentence
test_sentence = "Apple Inc. reported stronger-than-expected earnings this quarter, driven by robust iPhone sales and growth in its services segment. Investors reacted positively, pushing the stock up by 3% in after-hours trading. Analysts believe Apple is well-positioned for continued growth, especially with the upcoming product launches.On the other hand, Tesla shares dropped by 5% after the company missed its delivery targets and announced a temporary halt at its Berlin factory due to supply chain issues. Market sentiment remains cautious around Tesla, with concerns about rising competition in the EV space and fluctuating production numbers."

# Tokenize input
inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Ensure input tensors are in correct dtype
inputs["input_ids"] = inputs["input_ids"].long()  # Convert to long type
inputs["attention_mask"] = inputs["attention_mask"].long()  # Convert to long type

# Make prediction
with torch.no_grad():
    outputs = quantized_model(**inputs)

# Get predicted class
predicted_class = torch.argmax(outputs.logits, dim=1).item()
print(f"Predicted Class: {predicted_class}")


label_mapping = {0: "very_negative", 1: "nagative", 2: "neutral", 3: "Positive", 4: "very_positive"}  # Example

predicted_label = label_mapping[predicted_class]
print(f"Predicted Label: {predicted_label}")

```

## Performance Metrics

- **Accuracy:** 0.82 

## Fine-Tuning Details

### Dataset

The dataset is taken from Kaggle Stanford Sentiment Treebank v2 (SST2).

### Training

- Number of epochs: 3  
- Batch size: 8  
- Evaluation strategy: epoch  
- Learning rate: 2e-5  

### Quantization

Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safensors/     # Fine Tuned Model
├── README.md            # Model documentation
```

## Limitations

- The model may not generalize well to domains outside the fine-tuning dataset.  
- Quantization may result in minor accuracy degradation compared to full-precision models.  

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
