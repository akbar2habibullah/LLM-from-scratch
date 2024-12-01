# LLM from scratch, no pretrained models, no HF transformers

This is implementation of decoder-only transformer based LLM with next-token prediction objective. This implementation use `tokenizers` library from HF, attention use GQA (Grouped query attention), RMSnorm layer, GeGLU activation function, and RoPE (Rotary positional embedding).

There are four version:
- Using AdamW optimizer and lorem ipsum datasets
- Using SOAP optimizer and lorem ipsum datasets
- Using SOAP optimizer, synthetic number datasets, and larger parameter
- Using SOAP optimizer, synthetic number datasets, larger parameter, and larger epochs