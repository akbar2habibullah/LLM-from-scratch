![meme](meme.jpg)

# LLM from scratch, no pre-trained models, no HF transformers

This is implementation of decoder-only transformer based LLM with next-token prediction objective. This implementation use `tokenizers` library from HF, [GQA](https://arxiv.org/abs/2305.13245) (Grouped query attention), [normalized-GPT](https://arxiv.org/abs/2410.01131), RoPE (Rotary positional embedding), and [Liger Kernel](https://github.com/linkedin/Liger-Kernel).

There are 6 versions:
- Using AdamW optimizer and lorem ipsum datasets (Broken RoPE) [[colab notebook]](https://colab.research.google.com/drive/1IfC8lQBi-PIuuLL0dFziCQ2CakIDVGiZ?usp=sharing)
- Using SOAP optimizer and lorem ipsum datasets (Broken RoPE) [[colab notebook]](https://colab.research.google.com/drive/15ZIynpMotd2z7pRGU3qLLfylacCGJpVI?usp=sharing)
- Using SOAP optimizer, synthetic number datasets, and larger parameter (Broken RoPE) [[colab notebook]](https://colab.research.google.com/drive/1BekXGDokeM7DwgggZptjQzcgkzIviXQ7?usp=sharing)
- Using SOAP optimizer, synthetic number datasets, smaller parameters, and larger epochs (Broken RoPE) [[colab notebook]](https://colab.research.google.com/drive/1EYlVeVwdTwG6E3yo1cxc6LSFat42L6yd?usp=sharing)
- Using SOAP optimizer, harder synthetic number [datasets](https://huggingface.co/datasets/ChavyvAkvar/synthetic-number), optimized hyperparameter, liger kernel applied, and Fast-FFN (fixed RoPE) [[colab notebook]](https://colab.research.google.com/drive/1CN7ERhIIVt0zlp2Y0tLFL4aaF2Ny5Fqy?usp=sharing)
- Using [tuned SOAP optimizer](https://github.com/ClashLuke/HeavyBall), harder synthetic number [datasets](https://huggingface.co/datasets/ChavyvAkvar/synthetic-number), optimized hyperparameter, liger kernel applied, Fast-FFN, and normalized-GPT (fixed RoPE) [[colab notebook]](https://colab.research.google.com/drive/1Bzsb_ptSo6N8HXb5BgCrxFmNjifNWGRg?usp=sharing)

We publish the weights from the latest version on HF [Link](https://huggingface.co/ChavyvAkvar/llm-numbers)

Notes: There's a small mistake in RoPE implementation where RoPE is applied to value_embedding (it should be applied only to query and key). The latest two versions fixes this issue.
