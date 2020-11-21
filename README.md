## transformer

Simple transformer implementation from scratch in pytorch. 

## Classifier Transformer

<p align="left">
 <img  width="600" height="300" src="http://peterbloem.nl/files/transformers/classifier.svg">
</p>

## Generator Transformer

<p align="left">
 <img  width="600" height="300" src="http://peterbloem.nl/files/transformers/generator.svg">
</p>

## Limitations

<p align="left">
 <img  width="600" height="300" src="http://peterbloem.nl/files/transformers/encoder-decoder.svg">
</p>

The current models are designed to show the simplicity of transformer models and self-attention. As such 
they will not scale as far as the bigger transformers. For that you'll need a number of tricks that 
complicate the code (see the blog post for details).

All models so far are a single stack of transformer blocks (that is, no encoder/decoder structures). It 
turns out that this simple configuration often works best. 

source - http://peterbloem.nl/blog/transformers
