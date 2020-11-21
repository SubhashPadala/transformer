# transformer

Simple transformer implementation from scratch in pytorch. 
source http://peterbloem.nl/blog/transformers.

# Limitations

The current models are designed to show the simplicity of transformer models and self-attention. As such 
they will not scale as far as the bigger transformers. For that you'll need a number of tricks that 
complicate the code (see the blog post for details).

All models so far are a single stack of transformer blocks (that is, no encoder/decoder structures). It 
turns out that this simple configuration often works best. 
