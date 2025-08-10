# News Research Guidelines

EnkiBot supports deep research workflows for gathering and verifying news. Use the following rule of thumb when selecting a pipeline:

- **Best quality:** `o3-deep-research` → embeddings → web tools → verdict.
- **Balanced cost/latency:** `o4-mini-deep-research` → embeddings → web tools.
- **Lots of images:** use `GPT-4o` to extract information from images, then finalize with `o3-deep-research` or `o4-mini-deep-research`.

These sequences combine deep research models, embeddings for retrieval, and web tools to arrive at a final verdict.
