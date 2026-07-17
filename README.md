# RAG Evaluation Harness: Where and Why Does RAG Actually Fail?

Instead of building another RAG app, I built a harness to actually test *where* and *why* RAG breaks: does it fail at retrieval (can't find the right info) or generation (has the right info, still gets it wrong)?

I built a 28-article Wikipedia corpus on ML concepts, hand-wrote a 40-question test set with deliberately varied difficulty, and ran it through three retrieval strategies (dense embeddings, BM25, and a hybrid of both) to see what actually breaks and why.

**TL;DR : the surprising result:** BM25 (plain keyword search) beat dense embeddings on questions phrased to avoid the source's exact wording. Combining both methods into a hybrid retriever made things *worse*, not better, on those same questions. Full breakdown below.

## Install

```
pip install sentence-transformers chromadb rank_bm25 groq pandas matplotlib
```

Dependencies:
- `sentence-transformers` - local embedding model, no API key needed
- `chromadb` - vector store for the chunked corpus
- `rank_bm25` - keyword-based retrieval, for comparison against embeddings
- `groq` - free-tier LLM API, used for answer generation and LLM-as-judge scoring
- `pandas` / `matplotlib` - eval results + charts

You'll also need a free [Groq API key](https://console.groq.com) for the generation/judging steps. Retrieval alone runs fully local, no key needed.

## Quick start

The whole pipeline lives in `rag_eval_harness.ipynb`. Open it in Colab or Jupyter and run top to bottom:

1. **Fetch + clean the corpus** - pulls 28 ML-concept Wikipedia articles, strips citation clutter, replaces LaTeX formulas with [FORMULA] placeholders
2. **Chunk + embed** - splits into ~1,100 chunks, embeds locally with all-MiniLM-L6-v2, stores in Chroma
3. **Build the eval set** - 40 hand-labeled questions across 7 difficulty categories
4. **Embeddings retrieval + generation + judging** - runs the eval set through dense retrieval, generates answers with Llama 3.3 70B (via Groq), then a second LLM call judges each one
5. **BM25 retrieval evaluation** - same eval set, keyword-based retrieval instead
6. **Hybrid retrieval evaluation** - combines both via Reciprocal Rank Fusion
7. **Compare all three + charts** - category-level comparison bars, saved to charts/
   
Retrieval steps run instantly (all local). The generation + judging step is the slow part — ~40 questions × 2 LLM calls each, a couple minutes on Groq's free tier.

If you just want the results without re-running anything, `data/eval_results_full.csv` has the full output already, and the charts are in `charts/`.

## Why I split it this way

Most RAG demos test the whole pipeline as one black box, so when the answer's wrong, you can't tell if retrieval grabbed the wrong chunks or generation just messed up a good chunk. I scored these two steps separately on purpose, so I could actually point to *which* part of the system is the weak link.

## Corpus

- 28 Wikipedia articles on core ML/AI concepts (gradient descent, backpropagation, transformers, RNNs/LSTMs, GANs, autoencoders, word embeddings, key researchers, etc.)
- Cleaned up during preprocessing: stripped out "See Also"/"References" sections (pure noise for Q&A), and swapped LaTeX math notation for `[FORMULA]` placeholders. I scoped math-heavy derivations out on purpose — this eval is about conceptual retrieval, not reproducing formulas, and that's a known weak spot for text-embedding RAG in general, not something I'm hiding
- Chunked into 1,134 pieces (~800 chars, 100-char overlap), embedded with `all-MiniLM-L6-v2`, stored in Chroma

## Evaluation set

40 questions I wrote by hand, each mapped to the exact chunk(s) needed to answer it, with the ground-truth answer written in my own words (not copy-pasted from the source, otherwise you're just testing phrase-matching, not correctness). I built these across 7 categories to deliberately stress different failure modes instead of just testing easy lookups:

| Category | What it tests |
|---|---|
| `direct` | Simple lookup - answer lives in one chunk |
| `multi_chunk_compare` | Answer requires synthesizing two separate facts (e.g. contrasting two concepts) |
| `multi_chunk_aggregation` | Answer requires collecting *every* instance of something scattered across many chunks (tests retrieval completeness, not just correctness) |
| `ambiguous` | The question deliberately avoids the source's exact terminology, testing whether retrieval can bridge a vocabulary gap |
| `ambiguous_aggregation` | Combines both of the above - indirect phrasing *and* multi-chunk collection |
| `leading_question` | The question implies a false premise (e.g. "why is X better than Y" when the source makes no such claim) — tests whether the system corrects the premise instead of complying with it |
| `unanswerable` | No answer exists in the corpus - tests whether the system honestly declines instead of hallucinating |

## How the evaluation works

1. **Check retrieval**: run each question through retrieval, check if the correct chunk(s) actually showed up (`top_k=5` normally, bumped to `top_k=20` for the `ambiguous`/`aggregation` categories to see if a wider net helps)
2. **Generate an answer**: feed the retrieved chunks to an LLM (Llama 3.3 70B via Groq), telling it explicitly to only use the given context and say "I don't know" if it's not enough
3. **Judge the answer**: a second LLM call grades the generated answer against my ground truth as `CORRECT` / `PARTIAL` / `WRONG`
4. **Check for hallucination**: every `WRONG` answer gets tagged as either `safe_refusal` (it said "I don't know") or `possible_hallucination` (it made something up), then I spot-checked these by hand

I ran this same eval set through three different retrieval setups: dense embeddings, BM25 (plain keyword search), and a hybrid combining both via Reciprocal Rank Fusion.

## What I found

### 1. Retrieval is the real bottleneck, not generation
Wherever retrieval found the right chunk, the LLM gave a good answer. Wherever retrieval missed, the answer failed too — even though the LLM itself was reliable whenever it actually had the right context. If I wanted to improve this system, tweaking the generation prompt wouldn't move the needle much. Fixing retrieval would.

### 2. Embeddings did *worse* than plain keyword search on indirectly-phrased questions
| Category | Embeddings | BM25 | Hybrid (RRF) |
|---|---|---|---|
| `direct` | 82% | 91% | 100% |
| `multi_chunk_compare` | 100% | 67% | 100% |
| `multi_chunk_aggregation` | 100% | 100% | 100% |
| `ambiguous` | **0%** | **50%** | 25% |
| `leading_question` | 50% | 50% | 50% |

*(% = at least one correct chunk retrieved)*

<img width="1365" height="686" alt="image" src="https://github.com/user-attachments/assets/3c30a4ef-3406-40dd-89c3-2c3fbae877ca" />


I expected embeddings to win on the `ambiguous` questions — that's the whole point of dense retrieval, understanding meaning instead of just matching words. Instead BM25 crushed it, 50% to 0%. Turns out my "ambiguous" questions still shared enough individual words with the source text (even while avoiding the exact term) for keyword matching to work fine. Meanwhile the embedding model kept grabbing a *neighboring* chunk instead of the actual right one, I checked this directly, and the correct chunk sat right next to ones that did get retrieved. The embedding for that chunk just blended into the general topic vibe of its neighbors instead of standing out for its one specific fact.

### 3. Hybrid retrieval isn't automatically better but can make things worse
Combining both methods (via RRF) helped on `direct` and the `multi_chunk` categories, where both methods already mostly agreed. But on `ambiguous` questions, hybrid actually did *worse* than BM25 alone (25% vs 50%). When the two methods disagree, embeddings' confidently-wrong picks can drown out BM25's correct one instead of the two complementing each other. 
Lesson: don't assume hybrid = better without checking, it depends on how much your two retrievers agree in the first place.

### 4. It fails safely : no real hallucinations
Every time it got a "WRONG" answer, it was almost always because it honestly said "I don't know based on the given context," not because it made something up. Zero clear hallucinations across all 40 questions. One case flagged by my automated checker as a "possible hallucination" turned out, when I actually read it, to be a totally correct answer — just to a *different* valid fact than the one I'd picked as ground truth. That was my question being underspecified, not the model lying. Also a good reminder that a keyword-based hallucination checker can't tell "wrong" from "right but not what I expected" , you still have to read the actual answers.

<img width="1800" height="750" alt="image" src="https://github.com/user-attachments/assets/b986b6ac-3e11-4886-b185-cf00ad4193ee" />

### 5. Retrieval budget (`top_k`) caps how much you can find
For "list all X" style questions, where the answer is scattered across 5-7 chunks, retrieval never grabbed all of them at `top_k=5`, simple math, you can't return more chunks than your budget allows. This is a real, common issue in production RAG: if your retrieval budget is smaller than how spread out the answer is, you'll always come up short no matter how good your retriever is.

## Limitations

- Corpus is Wikipedia-only: findings may not generalize to noisier or more technical real-world document sets
- Mathematical/formula-heavy content was deliberately excluded from scope (see Corpus section)
- The LLM-as-judge introduces its own noise; borderline CORRECT/PARTIAL/WRONG calls were not independently human-verified at scale
- Hallucination classification uses keyword matching, a rough heuristic - spot-checked manually but not exhaustively
- 40 questions is enough for category-level directional findings, not statistically rigorous conclusions

## Stack
Python · `sentence-transformers` (`all-MiniLM-L6-v2`) · ChromaDB · `rank_bm25` · Groq (Llama 3.3 70B) · pandas · matplotlib

## Repository structure
```
rag_eval_harness.ipynb   # full pipeline
data/
  eval_questions.csv      # 40-question labeled eval set
  eval_results_full.csv   # full results across all methods
charts/
  partial_recall.png.png
  full_recall.png
  answer_quality_breakdown.png
README.md
```

## Future Improvements
- Try a bigger/better embedding model to see if the `ambiguous`-category gap closes
- Human-verify a larger sample of the LLM-judge's CORRECT/PARTIAL/WRONG calls
- Test a smarter hybrid fusion than plain RRF (e.g. weight by retriever confidence instead of rank alone)
- Expand the corpus beyond Wikipedia to see if findings hold on messier real-world docs
