# Codex Prompt: Low-Spec Grounded AI for EPC Smart Search

You are working inside the `EPC Smart Search` repository on Windows. Your job is to implement a local AI layer that works well on lower-spec customer laptops while preserving a strong retrieval-first contract search experience.

## Product Direction

This product must eventually become a strong local AI contract assistant, but it cannot depend on the model to discover the facts. The system must be grounded in a high-precision Lite retrieval engine. The AI layer is allowed to rewrite, compress, expand, or explain grounded evidence. It is not allowed to invent, infer, or choose unsupported facts.

The search experience should answer:

- `what does the contract require?`
- especially for:
  - party responsibilities
  - technical scope
  - risks and liabilities
  - definitions
  - system/equipment questions
  - technical follow-up questions

Most user questions will ask about a specific system, general piece of equipment, design condition, type/model, size, or a technical follow-up like how the system works.

Examples of failure that must be fixed:

- `What is the dew point configuration?`
  - Bad: random dew point information
  - Good: exact configuration for the current system, with location
- `What is the turbine we are using?`
  - Bad: random turbine information
  - Good: exact selected model/type, with location
- `What are our design conditions?`
  - Bad: generic blurbs containing the words `design` or `condition`
  - Good: exact design-condition clause/value set for the current system, with location

## User Constraints

- Highest priority: precision
- It must still be faster than a normal `Ctrl+F` workflow
- Success criteria:
  - top result contains the right clause about 90% of the time
  - common contract questions answer in under 10 seconds
- UI changes should be minimal
- Answer format should stay tight:
  - one short exact answer
  - then section/page/location
  - no broad explanation unless the user asks

## Hardware Target

Target low-spec customer hardware:

- Intel i7-10610U class CPU
- 16 GB RAM
- 4 GB graphics memory
- Windows laptop environment

Assume this machine is below the intended current Gemma target. The current full local AI path is too heavy for this hardware.

The AI system must be designed to run acceptably on hardware like this by reducing model size, context size, generation length, and responsibility. The model should be lightweight and low-thinking. Retrieval must carry the factual burden.

## Required Technical Direction

### 1. Retrieval-first architecture

Keep the current backend/storage/indexing foundation if it is still sound.

Use a hierarchical search system at the clause/subclause level:

1. parse the question into:
   - system/equipment
   - attribute being asked
   - intent
   - light follow-up context if cheap
2. search in tiers:
   - exact system + exact attribute
   - exact system in heading/clause
   - exact system aliases
   - broader semantic fallback only if needed
3. use a ranker + reranker design:
   - ranker for fast recall
   - reranker for exact question binding

The main failure to fix is partial keyword matching. The system must bind the whole question, not just one strong token.

### 2. Answer extraction before AI

Before any AI rewrite step, extract the exact answer from the top grounded clause.

Extraction should become attribute-aware:

- `configuration` -> arrangement/configuration sentence(s)
- `type` or `model` -> exact selected model/type sentence
- `design conditions` -> exact condition/value lines
- `size` -> exact size/rating/dimension line
- `responsibility` -> exact shall/must/provide sentence
- `function` -> exact functional description sentence block

If the top clause does not satisfy the full system + attribute match strongly enough, prefer refusal or extractive fallback over a blurry answer.

### 3. Low-spec local AI layer

Design the AI layer for low-resource inference:

- lightweight local model only
- short context window
- short outputs
- no deep thinking by default
- no large free-form synthesis
- no model-led fact selection

The AI layer should only do one of these:

- rewrite the grounded answer in plain English
- summarize a short grounded extract
- expand a grounded answer if the user explicitly asks

The AI layer must never replace the retrieval engine as the source of truth.

### 4. Path toward full AI system

Even though phase 1 is low-spec and retrieval-first, the architecture should leave a clean path to a stronger future grounded AI system:

- stronger reranker
- better multi-turn follow-up handling
- broader grounded answer synthesis
- upgraded local model path for better hardware

Do not build this future system by weakening the retrieval discipline.

## Implementation Priorities

Implement in this order:

1. query understanding
2. hierarchical ranker
3. reranker
4. attribute-specific answer extraction
5. confidence gating / refusal discipline
6. low-spec grounded AI rewrite layer
7. richer follow-up support later

## Non-Negotiable Guardrails

- Do not let the model decide facts without evidence
- Do not return generic keyword blurbs when the question is system-specific
- Do not broaden the answer beyond what was asked
- Do not rely on UI redesign to solve backend precision problems
- Do not optimize for “sounds smart” over “is exact and grounded”

## Expected Output Style

Default answer style:

1. short exact answer
2. section/page/location
3. no extra explanation unless asked

## Validation Requirements

Add or update tests that prove the system handles:

- exact system + configuration questions
- exact system + type/model questions
- exact system + design conditions questions
- exact responsibility questions
- exact definition questions
- exact function/how-it-works questions

Use the repo’s current tests as a base and extend them with regressions for full-question binding failures.

## Delivery Standard

When making changes:

- preserve what already works in the backend
- avoid large UI changes
- keep the app stable in Lite mode
- make small, testable checkpoints
- run tests after each meaningful retrieval/answer change

Your job is not just to add AI. Your job is to make the system reliably retrieve the exact clause first, then use a lightweight grounded AI layer that can run acceptably on low-spec laptops.
