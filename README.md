# SinatraHarness

An on-device LLM harness for Apple silicon, built on [Frigate](https://github.com/rao-studios/Frigate)'s MLX stack, with an injection layer between the transformer and the decoder.

When a turn is answered with retrieved context, the prompt and the context go into the LLM as usual. Separately, **only the retrieved context** is encoded with the LLM's own embedding table and read by a small time-series side model. That model weighs each retrieved partition, and its weights become a bias added to the logits **right before sampling**.

What the side model learns from is not the user's reaction. After each turn, the model itself scores its output twice: once with the retrieved context and once without it. The difference shows, token by token, where the context moved the model, where the output followed and where it drifted away. It also gives each retrieved partition's share of the answer, its citation value. Those measurements label the turn, and the steer learns to push harder toward what the context meant where the output kept drifting.

```
prompt + context ─▶ chat template ─▶ transformer ─▶ logits ────────────┐
retrieved context only ─▶ embedding table ─▶ SinatraNet ─▶ bias ─▶  + ─┴▶ sampler ─▶ tokens
tokens re-scored with and without the context (grounding) ─▶ labels ─▶ training ─┘
```

It is the on-device successor to Sewn's server-side Sinatra. The stock-indicator process and IMBHS (Improved Music-Based Harmony Search) are carried over; they now run over how grounded the owner's outputs have been. Sewn's reply-judging loop is gone: there is no LLM judge, and nothing waits for the user's next message.

## The loop

Each generation with retrieved context is one **turn**:

1. **Enforce the 30-day band.** Anything older is dropped.
2. **Encode the retrieved partitions**: token ids, the LLM's input-embedding rows, mean pooling, and a 64-d count sketch.
3. **Weigh each partition.** Priors from what earlier measurements taught, and SinatraNet's prediction, are blended by a reliability gate.
4. **Build the injection**, a sparse bias over each partition's content tokens, scaled once by its relevancy band.
5. **Decode.** The injection runs before the repetition penalties, and a trace can record its effect step by step.
6. **Measure the grounding.** Still under the gate, after the reply has streamed, the output is scored with and without its context.
7. **Label and record.** Each partition gets a steer target from the measurement. Training runs when due, and yields to any generation that is waiting.

### The grounding measurement

The decode leaves its KV cache behind. Trimmed back to the prompt, it re-scores the output exactly, with no prefill: those are the distributions the decode saw, before injection and penalties. A copy trimmed to the prefix the two prompts share starts the bare side, which only prefills the part of the bare prompt that differs. Both sides then score the same output in chunks of 64 positions. On Mistral Nemo, 100 output tokens take about 0.6 s.

Per step t, with y_t the token that was said:

| Quantity | Definition | Reads as |
|---|---|---|
| ι_t, influence | `log p_ctx(y_t) − log p_bare(y_t)` | nats the context added to what was said |
| KL_t | `KL(p_ctx ‖ p_bare) = E_{v∼p_ctx}[log p_ctx(v) − log p_bare(v)]` | how much the context moved this step, on average |
| drift_t | `max(0, KL_t − ι_t)` | how much less the output took than the context gave |
| tune_t | `argmax_v p_ctx(v) · (log p_ctx(v) − log p_bare(v))` over both sides' top-64 | the token the context moved the model toward |
| class | grounded if ι ≥ 0.5, contradicted if ι ≤ −0.5, else unsupported; `function` for non-content tokens | |
| risk_t | `1 − p_ctx(y_t)` on unsupported tokens, else 0 | the model's own doubt about a token the context did not back |

Drift is measured against the context's average lift, not against the single most-lifted token. Otherwise a rare token the context raised from near zero would dominate it.

Per turn:

* **grounding**: the share of content tokens the context pushed for.
* **drift and driftShare**: mean drift, and the share of content tokens drifting more than 1 nat.
* **contextDependence**: Σι, how much likelier the whole output is with its context.
* **hallucinationRisk**: mean risk over content tokens. **parrotShare**: the share of the output copied in runs of 6 or more tokens.
* **Attribution per partition.** `A_p = Σ ι⁺ · share_p(y_t)` is its citation value in nats, where `share_p` splits a token among the partitions holding it by term frequency, matching tokens by their trimmed, lowercased text. It also carries uptake (its share of the influence), intent (its share of the push the output did not take), coverage and parrot.

On the fixture notes, Nemo's summary measures 76% grounded with 10% of content tokens drifting. The drifting steps are the places it paraphrased the fresh note: " responses" where the note says "replies", " influence" where it says "steer". Steering toward that note cut drift from 0.23 to 0.08 nats per content token. Exact leave-one-out attribution (`attribute`) ranks the notes the same way the lexical attribution does.

### Labels and the steer target

A measured turn labels every partition at once:

```
y_p = clip(w_p + κ · driftShare · r_p · (intent_p − uptake_p) − λ · parrot_p · max(0, w_p), −1, 1)
```

* **A grounded turn keeps its steer.** With no drift, the target is the weight that was applied.
* **A drifting turn leans toward what the output missed.** A partition gains when the context pushed toward it and the output under-used it, and loses when the output leaned on it more than the context's push warranted.
* **Recency scales the correction.** `r_p` is the partition's relevancy over the turn's most relevant one. Relevancy is its band weight times `0.5 + 0.5 × score01`, and the band comes from the newest of its creation, modification and index times, with first-seen as the fallback.
* **Copying damps only the steer's own push.** A model quoting on its own is left alone.

κ (`steerGain`) and λ (`parrotPenalty`) are 1 by default. The series the indicators run over is each measured turn's grounding, with its content-token count as volume.

### Features (38 per partition)

The model sees 26 static features captured when the turn is prepared, 11 indicators rebuilt as of the turn, and a context-validity flag. Alongside them it gets the 64-d context sketch.

* **Recency.** Document age from its creation time, modification age, and a known flag for each. First-seen age, never-seen-before, recency of the last retrieval, and frequency over 30 days and over 7 days.
* **What earlier measurements taught.** The partition's mean steer target and its confidence, the document's mean steer target, the partition's mean uptake, and the document's mean drift.
* **Within the turn.** Rank, min–max score (distance flipped), partition size and relevancy.
* **The last measured turn.** How hard it steered, its grounding and its drift.
* **When.** Time-of-day and day-of-week as periodics, and the gap since the previous turn.
* **Indicators.** Over the owner's grounding series: EMA, SMA, MACD, MACD signal and previous signal, average interval change, volume-weighted average, stochastic %K and %D, momentum and velocity. IMBHS tunes the 11 windows every 5 training cycles once 20 have run.

The band weight itself is not a model input. It multiplies each partition's injection once, in the builder (fresh 1.0, 7–30 days 0.7, older 0.4), and decays sample weights. As a step-function input it made every document crossing day 7 an unseen region the net extrapolated into.

### SinatraNet and the gate

The net is `ctx 64→8 ⧺ features 38 → 32 → 16`. It has a zero-initialised tanh steer head (w_p ∈ [−1, 1]) and a sigmoid uptake head, about 2k parameters, trained with AdamW on the CPU device against the steer targets and the measured uptakes. Training is warm-started each cycle, uses input dropout, and stops early on a chronological validation split that keeps the best weights.

The gate g is the net's skill over the mean predictor on the most recent 20% of turns, ramped from 20 to 100 labelled events. The applied weight is `w_p = (1 − g) × w_prior + g × w_net`, where the prior is the partition's mean steer target (else its document's) times its confidence. A net that doesn't beat the baseline gets g = 0, and the priors carry the turn.

### The injection

* **`lexical`** (the default). For partition p, `m_p[t] = sqrt(tf) × idf`, max-normalised. The bias is `α Σ_p band_p × w_p × m_p[t]`, clamped to ±2 nats. Special and control tokens, punctuation and stopwords are excluded, and so are the owner's own "hot" tokens, learned with Space-Saving.
* **`dense`** (experimental). Each partition's pooled embedding goes through the model's own `lm_head`. The results are combined by weight, z-scored, and the top 4096 content tokens are kept.
* **`off`**. No injection. Turns are still measured and labelled.

The prompt is never inspected, since `prompt(_:)` is a no-op. The cost per token is one `[1, V]` add.

### Tracing: two layers

A turn's trace (`owners/<key>/traces/<turnId>.json`) has two layers.

* **The injection layer** records what Sinatra did to the logits. The injection is additive and constant within a turn, so its effect on every step is attributable exactly. A wrapper processor keeps both the un-injected logits z and the injected z', with penalties applied to each. A wrapper sampler draws the real token from z' and a **counterfactual** from z with an identically seeded shadow sampler. Per step it records both entropies, KL and Jensen–Shannon divergence, the sampled token's log-prob and rank on both sides, both argmaxes, and the probability mass moved into the **impact mask**, the tokens Sinatra biased. With `full` it adds top-k before and after and Δp for every mask token. Nothing syncs per step.
* **The grounding layer** records what the retrieved context did to the output: the per-step rows above, the turn's summary and the attribution. With `full` it adds each step's bare-side rank of the token that was said and the tokens the context moved the model toward most. It is written for every measured turn, traced or not.

`compare` decodes the same prompt with the injection off and on, using the same seed. It scores each output under both distributions and measures both outputs' grounding, so it shows whether the steer made the output follow its context. `analyze` sets the injection against the grounding over the owner's band. It gives r(bias, drift), steered against unsteered turns, the first half of the band against the second, and a citation leaderboard per document.

## Using it

```swift
import SinatraHarness

let harness = Harness(storeDirectory: storeURL)
try await harness.load(modelID: "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit")

let handle = try await harness.generate(GenerateRequest(
    input: UserInput(chat: [.system(systemWithContext), .user(userText)]),
    parameters: GenerateParameters(maxTokens: 400, temperature: 0.4, topP: 0.9),
    turn: TurnInput(
        owner: OwnerID(userId),
        retrieved: partitions),  // [Partition]: id, documentId, text, score, createdAt?, modifiedAt?
    bareInput: UserInput(chat: [.system(systemWithoutContext), .user(userText)])))
for await event in handle.stream { /* .chunk / .toolCall / .info, as MLXLMCommon.generate */ }
let completion = await handle.completion.value  // grounding, trace, owner summary, training scheduled?
```

A `GenerateRequest` without a `turn` is a plain generation: no injection, and nothing recorded. A turn without a `bareInput` is recorded but never measured, so it teaches nothing. The harness keeps one model resident and runs one generation at a time. Its gate also serialises encoding, trace evaluation, the grounding measurement, training and model swaps.

The store directory holds `owners/<key>/ledger.json`, `owners/<key>/model-<modelKey hash>.safetensors`, and the last 20 traces in `owners/<key>/traces/<turnId>.json`.

Sewn's `local` provider runs through this harness. [Integrating with an inference server](#integrating-with-an-inference-server) walks through how, and how to do the same in another server.

## Integrating with an inference server

SinatraHarness is built to be the on-device backend of a Swift server that already serves chat from MLX models. Sewn is the reference integration. The same pattern fits any Swift-on-Server stack on Apple silicon, such as Hummingbird, Vapor or a gRPC service, because nothing in SinatraHarness depends on a web framework.

The server keeps its routes, retrieval, prompt assembly, citations and streaming transport. SinatraHarness takes over three things:

* **Model residency.** One model is loaded, and a gate runs one generation at a time.
* **The decode.** The injection runs before sampling, with an optional trace.
* **The per-user grounding loop.** Each turn is measured against its own context after the decode, labelled from that measurement, and trained on.

### 1. Add the dependency

```swift
// Package.swift
dependencies: [
    // Frigate is the only MLX in the graph. Point it at the same place SinatraHarness does.
    .package(path: "../Frigate"),       // or its Git URL
    .package(path: "../SinatraHarness"),    // or https://github.com/riteshpakala/SinatraHarness.git
],
targets: [
    .executableTarget(
        name: "my-server",
        dependencies: [
            // Apple silicon only: condition the products so other platforms still build.
            .product(name: "SinatraHarness", package: "SinatraHarness", condition: .when(platforms: [.macOS])),
            // UserInput, GenerateParameters and Generation come from here.
            .product(name: "MLXLMCommon", package: "Frigate", condition: .when(platforms: [.macOS])),
        ]),
]
```

* **Declare Frigate yourself.** The API takes MLXLMCommon's `UserInput`, `GenerateParameters` and `Generation`, and SwiftPM only lets a target name products from packages its root declares.
* **Keep one Frigate in the graph.** Every package must point Frigate at the same location: the same path, or the same URL. Never add `ml-explore/mlx-swift`, because Frigate's targets have the same names.
* **Guard the call sites** with `#if canImport(SinatraHarness)`. Give other platforms a stub that refuses honestly; Sewn answers 503.
* **Ship the Metal library.** `swift build` never compiles MLX's shaders. After each build, run Frigate's `scripts/build-metallib.sh release --package <your package>`, and deploy `mlx.metallib` beside the installed binary (pass `--app` for an app bundle). Without it, the first GPU operation fails.

### 2. Keep one harness per process

```swift
import SinatraHarness

let harness = Harness(
    storeDirectory: dataRoot.appendingPathComponent("sinatra-harness", isDirectory: true),
    configuration: SinatraConfiguration())   // lexical injection, 30-day band, automatic trace

// At startup, so the first turn doesn't pay for loading:
try await harness.load(modelID: "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit")

// On shutdown:
await harness.flush()
```

* **`Harness` is an actor.** It owns the resident model, a first-in-first-out gate, and every owner's store. Where your module has its own `Harness`, qualify it as `SinatraHarness.Harness`. Loading a different model evicts the current one under the gate, never mid-generation.
* **Report its state.** `harness.state` is `cold`, `loading(fraction)`, `ready(model)` or `failed(reason)`. That is what a health or providers endpoint should report.
* **Keep all MLX work on this model inside the harness.** Its gate also covers context encoding, trace evaluation, training and model swaps. Work that bypasses it can overlap a decode.
* **Treat the store as user data.** It sits under your data root and holds per-user files; see the checklist.
* **Models resolve through Frigate's `HubDownloader`.** It reuses a complete snapshot already on disk, in `$HF_HOME/snapshots`, `$HF_HOME` or `~/Documents/huggingface`, before downloading anything.

### 3. Map each chat turn

A chat turn with retrieved context becomes one `GenerateRequest` with a `TurnInput` and the same chat without its context:

```swift
let handle = try await harness.generate(GenerateRequest(
    input: UserInput(chat: [.system(systemPromptWithContext)] + history + [.user(userText)]),
    parameters: GenerateParameters(
        maxTokens: request.maxTokens, temperature: request.temperature, topP: request.topP,
        repetitionPenalty: request.repetitionPenalty == 1 ? nil : request.repetitionPenalty,
        seed: request.seed),
    turn: TurnInput(
        owner: OwnerID(authenticatedUserID.lowercased()),
        retrieved: hits.map {
            Partition(id: $0.chunkID, documentId: $0.documentID, text: $0.text,
                      score: $0.distance, createdAt: $0.createdAt, modifiedAt: $0.modifiedAt)
        },
        conversationId: conversationID),
    bareInput: UserInput(chat: [.system(systemPromptWithoutContext)] + history + [.user(userText)])))
```

| Field | What to pass | Why it matters |
|---|---|---|
| `input` | Your prompt exactly as you would send it to MLX, retrieved context included | SinatraHarness never reads or changes the prompt |
| `retrieved` | The chunks your retriever returned for this turn | The only text the side model encodes |
| `Partition.id` | A stable id per chunk | The user's history (priors, frequency, first-seen) accrues per id |
| `Partition.score` | The retriever's distance, lower = closer | For a similarity score, set `scoreIsDistance = false` in the configuration |
| `Partition.createdAt`, `modifiedAt` | The document's creation and last-modified times, when known | The newest of them places it in the 30-day band; without either, the first time this owner retrieved it does |
| `owner` | A stable, normalised user id | Ledgers, weight models and traces are kept per owner |
| `bareInput` | The same chat with the retrieved context left out, and nothing else changed | The grounding measurement scores the output against it. Keep everything before the context identical, so the bare side reuses the prompt's KV cache for that prefix |
| `conversationId` | Optional thread id | Recorded with the turn |

* **`turn: nil`** makes a plain generation: no injection, and nothing recorded or learned. Use it for every generation that isn't a user-facing chat turn, such as summaries, compaction, titles and tool planning, and never pass a turn for hosted providers.
* **Per-request options** map onto `GenerateRequest`: `mode` (`off`, `lexical`, `dense`), `trace` (`automatic`, `off`, `summary`, `full`), and `record: false` for a dry run that plans and traces without recording the turn.
* **History** must suit the model's chat template. Mistral's rejects anything but alternating user and assistant turns after the system message, so normalise history before building `UserInput`. Sewn's `LocalMessageMapper` does this.

### 4. Stream, then report

```swift
for await event in handle.stream {
    switch event {
    case .chunk(let text):    /* write `text` to your SSE or websocket */ break
    case .toolCall(let call): /* your tool-call path */ break
    case .info:               break
    }
}
let completion = await handle.completion.value
// handle.plan?.diagnostics, completion.trace?.summary, completion.summary → one trailing metadata event
```

* **The stream carries MLXLMCommon's own `Generation` events,** so a writer built for `MLXLMCommon.generate` keeps working.
* **`completion` resolves once the turn is measured and recorded.** It holds the text, the grounding measurement, the trace (when one was requested, an injection ran, or the turn was measured), the owner's summary, and whether training was scheduled. Sewn sends this as a trailing `sinatra` object before `[DONE]`.
* **The measurement costs time after the reply,** not before it: about 0.6 s per 100 tokens on Nemo, capped by `groundingBudget` (6 s), and abandoned between chunks when another generation is waiting. `groundingMeasurement` set to `.traced` limits it to traced turns, and `.off` turns it off.
* **On a client disconnect, cancel the task that consumes the stream.** The harness stops the decode and drops the unfinished turn instead of labelling a partial answer.
* **Training runs by itself** after a turn once enough new labels have arrived. It holds the gate for at most `trainingBudget` (0.4 s by default) and stops early when a generation is waiting.

### 5. Endpoints worth exposing

| Endpoint | Harness call | Sewn's route |
|---|---|---|
| Per-user status | `summary(owner:)` | the local row of `GET /v1/providers` |
| Warm a model | `load(modelID:)` | `POST /v1/providers/local/warm`, optional `{"model": …}` |
| One trace | `trace(owner:turnId:)` | `GET /v1/providers/local/sinatra/traces/{id}` |
| Grounding over time | `groundingReport(owner:)` | `GET /v1/providers/local/sinatra/analysis` |
| Account deletion | `forget(owner:)` | the admin owner-delete route |

Scope the trace and analysis routes to the authenticated owner: traces contain decoded tokens from that user's context.

### How Sewn does it

Sewn's retrieval (Thread fan-out), prompt assembly, citations and billing are unchanged. The only change on the hot path is that on-device turns carry the retrieved partitions, the turn context and the bare system prompt into the provider.

| Piece | Where in Sewn |
|---|---|
| Dependency, product conditioned to macOS | `Package.swift` |
| The adapter actor: loading, state, request mapping, diagnostics | `Sources/Providers/Local/LocalInference.swift` |
| Turn context, sampling, the `sinatra` wire types | `Sources/Providers/Local/LocalTurn.swift` |
| Retrieved partitions with their scores, from Thread search | `Sources/Core/Models/Sewn.RetrievedPartition.swift` and `Sources/Core/Commands/Sewn+Search.swift` |
| The bare system prompt: the same persona and instructions, no context | `Sources/Core/Sewn.swift` (`handleChat`) |
| Handing them over only when `provider` is `local` | `Sources/API/Routes/Handles/handleChatStreamCompletions.swift`, the non-streaming handler, and the realtime grounded pass |
| Status, warm, traces, analysis | `Sources/API/Routes/Providers.swift` |
| Purge on owner delete | `Sources/API/Routes/Admin.swift` |

Environment overrides (`SEWN_SINATRA_MODE`, `SEWN_SINATRA_TRACE`, `SEWN_SINATRA_ALPHA`) set the harness configuration. `sewn-probe` exercises the whole path over HTTP.

### Checklist

* One Frigate in the graph, pointed at the same place by every package.
* `mlx.metallib` beside every binary that runs MLX. That includes test bundles: install it after the build, because the bundle is code-signed and a later copy breaks the seal (see `scripts/test-mlx.sh`).
* Stable owner ids.
* Retrieved chunks on `retrieved`, never the prompt.
* A `bareInput` on every user-facing turn: the same chat, context left out.
* `turn: nil` for utility generations and for any provider other than the on-device one.
* The store is per-user data: scope reads to the owner, call `forget(owner:)` when an account is deleted, and `flush()` on shutdown.
* One decode at a time per model: concurrent requests queue on the gate.
* Injected and measured decodes build their KV cache without quantisation. That is the iterator path MLXLMCommon offers for custom processors. Plain decodes keep it.

### Lower-level use

A server that already owns model residency and its own serialisation can use the parts directly:

* `SinatraSession` for `prepareTurn`, `groundingMeasurement`, `generationDidFinish` and `train`, built with `LanguageModelContextEncoder`, `MLXLMTokenizer` and `MLXWeightModel`
* `InjectedGeneration.start(input:parameters:context:plan:trace:measure:)` for the decode
* `GroundingScorer.score(model:decodeCache:contextTokens:bareTokens:sampled:options:)` for the measurement

It must then serialise every MLX call itself: the encoding inside `prepareTurn`, the decode, draining the trace, the measurement, and training. `drainTrace` gives it step traces and their summary, but only the harness assembles the stored trace files. The harness is the supported path.

## CLI

```bash
swift build
./scripts/metallib.sh debug     # MLX's Metal shaders; swift build never compiles them

# With and without the injection, traced step by step, and each output's grounding
# (--trace full adds the heatmap and, at every drifting step, what the context wanted instead)
.build/debug/sinatra-harness generate --model mlx-community/Mistral-Nemo-Instruct-2407-4bit \
  --prompt "Summarise what these notes say about retrieval feedback." \
  --context Fixtures/corpus/recent.md:2026-09-20 --context Fixtures/corpus/old.md:2026-01-05 \
  --temperature 0 --trace full

# Same seed, off vs on: cross-likelihoods, and grounding and drift for both outputs
.build/debug/sinatra-harness compare --model … --prompt … --context … --force-weights "1.0,-0.5"

# Exact citation values: the output re-scored without each partition in turn
.build/debug/sinatra-harness attribute --model … --prompt … --context … --context …

# The loop on a synthetic owner whose context keeps pointing at a note the stand-in model leaves out
.build/debug/sinatra-harness replay --synthesize 60 --train --mlx-weights --store /tmp/sinatra
.build/debug/sinatra-harness inspect --store /tmp/sinatra
.build/debug/sinatra-harness analyze --store /tmp/sinatra     # r(bias, drift), bins, citations
.build/debug/sinatra-harness trace --store /tmp/sinatra       # the latest stored trace, both layers
.build/debug/sinatra-harness features --context notes.md       # the 38 features, dry run
.build/debug/sinatra-harness encode --model … --context a.md --context b.md
```

`--force-weights` sets the partition weights by hand, so the injection shows on an owner with no history. Without a model, `replay` stands `LexicalGroundingEstimate` in for the model's measurement.

## Tests

```bash
swift test                 # pure-Swift suites: indicators, IMBHS, grounding signals, bands, filter, bias, session loop
./scripts/test-mlx.sh      # everything, with the MLX suites (injection, trace math, weight model, encoder,
                           # and the grounding scorer: cache reuse is exact, no context moves nothing)
```

## Packaging

Frigate is the only MLX in the graph: its targets are literally named `MLX`, `MLXNN`, and so on. Never add `ml-explore/mlx-swift` here, because the names collide. `Package.swift` takes Frigate from the sibling checkout (`../../rao/repositories/Frigate`), which is the same directory Sewn uses. The committed URL form is in the comment beside it.

The package was called SinatraMLX until 2026-09-24, with the actor named `SinatraHarness`. GitHub redirects the old repository URL; the module, the actor (now `Harness`) and the CLI (now `sinatra-harness`) changed with the name.
