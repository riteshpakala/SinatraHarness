//
//  SinatraNet.swift
//  SinatraHarness
//
//  WHAT: The time-series side model. Per retrieved partition it reads the 38 time,
//        grounding and indicator features plus a 64-d sketch of the partition's pooled
//        embedding, and emits a steer w_p ∈ [−1, 1] (how hard to push decoding toward that
//        partition) and an uptake forecast û_p ∈ [0, 1] (its share of the context's
//        influence on the output).
//
//          ctx_proj: 64 → 8 ──┐
//          features (38) ─────┴→ input: 46 → 32 → GELU → hidden: 32 → 16 → GELU
//                                   ├→ steer_head:  16 → 1 → tanh      (zero-initialised: w = 0 at birth)
//                                   └→ uptake_head: 16 → 1 → sigmoid
//
//  PIN:  ~2k parameters, ~10 KB on disk: sized for the few hundred rows a 30-day band
//        holds. v2 note: replace `hidden` with attention across the turn's partitions so
//        they compete for the injection.
//

import Foundation
import MLX
import MLXNN

final class SinatraNet: Module {
    @ModuleInfo(key: "ctx_proj") var ctxProj: Linear
    @ModuleInfo(key: "input") var input: Linear
    @ModuleInfo(key: "hidden") var hidden: Linear
    @ModuleInfo(key: "steer_head") var steerHead: Linear
    @ModuleInfo(key: "uptake_head") var uptakeHead: Linear
    /// Input dropout while training: no single drifting feature can carry the prediction.
    let featureDropout = Dropout(p: 0.2)
    let contextDropout = Dropout(p: 0.1)

    init(featureCount: Int, contextDim: Int) {
        self._ctxProj.wrappedValue = Linear(contextDim, 8)
        self._input.wrappedValue = Linear(featureCount + 8, 32)
        self._hidden.wrappedValue = Linear(32, 16)
        self._steerHead.wrappedValue = Linear(weight: MLXArray.zeros([1, 16]), bias: MLXArray.zeros([1]))
        self._uptakeHead.wrappedValue = Linear(16, 1)
        super.init()
    }

    func callAsFunction(features: MLXArray, context: MLXArray) -> (steer: MLXArray, uptake: MLXArray) {
        let c = ctxProj(contextDropout(context))
        let x = concatenated([featureDropout(features), c], axis: -1)
        let h = gelu(hidden(gelu(input(x))))
        let w = tanh(steerHead(h)).squeezed(axis: -1)
        let u = sigmoid(uptakeHead(h)).squeezed(axis: -1)
        return (w, u)
    }
}
