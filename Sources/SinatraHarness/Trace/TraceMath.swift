//
//  TraceMath.swift
//  SinatraHarness
//
//  WHAT: Per-step comparison of the un-injected (z) and injected (z') logits, as lazy MLX
//        ops: entropies, KL(p'‖p), Jensen–Shannon, the sampled token's log-prob and rank on
//        both sides, both argmaxes, and the probability mass moved into the impact mask.
//        `.full` adds top-k on both sides and the probability movement (p' − p) of every
//        mask token — large only where the token was plausible at that step, so it shows
//        where in the output Sinatra actually acted.
//

import Foundation
import MLX

enum TraceMath {

    struct Step {
        var entropyPre: MLXArray
        var entropyPost: MLXArray
        var kl: MLXArray
        var js: MLXArray
        var logprobPre: MLXArray
        var logprobPost: MLXArray
        var rankPre: MLXArray
        var rankPost: MLXArray
        var argmaxPre: MLXArray
        var argmaxPost: MLXArray
        var massIntoMask: MLXArray
        var sampled: MLXArray
        var counterfactual: MLXArray
        var topPreIds: MLXArray?
        var topPreValues: MLXArray?
        var topPostIds: MLXArray?
        var topPostValues: MLXArray?
        var movement: MLXArray?

        var arrays: [MLXArray] {
            [entropyPre, entropyPost, kl, js, logprobPre, logprobPost, rankPre, rankPost,
             argmaxPre, argmaxPost, massIntoMask, sampled, counterfactual]
                + [topPreIds, topPreValues, topPostIds, topPostValues, movement].compactMap { $0 }
        }
    }

    static func step(
        base: MLXArray, injected: MLXArray, sampled: MLXArray, counterfactual: MLXArray,
        mask: MLXArray?, topK: Int, full: Bool
    ) -> Step {
        let zb = base.asType(.float32)
        let zi = injected.asType(.float32)
        let lpB = zb - logSumExp(zb, axis: -1, keepDims: true)
        let lpI = zi - logSumExp(zi, axis: -1, keepDims: true)
        let pB = exp(lpB)
        let pI = exp(lpI)
        let zero = MLXArray(Float(0))

        let entropyPre = -sum(MLX.where(pB .> 0, pB * lpB, zero), axis: -1)
        let entropyPost = -sum(MLX.where(pI .> 0, pI * lpI, zero), axis: -1)
        let kl = sum(MLX.where(pI .> 0, pI * (lpI - lpB), zero), axis: -1)
        let logM = log((pB + pI) * 0.5 + 1e-30)
        let js = 0.5 * sum(MLX.where(pB .> 0, pB * (lpB - logM), zero), axis: -1)
            + 0.5 * sum(MLX.where(pI .> 0, pI * (lpI - logM), zero), axis: -1)

        let y = sampled.reshaped([1, 1]).asType(.int32)
        let logprobPre = takeAlong(lpB, y, axis: -1).reshaped([1])
        let logprobPost = takeAlong(lpI, y, axis: -1).reshaped([1])
        let rankPre = sum((zb .> takeAlong(zb, y, axis: -1)).asType(.float32), axis: -1)
        let rankPost = sum((zi .> takeAlong(zi, y, axis: -1)).asType(.float32), axis: -1)
        let argmaxPre = argMax(zb, axis: -1).asType(.int32)
        let argmaxPost = argMax(zi, axis: -1).asType(.int32)

        let massIntoMask: MLXArray
        if let mask {
            massIntoMask = sum(take(pI, mask, axis: -1), axis: -1) - sum(take(pB, mask, axis: -1), axis: -1)
        } else {
            massIntoMask = MLXArray.zeros([1])
        }

        var step = Step(
            entropyPre: entropyPre, entropyPost: entropyPost, kl: kl, js: js,
            logprobPre: logprobPre, logprobPost: logprobPost, rankPre: rankPre, rankPost: rankPost,
            argmaxPre: argmaxPre, argmaxPost: argmaxPost, massIntoMask: massIntoMask,
            sampled: sampled.reshaped([1]).asType(.int32),
            counterfactual: counterfactual.reshaped([1]).asType(.int32))

        if full {
            let k = max(1, min(topK, zb.dim(-1)))
            let preIds = argPartition(-lpB, kth: k - 1, axis: -1)[0..., ..<k]
            let postIds = argPartition(-lpI, kth: k - 1, axis: -1)[0..., ..<k]
            step.topPreIds = preIds.asType(.int32)
            step.topPreValues = takeAlong(lpB, preIds, axis: -1)
            step.topPostIds = postIds.asType(.int32)
            step.topPostValues = takeAlong(lpI, postIds, axis: -1)
            if let mask { step.movement = take(pI - pB, mask, axis: -1) }
        }
        return step
    }
}
