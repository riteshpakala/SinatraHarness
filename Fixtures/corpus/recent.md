Retrieval feedback in SinatraHarness is implicit. When the user answers quickly, writes back at length and reuses words from a retrieved note, that note's reward rises; notes that are retrieved but ignored drift toward a negative weight.

The weights steer decoding directly: each retrieved partition contributes a sparse bias over its own vocabulary, added to the logits right before sampling, and the trace records how far each step moved.
