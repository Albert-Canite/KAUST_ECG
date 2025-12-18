# Four-class collapse investigation

## What was wrong
- `compute_class_weights` normalized inverse-frequency weights by their own mean, then clamped around that mean. With the actual class counts (e.g., N≈48k, S≈2k, V≈2k, O≈7k), this normalization **cut the "O" weight below 1.0** and only mildly boosted the rare "S"/"V" classes. Cross-entropy therefore penalized mistakes on class 3 less than on the majority class and was far weaker than the sampler heuristic, so the model learned to behave almost like a binary N/V classifier and never produced "S"/"O" predictions (as seen in the provided confusion matrices).
- The sampler used a different balancing rule (`num_samples / (num_classes * count)`) so CE weights and the sampler pulled in opposite directions, further destabilizing class learning.

## Fix
- Reworked `compute_class_weights` to match the sampler’s uniform-prior assumption: weights now scale with `ideal_count / class_count` (optionally exponentiated) before clamping. This consistently upweights all minority classes (including "O") and aligns CE loss balancing with the sampler.
