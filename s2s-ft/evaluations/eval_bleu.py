from measures import wer, moses_multi_bleu
import numpy as np


hyp_fn = "/home/shiquan/Projects/unilm_joint_learning/unilm/s2s-ft/tmp/finetuned_models/ckpt-13500.test"
ref_fn = "/home/shiquan/Projects/unilm_joint_learning/unilm/s2s-ft/tmp/finetuned_models/ckpt-13500.test.gold"


with open(hyp_fn, "r") as f_hyp, open(ref_fn, "r") as f_ref:
    hyp = [l for l in f_hyp]
    ref = [l for l in f_ref]

print("Hypothesis: " + hyp[0])
print("Reference: " + ref[0])

bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
print("BLEU SCORE: " + str(bleu_score))