from mir_eval.melody import evaluate


def getMetric(ref, est):
    ref_time, ref_freq = ref
    est_time, est_freq = est
    
    scores = evaluate(ref_time, ref_freq, est_time, est_freq)
    VR, VFA, RPA, RCA, OA = scores['Voicing Recall'], scores['Voicing False Alarm'], scores[
        'Raw Pitch Accuracy'], scores['Raw Chroma Accuracy'], scores['Overall Accuracy']
    return VR, VFA, RPA, RCA, OA
