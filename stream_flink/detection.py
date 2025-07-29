from pyflink.datastream.functions import ProcessWindowFunction
from model_detector.metrics import binary_f1_score
from model_detector.repdetection import RepDetection
from window_size import suss

class Detection(ProcessWindowFunction):
    def __init__(self, n_timepoints=10_000, n_prerun=None, window_size=suss,
                 k_neighbours=3, score=binary_f1_score, jump=5, p_value=1e-50,
                 sample_size=1_000, similarity="pearson", stream_mode=True):
        self.n_timepoints = n_timepoints
        self.n_prerun = n_prerun
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score = score
        self.jump = jump
        self.p_value = p_value
        self.sample_size = sample_size
        self.similarity = similarity
        self.stream_mode = stream_mode
        self.stream = None
        self.last_cp = None

    def open(self, runtime_context):
        self.stream = RepDetection(
            n_timepoints=self.n_timepoints,
            n_prerun=self.n_prerun,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            score=self.score,
            jump=self.jump,
            p_value=self.p_value,
            sample_size=self.sample_size,
            similarity=self.similarity,
            stream_mode=self.stream_mode
        )

    def process(self, key, context, elements):
        for timepoint in elements:
            result = self.stream.update(timepoint)
            if self.stream_mode and result is not None:
                yield result

    def close(self):
        self.stream = None