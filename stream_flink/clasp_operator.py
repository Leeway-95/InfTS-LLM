from pyflink.datastream import ProcessFunction
from model_detector.metrics import binary_f1_score
from model_detector.repdetection import *


class ClaSSOperator(ProcessFunction):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state = None

    def open(self, runtime_context):
        # Initialize ClaSS instance
        self.clasp = RepDetection(
            n_timepoints=self.config['n_timepoints'],
            window_size=self.config['window_size'],
            k_neighbours=self.config['k_neighbours'],
            score=binary_f1_score,
            jump=self.config['jump'],
            p_value=self.config['p_value'],
            sample_size=self.config['sample_size'],
            similarity=self.config['similarity'],
            verbose=self.config['verbose']
        )

    def process_element(self, value, ctx):
        series_id, data_point = value
        profile = self.clasp.update(data_point)

        # Emit intermediate results
        yield (series_id, {
            'data_point': data_point,
            'profile': profile,
            'change_points': self.clasp.change_points,
            'representative_windows': self.clasp.representative_windows
        })