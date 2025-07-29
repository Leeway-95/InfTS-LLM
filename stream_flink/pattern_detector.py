from pyflink.datastream import ProcessFunction
from time_series import detect_patterns, classify_segment


class PatternDetector(ProcessFunction):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def process_element(self, value, ctx):
        series_id, data = value
        representative_windows = data['representative_windows']
        time_series = data['time_series']

        window_classifications = []
        for start, end in representative_windows:
            segment = time_series[start:end]
            classification = classify_segment(segment, self.config['period'])
            window_classifications.append(classification)

        yield (series_id, {
            **data,
            'window_classifications': window_classifications
        })