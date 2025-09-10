class MetricBase:
    '''
    Base class for all metrics
    '''
    name : str = 'metric_base'

    def __init__(self):
        pass

    def evaluate(self,anything):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass