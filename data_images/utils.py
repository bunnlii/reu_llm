# paper variables
k_4 = 1.0
k_5 = 1.0
k_6 = 1000.0

class Request:
    def __init__(self, id, prompt_length, output_length, latency, accuracy, required_accuracy=None, prompt=None, reference=None, arrival_time=None, input_length=None,time_taken=None):
        self.id = id
        self.prompt_length = prompt_length
        self.output_length = output_length
        self.latency = latency
        self.accuracy = accuracy
        self.required_accuracy = required_accuracy
        self.prompt = prompt
        self.reference = reference
        self.arrival_time = arrival_time
        self.input_length = input_length if input_length is not None else prompt_length
        self.time_taken = time_taken  # Add this line if not already there

    def get_bandwidth(self):
        return self.get_bandwidth_from_output_length(self.output_length)
    
    @staticmethod
    def get_bandwidth_from_output_length(output_length):
        return output_length * k_4 + output_length * output_length * k_5
    
    def __repr__(self):
        return f"[id: {self.id}, out: {self.output_length}, latency: {self.latency}]"
    
    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        if isinstance(other, Request):
            return self.id == other.id
        return False

