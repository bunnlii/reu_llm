# paper variables
k_4 = 1.0
k_5 = 1.0
k_6 = 1000.0

class Request:
    def __init__(self, id, prompt_length, output_length, latency, accuracy):
        self.id = id
        self.prompt_length = prompt_length
        self.output_length = output_length
        self.latency = latency
        self.accuracy = accuracy
    
    def get_bandwidth(self):
        return self.get_bandwidth_from_output_length(self.output_length)
    
    @staticmethod
    def get_bandwidth_from_output_length(output_length):
        return output_length * k_4 + output_length * output_length * k_5
    
    def __repr__(self):
        return f"[id: {self.id}, out: {self.output_length}, latency: {self.latency}]"
    
    def __hash__(self):
        return self.id
