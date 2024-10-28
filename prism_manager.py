import math

class PrismState:
    _id = 0

    def __init__(self, sequence:str, n_token:str, prob:float, measurements : dict, step:int=0):
        self.id = PrismState._id
        PrismState._id += 1
        self.sequence = sequence # The sequence of tokens
        self.n_word = n_token # The token that was added to the sequence
        self.prob = prob # The probability of the token being chosen given the sequence
        self.measurements = measurements # The measurements of the new sequence (sequence + n_token)
        self.step = step # The step in the generation process
        self.children = [] # The children of the state

    def get_prism_str(self):
        prism_str = ""
        prism_str += f"\t// {self.sequence} + {self.n_word}\n"
        # Guard
        prism_str += f"\t[] step={self.step} & id={self.id} "
        for measure in self.measurements:
            prism_str += f"& {measure}={self.measurements[measure]} "
        # Update
        prism_str += f"-> "
        if len(self.children) == 0:
            prism_str += f"1:(step'={self.step});\n"
        else:
            for child in self.children:
                prism_str += f"{child.prob}:(step'={child.step}) & (id'={child.id})"
                for measure in child.measurements:
                    prism_str += f" & ({measure}'={child.measurements[measure]})"
                prism_str += " + "
            prism_str = prism_str[:-3] + ";\n"

        return prism_str
    
    def get_measurement_string(self):
        measurement_str = ""
        for measure in self.measurements:
            measurement_str += f"{measure}={self.measurements[measure]} "
        return measurement_str
    
    def copy(self):
        n_state = PrismState(self.sequence, self.n_word, self.prob, self.measurements, self.step)
        n_state.id = self.id
        return n_state
    

class PrismStateAnalyzer:

    def __init__(self, prism_states):
        self.prism_states = prism_states
        self.max_steps = self.get_max_steps()
        self.measure_names = self.prism_states[0].measurements.keys()
        self.measure_boundaries = {}
        for measure in self.measure_names:
            self.max_measure = self.get_max_measure(measure)
            self.min_measure = self.get_min_measure(measure)
            self.measure_boundaries[measure] = (self.min_measure, self.max_measure)
        self.id_boundaries = (0, PrismState._id)
        self.step_boundaries = (0, self.max_steps)

    def get_max_steps(self):
        max_steps = 0
        for prism_state in self.prism_states:
            if prism_state.step > max_steps:
                max_steps = prism_state.step
        return max_steps
    
    def get_max_measure(self, measure):
        max_measure = -math.inf
        for prism_state in self.prism_states:
            if prism_state.measurements[measure] > max_measure:
                max_measure = prism_state.measurements[measure]
        return max_measure
    
    def get_min_measure(self, measure):
        min_measure = math.inf
        for prism_state in self.prism_states:
            if prism_state.measurements[measure] < min_measure:
                min_measure = prism_state.measurements[measure]
        return min_measure
    
    def get_top_prism_str(self):
        prism_str = "dtmc\n\n"
        prism_str += "module LLM\n"
        prism_str += f"\tstep : [{self.step_boundaries[0]}..{self.step_boundaries[1]}] init 0;\n"
        for measure in self.measure_names:
            prism_str += f"\t{measure} : [{self.measure_boundaries[measure][0]}..{self.measure_boundaries[measure][1]+1}] init {self.prism_states[0].measurements[measure]};\n"
        prism_str += f"\tid : [{self.id_boundaries[0]}..{self.id_boundaries[1]}] init 0;\n"
        return prism_str
    
    def get_state_transtion_str(self):
        state_transition_str = ""
        for prism_state in self.prism_states:
            state_transition_str += prism_state.get_prism_str()
        return state_transition_str
    
    def get_bottom_prism_str(self):
        return "endmodule\n\n"
    


if __name__ == "__main__":
    root = PrismState("Hello", "", 0.5, {"measure1": 0.5, "measure2": 0.5})
    print(root.get_prism_str())