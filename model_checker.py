import stormpy.examples
import stormpy.examples.files
import os
import time
import argparse

def model_checking(path, formula_str):
    prism_program = stormpy.parse_prism_program(path)
    model = stormpy.build_model(prism_program)
    print("Number of states: {}".format(model.nr_states))
    print("Number of transitions: {}".format(model.nr_transitions))
    print("Labels: {}".format(model.labeling.get_labels()))
    properties = stormpy.parse_properties(formula_str, prism_program)
    model = stormpy.build_model(prism_program, properties)
    print("Labels in the model: {}".format(sorted(model.labeling.get_labels())))
    print("Number of states: {}".format(model.nr_states))
    print("Number of transitions: {}".format(model.nr_transitions))
    properties = stormpy.parse_properties(formula_str, prism_program)
    model = stormpy.build_model(prism_program, properties)
    result = stormpy.model_checking(model, properties[0])
    initial_state = model.initial_states[0]
    print(result.at(initial_state))
    return result.at(initial_state), model.nr_states, model.nr_transitions

def args_parser():
    parser = argparse.ArgumentParser(description="Model checking for PRISM models")
    parser.add_argument("--path", type=str, help="Path to the PRISM model file", default="prism_files/google_gemma-2b-it_metric_readability_10_4_cpu_The_nurse_is.prism")
    parser.add_argument("--formula", type=str, help="Formula to check", default="P=? [ F measure=3155 ]")
    return parser.parse_args()


if "__main__" == __name__:
    args = args_parser()
    start_time = time.time()
    result, number_of_states, number_of_transitions = model_checking(args.path, args.formula)
    end_time = time.time()
    folder_path = "logs"
    file_name = args.path.split(".")[0] + ".txt"

    with open(file_name, "a") as file:
        file.write("====================================\n")
        # Date and time
        file.write("Date: {}\n".format(time.strftime("%d/%m/%Y")))
        file.write("Time: {}\n".format(time.strftime("%H:%M:%S")))
        file.write("Path: {}\n".format(args.path))
        file.write("Formula: {}\n".format(args.formula))
        file.write("Number of states: {}\n".format(number_of_states))
        file.write("Number of transitions: {}\n".format(number_of_transitions))
        file.write("Result: {}\n".format(result))
        file.write("Model Parsing-Building-Checking Time: {}\n".format(end_time - start_time))

    