import csv
import os
import random
import uuid

import numpy
import numpy as np
import pygad

import cobot_ml.data.datasets as dss
from cobot_ml.data.datasets import DatasetInputData
from cobot_ml.data.utilities import DsMode
from cobot_ml.models import LSTM
from cobot_ml.utilities import dumps_file
from generate_test import run_prediction
from training_v_cobot import process


class FitnessFunc:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def fitness_func(self, solution, idx):
        model = LSTM(features_count=len(solution), n_layers=2, forecast_length=10)

        os.chdir(os.path.join(os.path.dirname(__file__), "", ".."))
        dataset: dss.CoBot202210Data = DatasetInputData.create(dss.Datasets.CoBot202210, weights=solution)
        dataset.minmax()
        dataset.apply_weights(solution)

        folder = os.path.join(self.base_dir, f"{idx}_{str(uuid.uuid4())}")

        subfolder = process(
            input_length=5,
            forecast_length=10,
            model=model,
            dataset_name=dss.Datasets.CoBot202210,
            dataset=dataset,
            channel_name="train",
            ds_mode=DsMode.WITH_MPC,
            base=folder
        )
        dumps_file(os.path.join(subfolder, "weights.json"), solution)
        metrics = run_prediction(
            ds_mode=DsMode.WITH_MPC,
            output_path=subfolder,
            model_path=os.path.join(subfolder, "model.pt"),
            input_steps=5,
            output_steps=10,
            dataset=dataset,
            subset=["test"],
            batch_size=64,
            plot=True
        )
        fitness = metrics["test"]["all_mse"]
        with open(os.path.join(self.base_dir, "fitnesses.csv"), 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([subfolder, idx, -fitness])
        return -fitness


    def on_generation(self, ga: pygad.GA):
        import json
        from json import JSONEncoder
        tobedumped = {
            "best_solutions": ga.best_solutions,
            "best_solution_fitness": ga.best_solutions_fitness,
            "solutions": ga.solutions,
            "solutions_fitness": ga.solutions_fitness,
        }

        class NumpyArrayEncoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, numpy.ndarray):
                    return obj.tolist()
                return JSONEncoder.default(self, obj)

        with open(f"{self.base_dir}\\results_{ga.generations_completed}.json", 'w') as jsondump:
            json.dump(tobedumped, jsondump, indent=3, cls=NumpyArrayEncoder)


def bounds_numeric_0_to_1_bools_0_or_1(dataset):
    var_bounds = []
    for c in dataset.columns:
        c_type = dataset.metadata[c]["type"]
        if c_type == "bool":
            var_bounds.append([0, 1])
        else:
            var_bounds.append({'low': 0.0, 'high': 1.0})
    return var_bounds

def init_population_numeric_pearson_bool_pearson(dataset, epsilon):
    initial_population = []
    for i in range(10):
        solution = []
        for idx, c in enumerate(dataset.columns):
            corr = abs(dataset.mpc_correlations[idx])
            deviation = 2 * epsilon * random.random()
            solution.append(abs(corr) - epsilon + deviation)
        initial_population.append(solution)
    return np.clip(np.array(initial_population), 0.0, 1.0)

def init_population_random(dataset: dss.CoBot202210Data):
    initial_population = []
    for i in range(10):
        solution = []
        for idx, c in enumerate(dataset.columns):
            if dataset.metadata[c]['type'] == 'bool':
                solution.append(float(random.randint(0, 1)))
            else:
                solution.append(random.uniform(0.0, 1.0))
        initial_population.append(solution)
    return np.clip(np.array(initial_population), 0.0, 1.0)


def init_population_numeric_pearson_gaussian_0dot1_bool_pearson(dataset, sigma):
    initial_population = []
    for i in range(10):
        solution = []
        for idx, c in enumerate(dataset.columns):
            corr = abs(dataset.mpc_correlations[idx])
            solution.append(random.gauss(abs(corr), sigma))
        initial_population.append(solution)
    return np.clip(np.array(initial_population), 0.0, 1.0)


def _item_random_01(index, dataset):
    return f"a:\\202303_experiments_random_start_numeric[0-1]_bools[0,1]_{index}", init_population_random(dataset)

def _item_linear(index, dataset, deviation):
    return f"a:\\202303_experiments_start_Pearson_plusminus_{deviation}_numeric[0-1]_bools[0,1]_{index}", init_population_numeric_pearson_bool_pearson(dataset, deviation)

def _item_gaussian(index, dataset, sigma):
    return f"a:\\202303_experiments_start_Pearson_gaussian_{sigma}_numeric[0-1]_bools[0,1]_{index}", init_population_numeric_pearson_gaussian_0dot1_bool_pearson(dataset, sigma)

if __name__ == '__main__':
    dataset_ = DatasetInputData.create(dss.Datasets.CoBot202210)
    for base_dir, initial_population in [
        # _item_gaussian(1, dataset_, 0.4),
        _item_linear(7, dataset_, 0.1),
        _item_linear(7, dataset_, 0.2),
        _item_gaussian(7, dataset_, 0.1),
        _item_gaussian(7, dataset_, 0.2),
        _item_random_01(7, dataset_)
    ]:
        os.makedirs(base_dir, exist_ok=True)

        ff = FitnessFunc(base_dir)

        dataset = DatasetInputData.create(dss.Datasets.CoBot202210)

        var_bounds = bounds_numeric_0_to_1_bools_0_or_1(dataset)

        ga_instance = pygad.GA(
            initial_population=initial_population,
            num_generations=50,
            sol_per_pop=10,
            fitness_func=ff.fitness_func,
            num_genes=len(var_bounds),
            num_parents_mating=4,
            parallel_processing=["process", 8],
            save_solutions=True,
            save_best_solutions=True,
            gene_space=var_bounds,
            on_generation=ff.on_generation,
            crossover_probability=0.8,
            mutation_probability=0.15,
        )

        ga_instance.run()
