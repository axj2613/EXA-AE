import sys
import configparser
import pickle

from loguru import logger
from torch import optim

from population.single_population import SinglePopulation
from evolution.vit_mae_reproduction_selector import ViTMAEReproductionSelector

from genomes.vit_mae_genome import VisionTransformerMAEGenome, DEFAULT_HYPERPARAMETERS

from time_series.fmri_patch_dataset import FMRIPatchDataset

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=True, diagnose=True)

    config = configparser.ConfigParser()
    config.read("exa_vit_mae_config.ini")

    training_data = [
        filename.strip() for filename in config.get("DEFAULT", "training_data").split(",") if filename.strip()
    ]
    atlas_coordinates = config.get("DEFAULT", "atlas_coordinates")

    window_length = config.getint("DEFAULT", "window_length")
    population_size = config.getint("DEFAULT", "population_size")
    num_generations = config.getint("DEFAULT", "num_generations")
    num_iterations = config.getint("DEFAULT", "num_iterations")
    batches_per_iteration = config.getint("DEFAULT", "batches_per_iteration")
    batch_size = config.getint("DEFAULT", "batch_size")
    learning_rate = config.getfloat("DEFAULT", "learning_rate")

    dataset = FMRIPatchDataset(npz_filenames=training_data, atlas_coordinates_filename=atlas_coordinates)
    logger.info(f"loaded {len(dataset.recordings)} recording(s), {dataset.num_parcels} parcels")

    seed_genome = VisionTransformerMAEGenome(
        generation_number=0,
        num_parcels=dataset.num_parcels,
        window_length=window_length,
        parcel_coordinates=dataset.parcel_coordinates,
        hyperparameters=DEFAULT_HYPERPARAMETERS,
    )

    population = SinglePopulation(
        population_size=population_size,
        seed_genome=seed_genome,
        reproduction_selector=ViTMAEReproductionSelector(
            num_parcels=dataset.num_parcels,
            window_length=window_length,
            parcel_coordinates=dataset.parcel_coordinates,
        ),
    )

    for genome_number in range(num_generations):
        new_genome = population.generate_genome()
        print(f"evaluating genome: {new_genome.generation_number}, hyperparameters: {new_genome.hyperparameters}")
        optimizer = optim.Adam(new_genome.parameters(), lr=learning_rate)

        new_genome.train(
            dataset=dataset,
            optimizer=optimizer,
            iterations=num_iterations,
            batch_size=batch_size,
            batches_per_iteration=batches_per_iteration,
        )
        population.insert_genome(new_genome)

    print()
    print()

    best_fit_genome = population.population[0]
    print(f"{best_fit_genome}")

    # save the best-fit genome object
    pkl_filename = "./test_genomes/vit_mae_genome_" + str(best_fit_genome.generation_number) + ".pkl"
    with open(pkl_filename, "wb") as pkl_file:
        pickle.dump(best_fit_genome, pkl_file)
