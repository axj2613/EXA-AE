import sys
import configparser
from loguru import logger
from torch import optim
import pickle

from evolution.exagp import EXAGP
from evolution.bidirectionalAE_node_generator import BidirectionalAENodeGenerator
from evolution.bidirectionalAE_edge_generator import BidirectionalAEEdgeGenerator

from genomes.minimal_recurrent_genome import MinimalRecurrentGenome
from genomes.trivial_recurrent_genome import TrivialRecurrentGenome
from genomes.bAE_genome import BidirectionalAEGenome
from genomes.autoencoder_genome import AutoencoderGenome

from time_series.time_series import TimeSeries

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=True, diagnose=True)

    config = configparser.ConfigParser()
    config.read('exa_transformer_ae_config.ini')

    fmri_npz = config['DEFAULT']['training_data']
    num_generations = config.getint('DEFAULT', 'num_generations')
    num_iterations = config.getint('DEFAULT', 'num_iterations')
    learning_rate = config.getfloat('DEFAULT', 'learning_rate')

    initial_series = TimeSeries.create_from_fmri_npz(filename=fmri_npz)
    logger.info("Initial series loaded.")
    # logger.info(f"Keys: {list(initial_series.series_dictionary.keys())}")

    max_sequence_length = initial_series.series_length
    logger.info(f"max sequence length: {max_sequence_length}")

    seed_genome = AutoencoderGenome(
            generation_number=0,
            input_series_names=initial_series.series_dictionary.keys(),
            output_series_names=initial_series.series_dictionary.keys(),
            max_sequence_length=max_sequence_length,
        )

    exagp = EXAGP(seed_genome=seed_genome, autoencoder=True)

    for genome_number in range(num_generations):
        new_genome = exagp.generate_genome()
        print(f"evaluating genome: {new_genome.generation_number}")
        optimizer = optim.Adam(new_genome.parameters(), lr=learning_rate)

        new_genome.train(
            input_series=initial_series,
            output_series=initial_series,
            optimizer=optimizer,
            iterations=num_iterations,
        )
        exagp.insert_genome(new_genome)

    print()
    print()

    best_fit_genome = exagp.population_strategy.population[0]
    print(f"{best_fit_genome}")

    # save the best-fit genome object
    pkl_filename = './test_genomes/genome_' + str(best_fit_genome.generation_number) + '.pkl'
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(best_fit_genome, pkl_file)