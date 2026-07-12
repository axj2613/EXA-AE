import sys
import configparser
import pickle
from tqdm import tqdm

from loguru import logger
from torch import optim

from population.single_population import SinglePopulation

from evolution.vision_transformer_block_edge_generator import VisionTransformerBlockEdgeGenerator
from evolution.vision_transformer_block_node_generator import VisionTransformerBlockNodeGenerator
from evolution.vision_transformer_block_reproduction_selector import VisionTransformerBlockReproductionSelector

from genomes.vision_transformer_block_genome import VisionTransformerBlockGenome

from time_series.fmri_patch_dataset import FMRIPatchDataset

from weight_generators.lamarckian_block_weight_generator import LamarckianBlockWeightGenerator

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=True, diagnose=True)

    config = configparser.ConfigParser()
    config.read("exa_vit_mae_evolved_config.ini")

    training_data = [
        filename.strip() for filename in config.get("DEFAULT", "training_data").split(",") if filename.strip()
    ]
    atlas_coordinates = config.get("DEFAULT", "atlas_coordinates")

    window_length = config.getint("DEFAULT", "window_length")
    d_model = config.getint("DEFAULT", "d_model")
    num_heads = config.getint("DEFAULT", "num_heads")
    d_ff = config.getint("DEFAULT", "d_ff")
    dropout = config.getfloat("DEFAULT", "dropout")
    mask_ratio = config.getfloat("DEFAULT", "mask_ratio")

    # comma-separated list controlling which block types evolution may introduce, e.g.
    # "attention" for an attention-only search or the full list for a mixed-cell search.
    node_types = [
        t.strip() for t in
        config.get("DEFAULT", "node_types", fallback="attention,simple,sequence_lstm,temporal_lstm").split(",")
        if t.strip()
    ]

    population_size = config.getint("DEFAULT", "population_size")
    num_generations = config.getint("DEFAULT", "num_generations")
    num_iterations = config.getint("DEFAULT", "num_iterations")
    batches_per_iteration = config.getint("DEFAULT", "batches_per_iteration")
    batch_size = config.getint("DEFAULT", "batch_size")
    learning_rate = config.getfloat("DEFAULT", "learning_rate")

    dataset = FMRIPatchDataset(npz_filenames=training_data, atlas_coordinates_filename=atlas_coordinates)
    logger.info(f"loaded {len(dataset.recordings)} recording(s), {dataset.num_parcels} parcels")

    weight_generator = LamarckianBlockWeightGenerator()

    seed_genome = VisionTransformerBlockGenome(
        generation_number=0,
        num_parcels=dataset.num_parcels,
        window_length=window_length,
        parcel_coordinates=dataset.parcel_coordinates,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        mask_ratio=mask_ratio,
        weight_generator=weight_generator,
    )

    node_generator = VisionTransformerBlockNodeGenerator(
        num_heads=num_heads, d_ff=d_ff, dropout=dropout, allowed_node_types=node_types
    )
    logger.info(f"evolving with node types: {node_generator.allowed_node_types}")
    edge_generator = VisionTransformerBlockEdgeGenerator()

    population = SinglePopulation(
        population_size=population_size,
        seed_genome=seed_genome,
        reproduction_selector=VisionTransformerBlockReproductionSelector(
            node_generator=node_generator,
            edge_generator=edge_generator,
            weight_generator=weight_generator,
        ),
    )

    for genome_number in tqdm(range(num_generations)):
        new_genome = population.generate_genome()
        print(f"evaluating genome: {new_genome.generation_number}")
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
    pkl_filename = "./test_genomes/vit_mae_evolved_genome_" + str(best_fit_genome.generation_number) + ".pkl"
    with open(pkl_filename, "wb") as pkl_file:
        pickle.dump(best_fit_genome, pkl_file)
