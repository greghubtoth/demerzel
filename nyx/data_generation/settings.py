from nyx.data_generation.data_generators import (
    BaselineLeeEtAlDataGenerator, BaselineLeeEtAlDataGeneratorWithLangChain,
    ExpelZhaoEtAlAdaptedDataGenerator)

BASELINE_LEE_ET_AL = 'Baseline-Lee-et-al'
ADAPTED_EXPEL_ET_AL = 'Adapted-ExpeL-et-al'

LABELLING_CLASSES = {
    f'{BASELINE_LEE_ET_AL}_single_gpu': BaselineLeeEtAlDataGenerator,
    BASELINE_LEE_ET_AL: BaselineLeeEtAlDataGeneratorWithLangChain,
    ADAPTED_EXPEL_ET_AL: ExpelZhaoEtAlAdaptedDataGenerator,
}
