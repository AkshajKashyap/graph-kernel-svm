from graph_kernel_svm.data import SUPPORTED_TU_DATASETS
from graph_kernel_svm.scripts.download_tu_dataset import build_parser


def test_download_parser_supports_expected_datasets() -> None:
    assert SUPPORTED_TU_DATASETS == ("MUTAG", "PTC_MR", "PROTEINS")

    for dataset in SUPPORTED_TU_DATASETS:
        args = build_parser().parse_args(["--dataset", dataset])
        assert args.dataset == dataset
