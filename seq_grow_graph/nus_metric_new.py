import sys
import argparse
from projects.SeqGrowGraph.seq_grow_graph.nus_reach_metric import NuScenesReachMetric
from mmengine import Config, load
from mmengine.logging import MMLogger
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

data_root = "./data/nuscenes/"

# grid_conf = dict(
#         xbound=[1.0, 50.0, 0.5],
#         ybound=[-25.0, 25.0, 0.5],
#         zbound=[-10.0, 10.0, 20.0],
#         dbound=[4.0, 48.0, 1.0],)

grid_conf = dict(
    xbound=[-48.0, 48.0, 0.5],
    ybound=[-32.0, 32.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 48.0, 1.0],
)

bz_grid_conf = dict(
    xbound=[-55.0, 55.0, 0.5],
    ybound=[-55.0, 55.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 48.0, 1.0],
)

backend_args = None

params = dict(
    data_root=data_root,
    ann_file=data_root + "nuscenes_centerline_infos_val.pkl",
    metric="ar_reach",
    backend_args=backend_args,
    grid_conf=grid_conf,
    bz_grid_conf=bz_grid_conf,
    jsonfile_prefix="save_result",
    landmark_thresholds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    reach_thresholds=[1, 2, 3, 4, 5],
    is_new=True,
)

class NewNuScenesReachMetric(NuScenesReachMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_metrics(self, result_path) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        print(result_path)
        # classes = self.dataset_meta['classes']
        # self.version = self.dataset_meta['version']
        # load annotations
        self.data_infos = load(self.ann_file, backend_args=self.backend_args)["infos"]
        # result_path, tmp_dir = self.format_results(results, classes,
        #    self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(f"results are saved in {osp.basename(self.jsonfile_prefix)}")
            return metric_dict

        for metric in self.metrics:
            ap_dict = self._evaluate_single(result_path, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()
        return metric_dict

def main():
    parser = argparse.ArgumentParser(description='Compute NuScenes Reach Metrics')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to the results JSON file')
    
    args = parser.parse_args()
    
    metric = NewNuScenesReachMetric(**params)
    metric.compute_metrics(result_path=args.result_path)

if __name__ == "__main__":
    main()