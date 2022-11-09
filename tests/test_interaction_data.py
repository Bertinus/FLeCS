from flecs.data.interaction_data import load_interaction_data


def test_load_interaction_data():

    NETWORKS = {
        "calcium_pathway": [{}],
        "regulon_db": [{}, {"tf_only": True}],
        "encode": [{}, {"tf_only": True, "subsample_edge_prop": 0.5}],
        "fantom5": [
            {"realnet_tissue_type_file": "01_neurons_fetal_brain.txt.gz"},
            {
                "realnet_tissue_type_file": "01_neurons_fetal_brain.txt.gz",
                "tf_only": True,
            },
            {
                "realnet_tissue_type_file": "01_neurons_fetal_brain.txt.gz",
                "tf_only": True,
                "subsample_edge_prop": 0.5,
            },
        ],
        "string": [{"subsample_edge_prop": 0.5}],
        "random": [{"n_nodes": 10, "avg_num_parents": 3}],
    }

    for network_name, all_kwargs in NETWORKS.items():
        for kwargs in all_kwargs:
            try:
                res = load_interaction_data(network_name, **kwargs).__repr__()
                print(res)
            except:
                raise Exception(
                    "Network={}, args={} failed!".format(network_name, kwargs)
                )
