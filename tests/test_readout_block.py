from tests.test_utils import load_data_model


def test_readout_block():
    batch, model = load_data_model()

    max_nei = model.molecular_graph_cfg.max_neighbors
    hidden_size = model.global_cfg.hidden_size

    x = model.data_preprocess(batch)
    N = x.node_padding_mask.shape[0]

    node_features, edge_features = model.exportable_model.input_block(x)

    node_readout, edge_readout = model.exportable_model.readout_layers[0](
        node_features, edge_features
    )

    assert node_readout.shape == (N, hidden_size)
    assert edge_readout.shape == (N, max_nei, hidden_size)
