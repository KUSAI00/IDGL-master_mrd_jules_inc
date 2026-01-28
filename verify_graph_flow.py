
import torch
import torch.nn as nn
from src.core.models.text_graph import TextGraphRegression
from src.core.utils.bert_utils import DistilBertEmbeddings
from src.core.utils.generic_utils import create_mask

# Base config
base_config = {
    'model_name': 'TextGraphRegression',
    'device': 'cpu',
    'word_embed_dim': 768,
    'hidden_size': 64,
    'dropout': 0.1,
    'word_dropout': 0.1,
    'rnn_dropout': 0.1,
    'graph_learn': True, # Enable graph learning
    'graph_learn_hidden_size': 32,
    'graph_learn_topk': 5,
    'graph_learn_epsilon': 0.1,
    'graph_learn_num_pers': 2,
    'graph_metric_type': 'cosine',
    'graph_learn_regularization': False,
    'graph_module': 'gcn',
    'graph_skip_conn': 0.8,
    'fix_vocab_embed': True,
    'input_graph_knn_size': 5,
    'use_distilbert': True,
    'scalable_run': False,
    'graph_hops': 2,
    'batch_norm': False,
    'no_gnn': False
}

class MockVocab:
    pass

def test_model():
    print("Initializing DistilBertEmbeddings...")
    w_embedding = DistilBertEmbeddings()

    print("\n--- Verifying Graph Flow with DistilBERT ---")
    model = TextGraphRegression(base_config, w_embedding, MockVocab())

    # Check graph_learner input dimension
    # It stores 'input_size' in some way? GraphLearner init:
    # self.weight = nn.Parameter(torch.Tensor(input_size, hidden_size)) (in some versions)
    # The patch updated init to use hidden_size (64) instead of 768.
    # Let's inspect the weights of graph_learner.

    gl_weight_shape = model.graph_learner.weight.shape
    print(f"Graph Learner Weight Shape: {gl_weight_shape}")
    # AnchorGraphLearner/GraphLearner usually has a weight for metric learning
    # Expected: [64, graph_learn_hidden_size] or similar. Definitely not [768, ...]

    batch_size = 2
    seq_len = 10
    # Dummy input_ids (must be valid ranges for BERT if we care about output, but for shapes it matters less unless OOB)
    context = torch.randint(0, 1000, (batch_size, seq_len))
    context_lens = torch.tensor([seq_len, seq_len])

    print("Running prepare_init_graph...")
    raw, ctx, mask, adj = model.prepare_init_graph(context, context_lens)
    print(f"Raw Context (Projected) Shape: {raw.shape}") # Should be [2, 10, 64]
    print(f"Context (Projected) Shape: {ctx.shape}") # Should be [2, 10, 64]

    if raw.shape[-1] == 64:
        print("Success: Projected features used for graph construction.")
    else:
        print(f"Failure: Expected 64 dim, got {raw.shape[-1]}")

    print("Running Encoder...")
    # This invokes model.learn_graph internally if graph_learn is True
    # If raw is 64-dim and graph_learner expects 64-dim, this should pass.
    # If raw was 768-dim, it would crash.

    # We need to manually simulate forward or call encoder + learn_graph logic?
    # TextGraphRegression does not implement forward() in the snippet provided in `read_file`.
    # It seems `forward` was truncated or I missed reading it.
    # But I can call learn_graph directly.

    print("Running learn_graph...")
    # learn_graph(self, graph_learner, node_features, ...)
    try:
        res = model.learn_graph(model.graph_learner, raw, node_mask=mask)
        print("learn_graph ran successfully.")
    except RuntimeError as e:
        print(f"learn_graph failed: {e}")

if __name__ == "__main__":
    test_model()
