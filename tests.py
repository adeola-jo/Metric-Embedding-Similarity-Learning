import unittest
import torch
from utils import compute_representations
from utils import make_predictions
from model import SimpleMetricEmbedding
from utils import evaluate



# ------------------ Tests from lab4/dataset.py ------------------
class TestSimpleMetricEmbedding(unittest.TestCase):
    def test_get_features(self):
        model = SimpleMetricEmbedding(1, 32)
        img = torch.randn(1, 1, 28, 28)
        output = model.get_features(img)
        self.assertEqual(output.shape, (1, 32))

    def test_loss(self):
        model = SimpleMetricEmbedding(1, 32)
        anchor = torch.randn(1, 1, 28, 28)
        positive = torch.randn(1, 1, 28, 28)
        negative = torch.randn(1, 1, 28, 28)
        output = model.loss(anchor, positive, negative)
        self.assertEqual(output.shape, ())
        self.assertTrue(output >= 0)


# ------------------ Tests from lab4/utils.py ------------------
class TestEvaluate(unittest.TestCase):
    def test_evaluate(self):
        model = SimpleMetricEmbedding(1, 32)
        repr = torch.randn(10, 32)
        loader = [torch.randn(1, 1, 28, 28) for _ in range(5)]
        device = 'cpu'
        output = evaluate(model, repr, loader, device)
        self.assertTrue(0 <= output <= 1)

class TestMakePredictions(unittest.TestCase):
    def test_make_predictions(self):
        representations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        r = torch.tensor([1.0, 2.0])
        expected_output = torch.tensor([0.0, 8.0])
        output = make_predictions(representations, r)
        self.assertTrue(torch.allclose(output, expected_output))

class TestComputeRepresentations(unittest.TestCase):
    def test_compute_representations(self):
        # Create a mock model
        # model = torch.nn.Linear(10, 10)
        model = SimpleMetricEmbedding(1, 32)
        
        # Create a mock data loader
        data_loader = [torch.randn(1, 1, 28, 28) for _ in range(5)]
        
        # Call the function
        output = compute_representations(model, data_loader, identities_count=5, emb_size=10, device='cpu')
        
        # Check the output shape
        self.assertEqual(output.shape, (5, 10))

        # Check that the output is normalized
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.tensor([1.0]*5)))

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromName('test_compute_representations', TestComputeRepresentations)
    unittest.TextTestRunner().run(suite)