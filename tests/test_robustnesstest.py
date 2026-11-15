import pytest
import torch
from robustnesstest import RobustnessTest, create_robustnesstest

class TestRobustnessTest:
    def test_init(self):
        m = RobustnessTest(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = RobustnessTest(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_robustnesstest()
        assert isinstance(m, RobustnessTest)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
