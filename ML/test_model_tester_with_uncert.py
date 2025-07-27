# test_model_tester_with_uncert.py

import numpy as np
import tempfile
import os
import model_tester_with_uncert

def test_generate_posterior_samples(tmp_path):
    # Prepare dummy data
    y_pred = np.array([
        [0.5, -2.0, 0.1],
        [0.7, -1.5, 0.2]
    ])
    y_test = np.array([
        [0.5, -2.0],
        [0.7, -1.5]
    ])
    numsamples = 5
    output_dir = tmp_path
    label = "_test"

    # Patch logger to avoid errors
    import types
    model_tester_with_uncert.logger = types.SimpleNamespace()
    model_tester_with_uncert.logger.info = print

    # Call the function
    model_tester_with_uncert.generate_posterior_samples(y_pred, y_test, numsamples, output_dir, label)

    # Check file exists
    filename = os.path.join(output_dir, f"generated_posterior_samples{label}.npy")
    assert os.path.exists(filename)

    # Load and check shape
    print(f"filename={filename}")
    data = np.load(filename)
    # Should have numsamples * nrows rows, and at least 4 columns (2 generated + 2 test)
    assert data.shape[0] == y_pred.shape[0] * numsamples
    assert data.shape[1] >= 4

    # Check that the test values are repeated correctly
    for i in range(y_pred.shape[0]):
        test_vals = y_test[i]
        # Each test value should be repeated numsamples times in the output
        np.testing.assert_array_equal(
            data[i*numsamples:(i+1)*numsamples, 2:4],
            np.tile(test_vals, (numsamples, 1))
        )

if __name__ == "__main__":
    import pytest
    pytest.main([__file__]) 