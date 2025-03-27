"""
We are evaluating a binary classification model. We have the number of true positives, true
negatives, false positives, and false negatives for confidence score thresholds 0.1, 0.2, 0.3, ...,
0.9 respectively (feel free to assume the data structure for this input data and provide an
explanation).
Write a function to return THE BEST threshold that yields a recall >= 0.9. Unit tests for
this function are also encouraged.

(Note that you don’t have to write unit tests for this function. Alternatively, you must
provide example code for calling the function with realistic input data.)
"""

import unittest

class InvalidDataException(Exception):
    pass

#
# data structure is composed of 4 lists: true positives, true negatives, false positives and false negatives
# each list is assumed to have 9 entries one for each confidence score threshold. The index i in each list
# has the value associated with the confidence score of (i+1)/10 for each category
# 
# to calculate the recall you only need the true positive and the false negative values
# recall = tp / (tp + fn)
#

def find_best_recall_threshold(tp_list: list[int], fn_list: list[int], recall_threshold: float = 0.9) -> float | None :
    """
    Finds the best threshold for a binary classification model, aiming for a recall >= recall_threshold parameter.

    Args:
        tp_list: A list of true positives for each threshold (0.1, 0.2, ..., 0.9).
        fn_list: A list of false negatives for each threshold (0.1, 0.2, ..., 0.9).

    Returns:
        The best threshold (float) that yields a recall >= recall_threshold (default to 0.9), or None if no threshold meets the criteria.
    """
    if len(tp_list) != 9 or len(fn_list) != len(tp_list):
        raise InvalidDataException()
    
    best_threshold = None
    # Initialize with a value lower than any possible recall
    best_recall = -1  

    for i, (tp, fn) in enumerate(zip(tp_list, fn_list)):
        threshold = (i + 1) / 10  # Thresholds are 0.1, 0.2, ..., 0.9

        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero

        if recall >= recall_threshold and (best_recall < recall or best_threshold is None):
            best_recall = recall
            best_threshold = threshold

    return best_threshold


class TestBestRecallThreshold(unittest.TestCase):
    def test_is_not_none(self):
        tp_list = [100, 90, 80, 70, 60, 50, 40, 30, 20]
        fn_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        best_threshold = find_best_recall_threshold(tp_list, fn_list, 0.9)
        self.assertIsNotNone(best_threshold)

    def test_expect_9(self):
        tp_list = [100, 90, 80, 70, 60, 50, 40, 30, 20]
        fn_list = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        best_threshold = find_best_recall_threshold(tp_list, fn_list, 0.9)
        self.assertIsNotNone(best_threshold)
        self.assertEqual(best_threshold, 0.9)

    def test_expect_real(self):
        tp_list = [100, 95, 90, 70, 60, 50, 40, 30, 20]
        fn_list = [4, 3, 2, 1, 2, 4, 5, 6, 7]
        best_threshold = find_best_recall_threshold(tp_list, fn_list, 0.9)
        self.assertIsNotNone(best_threshold)
        print(best_threshold)
        self.assertEqual(best_threshold, 0.4)

    def test_bad_data_sizes(self):
        tp_list = [100, 90, 80, 70, 60, 50, 40, 30, 20]
        fn_list = [2, 3, 4, 5, 6, 7, 8, 9]
        with self.assertRaises(InvalidDataException):
            best_threshold = find_best_recall_threshold(tp_list, fn_list, 0.9)
            self.assertIsNotNone(best_threshold)

    def test_bad_data_length(self):
        tp_list = [90, 80, 70, 60, 50, 40, 30, 20]
        fn_list = [2, 3, 4, 5, 6, 7, 8, 9]
        with self.assertRaises(InvalidDataException):
            best_threshold = find_best_recall_threshold(tp_list, fn_list, 0.9)
            self.assertIsNotNone(best_threshold)

if __name__ == "__main__":
    unittest.main()
