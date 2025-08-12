import numpy as np


class Metrics:
    @staticmethod  # don't have to instantiate the class to use these methods, so you can just call them directly using Metrics.accuracy(), Utility functions logically related to the class
    def confusion_matrix(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    @staticmethod
    def accuracy(y_true, y_pred):
        """Calculate accuracy of the model"""
        # conf_matrix = Metrics.confusion_matrix(y_true, y_pred)
        # tp, fp, tn, fn = conf_matrix.get('TP'), conf_matrix.get('FP'), conf_matrix.get('TN'), conf_matrix.get('FN')
        # return (tp+tn)/(tp+fp+tn+fn)
        correct_predictions = np.sum(y_true == y_pred)
        return correct_predictions / len(y_true)

    @staticmethod
    def precision(y_true, y_pred):
        # true positives / predicted positives

        # conf_matrix = Metrics.confusion_matrix(y_true, y_pred)
        # tp, fp, tn, fn = conf_matrix.get('TP'), conf_matrix.get('FP'), conf_matrix.get('TN'), conf_matrix.get('FN')
        # if tp + fp == 0:
        #     return 0
        # return tp / (tp+fn)

        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))

        if true_positive + false_positive == 0:
            return 0.0  # Avoid division by zero
        return true_positive / (true_positive + false_positive)

    @staticmethod
    def recall(y_true, y_pred):
        # true positives / actual positives

        # conf_matrix = Metrics.confusion_matrix(y_true, y_pred)
        # tp, fp, tn, fn = conf_matrix.get('TP'), conf_matrix.get('FP'), conf_matrix.get('TN'), conf_matrix.get('FN')
        # if tp + fp == 0:
        #     return 0
        # return tp / (tp + fn)

        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_negative = np.sum((y_true == 1) & (y_pred == 0))

        if true_positive + false_negative == 0:
            return 0.0  # Avoid division by zero
        return true_positive / (true_positive + false_negative)

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculate the F1 score of the model.
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        p = Metrics.precision(y_true, y_pred)
        r = Metrics.recall(y_true, y_pred)

        if p + r == 0:
            return 0.0  # Avoid division by zero
        return 2 * (p * r) / (p + r)
