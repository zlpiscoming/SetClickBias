import os
import sys
import random
import json

import torch


class ClickModel:
    def __init__(self, neg_click_prob=0.0, pos_click_prob=1.0,
                 relevance_grading_num=1, eta=1.0):
        self.exam_prob = None
        self.setExamProb(eta)
        self.setClickProb(
            neg_click_prob,
            pos_click_prob,
            relevance_grading_num)

    @property
    def model_name(self):
        return 'click_model'

    # Serialize model into a json.
    def getModelJson(self):
        desc = {
            'model_name': self.model_name,
            'eta': self.eta,
            'click_prob': self.click_prob,
            'exam_prob': self.exam_prob
        }
        return desc

    # Generate noisy click probability based on relevance grading number
    # Inspired by ERR
    def setClickProb(self, neg_click_prob, pos_click_prob,
                     relevance_grading_num):
        b = (pos_click_prob - neg_click_prob) / \
            (pow(2, relevance_grading_num) - 1)
        a = neg_click_prob - b
        self.click_prob = [
            a + pow(2, i) * b for i in range(relevance_grading_num + 1)]

    # Set the examination probability for the click model.
    def setExamProb(self, eta):
        self.eta = eta
        return

    # Sample clicks for a list
    def sampleClicksForOneList(self, label_list):
        return None

    # Estimate propensity for clicks in a list
    def estimatePropensityWeightsForOneList(
            self, click_list, use_non_clicked_data=False):
        return None

class PositionBiasedModel(ClickModel):

    @property
    def model_name(self):
        return 'position_biased_model'

    def setExamProb(self, eta):
        self.eta = eta
        self.original_exam_prob = [0.68, 0.61, 0.48,
                                   0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06]
        self.exam_prob = [pow(x, eta) for x in self.original_exam_prob]

    def sampleClicksForOneList(self, label_list, ranks):
        click_list, exam_p_list, click_p_list = [], [], []
        for i in range(len(label_list)):
            click, exam_p, click_p = self.sampleClick(ranks[i], label_list[i])
            click_list.append(click)
            exam_p_list.append(exam_p)
            click_p_list.append(click_p)
        return click_list, exam_p_list, click_p_list

    def estimatePropensityWeightsForOneList(
            self, click_list, use_non_clicked_data=False):
        propensity_weights = []
        for r in range(len(click_list)):
            pw = 0.0
            if use_non_clicked_data | click_list[r] > 0:
                pw = 1.0 / self.getExamProb(r) * self.getExamProb(0)
            propensity_weights.append(pw)
        return propensity_weights

    def sampleClick(self, rank, relevance_label):
        if not relevance_label == int(relevance_label):
            print('RELEVANCE LABEL MUST BE INTEGER!')
        relevance_label = int(relevance_label) if relevance_label > 0 else 0
        exam_p = self.getExamProb(rank)
        click_p = self.click_prob[relevance_label if relevance_label < len(
            self.click_prob) else -1]
        click = 1 if random.random() < exam_p * click_p else 0
        return click, exam_p, click_p

    def getExamProb(self, rank):
        return self.exam_prob[rank if rank < len(self.exam_prob) else -1]

def MakeRank(model, data):
    labels = model(data)
    labels, ranks = torch.sort(labels, dim=1, descending=True)
    return labels, ranks


def GenerateClickData(model, data):
    labels, ranks = MakeRank(model, data)
    cm = PositionBiasedModel()
    for i in range(len(labels)):
        labels[i] = cm.sampleClicksForOneList(labels[i], ranks[i])
    return labels

GenerateClickData(model, data)
