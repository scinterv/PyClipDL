import os
import torch


"""Args:
    data_file: data file where aggregated data points stored

    format: aggregated_points_1 | original_points_1 | original_points_2 | ...
            aggregated_points_2 | original_points_1 | original_points_2 | ...
    for each points: label; feature_1, feature,2, ...
"""


def load_agg_data(data_file):
    if os.path.isfile(data_file):
        _data = []
        _targets = []
        with open(data_file) as infile:
            for line in infile:
                nline = line.strip().split('|')
                sample_data = []
                sample_target = []
                for i in range(len(nline)):
                    sample = nline[i].strip().split(';')
                    target = int(float(sample[0])) - 1
                    features = list(map(lambda v: float(v),
                                        sample[1].split(',')))
                    sample_data.append(features)
                    sample_target.append(target)
                _data.append(sample_data)
                _targets.append(sample_target)
        return torch.tensor(_data), torch.tensor(_targets)
    else:
        _data = None
        _targets = None
        for item in os.listdir(data_file):
            nfile = os.path.join(data_file, item)
            if _data is None and _targets is None:
                _data, _targets = load_agg_data(nfile)
            else:
                tdata, ttargets = load_agg_data(nfile)
                _data = torch.cat((_data, tdata), dim=0)
                _targets = torch.cat((_targets, ttargets), dim=0)
        return _data, _targets


"""Args:
     data_file: data file where test points stored

    format: original_points_1
            original_points_2
            ...
    for each points: label; feature_1, feature,2, ...
"""


def load_test_data(data_file):
    if os.path.isfile(data_file):
        _data = []
        _targets = []
        with open(data_file) as infile:
            for line in infile:
                nline = line.strip().split(';')
                target = int(float(nline[0])) - 1
                features = list(map(lambda v: float(v), nline[1].split(',')))
                _targets.append(target)
                _data.append(features)
        return torch.tensor(_data), torch.tensor(_targets)
    else:
        _data = None
        _targets = None
        for item in os.listdir(data_file):
            nfile = os.path.join(data_file, item)
            if _data is None and _targets is None:
                _data, _targets = load_test_data(nfile)
            else:
                tdata, ttargets = load_test_data(nfile)
                _data = torch.cat((_data, tdata), dim=0)
                _targets = torch.reshape(_targets, (-1, 1))
                ttargets = torch.reshape(ttargets, (-1, 1))
                _targets = torch.cat((_targets, ttargets), dim=0)
        return _data, _targets
