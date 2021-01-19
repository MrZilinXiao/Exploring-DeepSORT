import time
import os

from genetic_sort.genetic.population import ConvUnit, ResUnit, PoolUnit, Individual
from utils.general import sync_template


class NetworkBuilder:
    # static vars below should match those in my_template.py
    conv2d_block_name = 'Conv2d_with_padding'
    res_block_name = 'ResidualBlock'
    max_pool_name = 'nn.MaxPool2d'
    mean_pool_name = 'nn.AvgPool2d'
    activation_name = 'nn.ELU'
    drop_out_prob = 0.6
    save_path = './scripts/'
    p1, p2, p3 = None, None, None

    @staticmethod
    def get_last_unit_output_channels(units_list: list):
        for unit in reversed(units_list):
            if hasattr(unit, 'out_channels'):
                return unit.out_channels

    @classmethod
    def read_template(cls, template_path='./template/my_template.py'):
        """
        Read in 3 parts in my_template.py, separated by `#generated_init` and `#generated_forward`
        :param template_path:
        :return:
        """
        if (cls.p1, cls.p2, cls.p3) != (None, None, None):
            return cls.p1, cls.p2, cls.p3
        cls.p1, cls.p2, cls.p3 = [], [], []
        with open(template_path, 'r') as f:
            f.readline()  # skip '"""'
            line = f.readline().rstrip()
            while line.strip() != '#generated_init':
                cls.p1.append(line)
                line = f.readline().rstrip()

            line = f.readline().rstrip()  # skip the comment '#generated_init'
            while line.strip() != '#generated_forward':
                cls.p2.append(line)
                line = f.readline().rstrip()

            line = f.readline().rstrip()  # skip the comment '#generate_forward'
            while line.strip() != '"""':
                cls.p3.append(line)
                line = f.readline().rstrip()
        return cls.p1, cls.p2, cls.p3

    @classmethod
    def generate_py_file(cls, individual: Individual):
        # network part for self.net = nn.ModuleList
        unit_list = ['self.net = nn.Sequential(']
        for layer_id, unit in enumerate(individual.units):  # prepare Module for Module List
            if isinstance(unit, ConvUnit):
                layer_str = '%s(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d),\n' % \
                            (cls.conv2d_block_name, unit.in_channels, unit.out_channels, unit.kernel_size, unit.stride)
                layer_str += 'nn.BatchNorm2d(%d),\n' % unit.out_channels
                layer_str += cls.activation_name + '(inplace=True),\n'
                unit_list.append(layer_str)
            elif isinstance(unit, ResUnit):
                layer_str = '%s(in_channels=%d, out_channels=%d),' % \
                            (cls.res_block_name, unit.in_channels, unit.out_channels)
                unit_list.append(layer_str)
            elif isinstance(unit, PoolUnit):
                layer_str = '%s(kernel_size=%d, stride=%d),' % \
                            (cls.max_pool_name if unit.pool_type == 'max' else cls.mean_pool_name, unit.kernel_size,
                             unit.stride)
                unit_list.append(layer_str)
            else:
                return NotImplemented
        unit_list.append(')')
        _net_str = '        ' + '        \n        '.join(unit_list)

        linear_list = ['self.linear = nn.Sequential(',
                       'nn.Dropout(p=%.2f),' % cls.drop_out_prob,
                       'nn.Linear(in_features=%d, out_features=128),'  # gap
                       % (cls.get_last_unit_output_channels(individual.units) * individual.real_output_size[0] *
                          individual.real_output_size[1]),
                       ')']
        # linear_list = ['self.linear = nn.Sequential(',
        #                'nn.Dropout(p=%.2f),' % cls.drop_out_prob,
        #                'nn.Linear(in_features=%d, out_features=128),'  # gap
        #                % (cls.get_last_unit_output_channels(individual.units)),
        #                ')']
        _linear_str = '        ' + '        \n        '.join(linear_list)

        # forward_list = ['out = self.gap(self.net(input))',
        #                 'out = out.view(out.shape[0], -1)',   # flatten from (N,out_channel,1,1) -> (N, out_channel)
        #                 'out = self.linear(out)'
        #                 ]
        forward_list = ['out = self.net(input)',
                        'out = out.view(out.shape[0], -1)',   # flatten from (N,out_channel,1,1) -> (N, out_channel)
                        'out = self.linear(out)'
                        ]

        _forward_str = '        ' + '        \n        '.join(forward_list)
        p1, p2, p3 = cls.read_template()
        _str = ['"""',
                'Generated on: ' + str(time.strftime("%Y-%m-%d  %H:%M:%S")),
                '"""']
        _str.extend(p1)
        _str.append(_net_str)
        _str.append(_linear_str)
        _str.extend(p2)
        _str.append(_forward_str)
        _str.extend(p3)

        if not os.path.exists(cls.save_path):
            os.mkdir(cls.save_path)
        with open(os.path.join(cls.save_path, '%s.py' % individual.id), 'w') as f:
            f.write('\n'.join(_str))


if __name__ == '__main__':
    from utils.general import ConfigManager

    sync_template()  # sync with the latest template

    config = ConfigManager('./config.yaml')
    individual = Individual(config.config_dict, 'indi0001')
    individual.initialize()
    NetworkBuilder.generate_py_file(individual)
