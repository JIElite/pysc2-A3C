import torch.nn as nn
import torch.nn.functional as F



class FullyConv(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConv, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

        # init weight
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.spatial_policy.weight.data)
        nn.utils.weight_norm(self.non_spatial_branch)
        nn.utils.weight_norm(self.value)
        self.non_spatial_branch.bias.data.fill_(0)
        self.value.bias.data.fill_(0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)
        return action_prob, value


class FullyConvExtended(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvExtended, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

        # init weight
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.conv3.weight.data)
        nn.init.xavier_uniform(self.spatial_policy.weight.data)
        nn.utils.weight_norm(self.non_spatial_branch)
        nn.utils.weight_norm(self.value)
        self.non_spatial_branch.bias.data.fill_(0)
        self.value.bias.data.fill_(0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)
        return action_prob, value


class FullyConvMultiUnitCollectBaseline(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvMultiUnitCollectBaseline, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

        # init weight
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.spatial_policy.weight.data)
        nn.init.xavier_uniform(self.select_unit.weight.data)
        nn.utils.weight_norm(self.non_spatial_branch)
        nn.utils.weight_norm(self.value)
        self.non_spatial_branch.bias.data.fill_(0)
        self.value.bias.data.fill_(0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # select unit branch
        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value


class FullyConvMultiUnitCollectBaselineExtended(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvMultiUnitCollectBaselineExtended, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

        # init weight
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.conv3.weight.data)
        nn.init.xavier_uniform(self.spatial_policy.weight.data)
        nn.init.xavier_uniform(self.select_unit.weight.data)
        nn.utils.weight_norm(self.non_spatial_branch)
        nn.utils.weight_norm(self.value)
        self.non_spatial_branch.bias.data.fill_(0)
        self.value.bias.data.fill_(0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # select unit branch
        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob =  nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value


class FullyConvSelecAction(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvSelecAction, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)

        # means task action or no_op
        self.non_spatial_policy = nn.Linear(256, 2)
        # The value is used to evaluation certain state, so that the output size is 1
        self.value = nn.Linear(256, 1)

        # init weight
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.spatial_policy.weight.data)
        nn.utils.weight_norm(self.non_spatial_branch)
        nn.utils.weight_norm(self.non_spatial_policy)
        nn.utils.weight_norm(self.value)
        self.non_spatial_branch.bias.data.fill_(0)
        self.non_spatial_policy.bias.data.fill_(0)
        self.value.bias.data.fill_(0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # spatial policy branch
        spatial_policy_branch = self.spatial_policy(x)
        policy_branch = spatial_policy_branch.view(spatial_policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial policy branch
        flatten_state_represenation = F.relu(self.non_spatial_branch(x.view(-1)))
        non_spatial_policy = self.non_spatial_policy(flatten_state_represenation).unsqueeze(0)
        non_spatial_policy_prob = F.softmax(non_spatial_policy, dim=1)
        value = self.value(flatten_state_represenation)

        return non_spatial_policy_prob, spatial_action_prob, value
