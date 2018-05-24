# this file is used for CollectAndDestroy Task
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConvSelectPolicy(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvSelectPolicy, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.select_unit_policy= nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy = nn.Linear(256, 2)

        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.select_task_policy.weight.data)
        nn.utils.weight_norm(self.non_spatial_branch)
        nn.utils.weight_norm(self.value)
        self.non_spatial_branch.bias.data.fill_(0)
        self.value.bias.data.fill_(0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        policy_branch = self.select_unit_policy(x)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        collect_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))
        return collect_prob, value, select_task_prob


class CollectAndDestroyGraftingNet(nn.Module):
    """
    Multitask model with 1 shared hidden layers
    task1:
        select unit and task policy
    task2:
        collect mineral shards
    task3:
        attack buildings
    """
    def __init__(self, screen_channels, screen_resolution):
        self.conv_select = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_collect = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_destroy = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # shared hidden layers
        self.conv2 = nn.Conv2d(49, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.select_unit_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.collect_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.destroy_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy(256, 2)

    def forward(self, x, action_features):
        select_x = F.relu(self.conv_select(x))
        collect_x = F.relu(self.conv_collect(x))
        destroy_x = F.relu(self.conv_destroy(x))

        concat_feature_layers = torch.cat([select_x, collect_x, destroy_x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        # compute prob of unit selection
        select_unit_branch = self.select_unit_policy(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

        # compute prob of collect minerals
        collect_branch = self.collect_policy(x)
        collect_branch = collect_branch.view(collect_branch.shape[0], -1)
        collect_prob = nn.functional.softmax(collect_branch, dim=1)

        # compute prob of destroy buildings
        destroy_branch = self.destroy_policy(x)
        destroy_branch = destroy_branch.view(destroy_branch.shape[0], -1)
        destroy_prob = nn.functional.softmax(destroy_branch, dim=1)

        # non spatial branch
        # flatten
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))

        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_unit_policy(non_spatial_represenatation))

        return select_unit_prob, collect_prob, destroy_prob, value, select_task_prob

