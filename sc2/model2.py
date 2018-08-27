# this file is used for CollectAndDestroy Task
import torch
import torch.nn as nn
import torch.nn.functional as F


class CollectFiveDropout(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(CollectFiveDropout, self).__init__()
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)

        # grafting
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

        self.dropout_rate = 0.95

    def anneal_dropout(self):
        new_dropout_rate = self.dropout_rate - 0.0001
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, task_type):
        if task_type == 0:
            master_x = F.relu(self.conv_master(x))
            sub_x = F.relu(self.conv_sub(x))
            sub_x - F.dropout2d(sub_x, p=self.dropout_rate, training=self.training)

            concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
            x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))

            select_unit_branch = self.select_unit(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

            # non spatial branch
            non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
            value = self.value(non_spatial_represenatation)
            return select_unit_prob, None, value, None

        elif task_type == 1:
            master_x = F.relu(self.conv_master(x))
            master_x = F.dropout2d(master_x, p=self.dropout_rate, training=self.training)
            sub_x = F.relu(self.conv_sub(x))
            concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
            x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))

            # spatial policy branch
            policy_branch = self.spatial_policy(x)
            spatial_vis = policy_branch
            policy_branch = policy_branch.view(policy_branch.shape[0], -1)
            spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

            # non spatial branch
            non_spatial_represenatation = F.relu(
                self.non_spatial_branch(x.view(-1)))  # flatten the state representation
            value = self.value(non_spatial_represenatation)
            return None, spatial_action_prob, value, F.softmax(spatial_vis[0][0], dim=1)


class CollectFiveDropoutConv3(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(CollectFiveDropoutConv3, self).__init__()
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(32)

        # grafting
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

        self.dropout_rate = 0.95

    def anneal_dropout(self):
        new_dropout_rate = self.dropout_rate - 0.0001
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, task_type):
        if task_type == 0:
            master_x = F.relu(self.conv_master(x))
            sub_x = F.relu(self.conv_sub(x))
            sub_x - F.dropout2d(sub_x, p=self.dropout_rate, training=self.training)

            concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
            x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(self.conv4_bn(self.conv4(x)))

            select_unit_branch = self.select_unit(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

            # non spatial branch
            non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
            value = self.value(non_spatial_represenatation)
            return select_unit_prob, None, value, None

        elif task_type == 1:
            master_x = F.relu(self.conv_master(x))
            master_x = F.dropout2d(master_x, p=self.dropout_rate, training=self.training)
            sub_x = F.relu(self.conv_sub(x))
            concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
            x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))
            x = F.relu(self.conv3_bn(self.conv3(x)))

            # spatial policy branch
            policy_branch = self.spatial_policy(x)
            spatial_vis = policy_branch
            policy_branch = policy_branch.view(policy_branch.shape[0], -1)
            spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

            # non spatial branch
            non_spatial_represenatation = F.relu(
                self.non_spatial_branch(x.view(-1)))  # flatten the state representation
            value = self.value(non_spatial_represenatation)
            return None, spatial_action_prob, value, F.softmax(spatial_vis[0][0], dim=1)


class CollectFiveDropoutConv4(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(CollectFiveDropoutConv4, self).__init__()
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(32)

        # grafting
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

        self.dropout_rate = 0.95

    def anneal_dropout(self):
        new_dropout_rate = self.dropout_rate - 0.0001
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, task_type):
        if task_type == 0:
            master_x = F.relu(self.conv_master(x))
            sub_x = F.relu(self.conv_sub(x))
            sub_x - F.dropout2d(sub_x, p=self.dropout_rate, training=self.training)

            concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
            x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(self.conv4_bn(self.conv4(x)))

            select_unit_branch = self.select_unit(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

            # non spatial branch
            non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
            value = self.value(non_spatial_represenatation)
            return select_unit_prob, None, value, None

        elif task_type == 1:
            master_x = F.relu(self.conv_master(x))
            master_x = F.dropout2d(master_x, p=self.dropout_rate, training=self.training)
            sub_x = F.relu(self.conv_sub(x))
            concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
            x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(self.conv4_bn(self.conv4(x)))

            # spatial policy branch
            policy_branch = self.spatial_policy(x)
            spatial_vis = policy_branch
            policy_branch = policy_branch.view(policy_branch.shape[0], -1)
            spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

            # non spatial branch
            non_spatial_represenatation = F.relu(
                self.non_spatial_branch(x.view(-1)))  # flatten the state representation
            value = self.value(non_spatial_represenatation)
            return None, spatial_action_prob, value, F.softmax(spatial_vis[0][0], dim=1)

class FullyConvSelectPolicy(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvSelectPolicy, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.select_unit_policy= nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        policy_branch = self.select_unit_policy(x)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        select_unit_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))
        return select_unit_prob, value, select_task_prob


class CollectAndDestroyBaseline(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(CollectAndDestroyBaseline, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels+1, 49, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(49, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1,1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0]*screen_resolution[1]*32, 256)
        self.value = nn.Linear(256, 1)
        self.task_policy = nn.Linear(256, 2)

    def forward(self, x, action_features, action_type):
        x = torch.cat([x, action_features], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        spatial_policy = self.spatial_policy(x)
        spatial_policy = spatial_policy.view(spatial_policy.shape[0], -1)
        spatial_prob = F.softmax(spatial_policy, dim=1)

        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))
        value = self.value(non_spatial_represenatation)
        task_prob = F.softmax(self.task_policy(non_spatial_represenatation))
        return spatial_prob, value, task_prob


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
        super(CollectAndDestroyGraftingNet, self).__init__()
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
        self.select_task_policy = nn.Linear(256, 2)

    def forward(self, x, action_features, type):
        select_x = F.relu(self.conv_select(x))
        collect_x = F.relu(self.conv_collect(x))
        destroy_x = F.relu(self.conv_destroy(x))

        concat_feature_layers = torch.cat([select_x, collect_x, destroy_x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        if type == 0:
            # compute prob of unit selection
            select_unit_branch = self.select_unit_policy(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(select_unit_branch, dim=1)
        elif type == 1:
            # compute prob of collect minerals
            collect_branch = self.collect_policy(x)
            collect_branch = collect_branch.view(collect_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(collect_branch, dim=1)
        elif type == 2:
            # compute prob of destroy buildings
            destroy_branch = self.destroy_policy(x)
            destroy_branch = destroy_branch.view(destroy_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(destroy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))

        return spatial_prob, value, select_task_prob


class CollectAndDestroyGraftingDropoutNet(nn.Module):
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
        super(CollectAndDestroyGraftingDropoutNet, self).__init__()
        self.conv_select = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_collect = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_destroy = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # shared hidden layers
        self.conv2 = nn.Conv2d(49, 32, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv2_bn = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv3_bn = nn.BatchNorm2d(32)

        # grafting
        self.select_unit_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.collect_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.destroy_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy = nn.Linear(256, 2)
        self.dropout_rate = 0.95
        self.scale = 1.0

    def anneal_scale(self):
        # new_scale = self.scale * 0.999980
        new_scale = self.scale * 0.999880
        if new_scale < 0.05:
            self.scale = 0.05
        else:
            self.scale = new_scale

    def get_dropout_rate(self):
        return self.dropout_rate

    def set_dropout_rate(self, rate):
        self.dropout_rate = rate

    def anneal_dropout_rate(self):
        new_dropout_rate = self.dropout_rate - 0.00001
        # new_dropout_rate = self.dropout_rate - 0.0001
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, type):
        if type == 0:
            select_x = F.relu(self.conv_select(x))
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)

        elif type == 1:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)
        elif type == 2:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))

        concat_feature_layers = torch.cat([select_x, collect_x, destroy_x, action_features], dim=1)
        x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))
        # x = F.relu(self.conv2(concat_feature_layers))
        x = F.relu(self.conv3_bn(self.conv3(x)))

        if type == 0:
            # compute prob of unit selection
            select_unit_branch = self.select_unit_policy(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(select_unit_branch, dim=1)
        elif type == 1:
            # compute prob of collect minerals
            collect_branch = self.collect_policy(x)
            collect_branch = collect_branch.view(collect_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(collect_branch, dim=1)
        elif type == 2:
            # compute prob of destroy buildings
            destroy_branch = self.destroy_policy(x)
            destroy_branch = destroy_branch.view(destroy_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(destroy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))
        return spatial_prob, value, select_task_prob


class CollectAndDestroyGraftingDropoutNetWOBN(nn.Module):
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
        super(CollectAndDestroyGraftingDropoutNetWOBN, self).__init__()
        self.conv_select = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_collect = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_destroy = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # shared hidden layers
        self.conv2 = nn.Conv2d(49, 32, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv2_bn = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv3_bn = nn.BatchNorm2d(32)

        # grafting
        self.select_unit_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.collect_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.destroy_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy = nn.Linear(256, 2)
        self.dropout_rate = 0.95
        self.scale = 1.0

    def anneal_scale(self):
        # new_scale = self.scale * 0.999980
        new_scale = self.scale * 0.999880
        if new_scale < 0.05:
            self.scale = 0.05
        else:
            self.scale = new_scale

    def get_dropout_rate(self):
        return self.dropout_rate

    def set_dropout_rate(self, rate):
        self.dropout_rate = rate

    def anneal_dropout_rate(self):
        new_dropout_rate = self.dropout_rate - 0.00001
        # new_dropout_rate = self.dropout_rate - 0.0001
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, type):
        if type == 0:
            select_x = F.relu(self.conv_select(x))
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)

        elif type == 1:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)
        elif type == 2:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))

        concat_feature_layers = torch.cat([select_x, collect_x, destroy_x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        if type == 0:
            # compute prob of unit selection
            select_unit_branch = self.select_unit_policy(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(select_unit_branch, dim=1)
        elif type == 1:
            # compute prob of collect minerals
            collect_branch = self.collect_policy(x)
            collect_branch = collect_branch.view(collect_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(collect_branch, dim=1)
        elif type == 2:
            # compute prob of destroy buildings
            destroy_branch = self.destroy_policy(x)
            destroy_branch = destroy_branch.view(destroy_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(destroy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))
        return spatial_prob, value, select_task_prob


class CollectAndDestroyGraftingDropoutNetConv4(nn.Module):
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
        super(CollectAndDestroyGraftingDropoutNetConv4, self).__init__()
        self.conv_select = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_collect = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_destroy = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # shared hidden layers
        self.conv2 = nn.Conv2d(49, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(32)

        # grafting
        self.select_unit_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.collect_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.destroy_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy = nn.Linear(256, 2)
        self.dropout_rate = 0.95

    def get_dropout_rate(self):
        return self.dropout_rate

    def set_dropout_rate(self, rate):
        self.dropout_rate = rate

    def anneal_dropout_rate(self):
        new_dropout_rate = self.dropout_rate - 0.0001
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, type):
        if type == 0:
            select_x = F.relu(self.conv_select(x))
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)

        elif type == 1:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)
        elif type == 2:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))

        concat_feature_layers = torch.cat([select_x, collect_x, destroy_x, action_features], dim=1)
        x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))

        # x = F.relu(self.conv3_bn(self.conv3(x)))

        if type == 0:
            # compute prob of unit selection
            select_unit_branch = self.select_unit_policy(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(select_unit_branch, dim=1)
        elif type == 1:
            # compute prob of collect minerals
            collect_branch = self.collect_policy(x)
            collect_branch = collect_branch.view(collect_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(collect_branch, dim=1)
        elif type == 2:
            # compute prob of destroy buildings
            destroy_branch = self.destroy_policy(x)
            destroy_branch = destroy_branch.view(destroy_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(destroy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))
        return spatial_prob, value, select_task_prob


class CollectAndDestroyGraftingDropoutNetConv6(nn.Module):
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
        super(CollectAndDestroyGraftingDropoutNetConv6, self).__init__()
        self.conv_select = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_collect = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_destroy = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # shared hidden layers
        self.conv2 = nn.Conv2d(49, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(32)

        # grafting
        self.select_unit_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.collect_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.destroy_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy = nn.Linear(256, 2)
        self.dropout_rate = 0.95

    def get_dropout_rate(self):
        return self.dropout_rate

    def set_dropout_rate(self, rate):
        self.dropout_rate = rate

    def anneal_dropout_rate(self):
        new_dropout_rate = self.dropout_rate - 0.0001
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, type):
        if type == 0:
            select_x = F.relu(self.conv_select(x))
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)

        elif type == 1:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)
        elif type == 2:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))

        concat_feature_layers = torch.cat([select_x, collect_x, destroy_x, action_features], dim=1)
        x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))

        # x = F.relu(self.conv3_bn(self.conv3(x)))

        if type == 0:
            # compute prob of unit selection
            select_unit_branch = self.select_unit_policy(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(select_unit_branch, dim=1)
        elif type == 1:
            # compute prob of collect minerals
            collect_branch = self.collect_policy(x)
            collect_branch = collect_branch.view(collect_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(collect_branch, dim=1)
        elif type == 2:
            # compute prob of destroy buildings
            destroy_branch = self.destroy_policy(x)
            destroy_branch = destroy_branch.view(destroy_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(destroy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))
        return spatial_prob, value, select_task_prob


class CollectAndDestroyGraftingDropoutNetBN(nn.Module):
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
        super(CollectAndDestroyGraftingDropoutNetBN, self).__init__()
        self.conv_select = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_collect = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_destroy = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        self.cs_bn = nn.BatchNorm2d(16)
        self.cc_bn = nn.BatchNorm2d(16)
        self.cd_bn = nn.BatchNorm2d(16)

        # shared hidden layers
        self.conv2 = nn.Conv2d(49, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)

        # grafting
        self.select_unit_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.collect_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.destroy_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy = nn.Linear(256, 2)
        self.dropout_rate = 0.95

    def get_dropout_rate(self):
        return self.dropout_rate

    def set_dropout_rate(self, rate):
        self.dropout_rate = rate

    def anneal_dropout_rate(self):
        new_dropout_rate = self.dropout_rate - 0.000025
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, type):
        if type == 0:
            select_x = F.relu(self.cs_bn(self.conv_select(x)))
            collect_x = F.relu(self.cc_bn(self.conv_collect(x)))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.cd_bn(self.conv_destroy(x)))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)

        elif type == 1:
            select_x = F.relu(self.cs_bn(self.conv_select(x)))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.cc_bn(self.conv_collect(x)))
            destroy_x = F.relu(self.cd_bn(self.conv_destroy(x)))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)
        elif type == 2:
            select_x = F.relu(self.cs_bn(self.conv_select(x)))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.cc_bn(self.conv_collect(x)))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.cd_bn(self.conv_destroy(x)))

        concat_feature_layers = torch.cat([select_x, collect_x, destroy_x, action_features], dim=1)
        x = F.relu(self.conv2_bn(self.conv2(concat_feature_layers)))

        if type == 0:
            # compute prob of unit selection
            select_unit_branch = self.select_unit_policy(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(select_unit_branch, dim=1)
        elif type == 1:
            # compute prob of collect minerals
            collect_branch = self.collect_policy(x)
            collect_branch = collect_branch.view(collect_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(collect_branch, dim=1)
        elif type == 2:
            # compute prob of destroy buildings
            destroy_branch = self.destroy_policy(x)
            destroy_branch = destroy_branch.view(destroy_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(destroy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))
        return spatial_prob, value, select_task_prob


class MultiFeaturesGrafting(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(MultiFeaturesGrafting, self).__init__()
        self.select_conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.select_conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.collect_conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.collect_conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.select_unit = nn.Conv2d(65, 1, kernel_size=(1, 1))
        self.spatial_policy = nn.Conv2d(65, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 65, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        select_x = F.relu(self.select_conv1(x))
        select_x = F.relu(self.select_conv2(select_x))

        collect_x = F.relu(self.collect_conv1(x))
        collect_x = F.relu(self.collect_conv2(collect_x))

        concat_feature_layers = torch.cat([select_x, collect_x, action_features], dim=1)

        # select
        select_unit_branch = self.select_unit(concat_feature_layers)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

        # collect
        policy_branch = self.spatial_policy(concat_feature_layers)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(concat_feature_layers.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, None


class CollectAndDestroyGraftingDropoutNetNoBN(nn.Module):
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
        super(CollectAndDestroyGraftingDropoutNetNoBN, self).__init__()
        self.conv_select = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_collect = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_destroy = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # shared hidden layers
        self.conv2 = nn.Conv2d(49, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.select_unit_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.collect_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.destroy_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)
        self.select_task_policy = nn.Linear(256, 2)
        self.dropout_rate = 0.95

    def get_dropout_rate(self):
        return self.dropout_rate

    def set_dropout_rate(self, rate):
        self.dropout_rate = rate

    def anneal_dropout_rate(self):
        new_dropout_rate = self.dropout_rate - 0.0001
        if new_dropout_rate > 0:
            self.dropout_rate = new_dropout_rate
        else:
            self.dropout_rate = 0

    def forward(self, x, action_features, type):
        if type == 0:
            select_x = F.relu(self.conv_select(x))
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)

        elif type == 1:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            destroy_x = F.relu(self.conv_destroy(x))
            destroy_x = F.dropout(destroy_x, p=self.dropout_rate, training=self.training)
        elif type == 2:
            select_x = F.relu(self.conv_select(x))
            select_x = F.dropout(select_x, p=self.dropout_rate, training=self.training)
            collect_x = F.relu(self.conv_collect(x))
            collect_x = F.dropout(collect_x, p=self.dropout_rate, training=self.training)
            destroy_x = F.relu(self.conv_destroy(x))

        concat_feature_layers = torch.cat([select_x, collect_x, destroy_x, action_features], dim=1)
        x = F.relu((self.conv2(concat_feature_layers)))
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))

        if type == 0:
            # compute prob of unit selection
            select_unit_branch = self.select_unit_policy(x)
            select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(select_unit_branch, dim=1)
        elif type == 1:
            # compute prob of collect minerals
            collect_branch = self.collect_policy(x)
            collect_branch = collect_branch.view(collect_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(collect_branch, dim=1)
        elif type == 2:
            # compute prob of destroy buildings
            destroy_branch = self.destroy_policy(x)
            destroy_branch = destroy_branch.view(destroy_branch.shape[0], -1)
            spatial_prob = nn.functional.softmax(destroy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))
        value = self.value(non_spatial_represenatation)
        select_task_prob = F.softmax(self.select_task_policy(non_spatial_represenatation))
        return spatial_prob, value, select_task_prob