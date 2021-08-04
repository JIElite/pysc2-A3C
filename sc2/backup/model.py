import torch
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
        spatial_policy = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)
        return action_prob, value, F.softmax(spatial_policy[0][0], dim=1)


class FullyConvBN(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvBN, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        spatial_policy = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)
        return action_prob, value, F.softmax(spatial_policy[0][0], dim=1)


class FullyConvExtended(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvExtended, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
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
        return action_prob, value, None


class FullyConvWithActionIndicator(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvWithActionIndicator, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels+1, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

        # init weight
        # nn.init.xavier_uniform(self.conv1.weight.data)
        # nn.init.xavier_uniform(self.conv2.weight.data)
        # nn.init.xavier_uniform(self.spatial_policy.weight.data)
        # nn.utils.weight_norm(self.non_spatial_branch)
        # nn.utils.weight_norm(self.value)
        # self.non_spatial_branch.bias.data.fill_(0)
        # self.value.bias.data.fill_(0)

    def forward(self, x, action_indicator):
        x = torch.cat([x, action_indicator], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        spatial_policy = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)
        return action_prob, value, F.softmax(spatial_policy[0][0], dim=1)


class FullyConvExtendConv3WithActionIndicator(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConvExtendConv3WithActionIndicator, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels+1, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
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

    def forward(self, x, action_indicator):
        x = torch.cat([x, action_indicator], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        spatial_policy = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)
        return action_prob, value, F.softmax(spatial_policy[0][0], dim=1)


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
        spatial_vis = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1))) # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, F.softmax(spatial_vis[0][0])


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

        return select_unit_prob, spatial_action_prob, value, None


class Grafting_MultiunitCollect(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(Grafting_MultiunitCollect, self).__init__()
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        master_x = F.relu(self.conv_master(x))
        sub_x = F.relu(self.conv_sub(x))

        concat_feature_layers = torch.cat([master_x, sub_x], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, None


class ExtendConv3Grafting_MultiunitCollect(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        # grafting
        super(ExtendConv3Grafting_MultiunitCollect, self).__init__()
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        master_x = F.relu(self.conv_master(x))
        sub_x = F.relu(self.conv_sub(x))

        concat_feature_layers = torch.cat([master_x, sub_x], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))
        x = F.relu(self.conv3(x))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob =  nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        temp_policy_branch = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, None


class Grafting_MultiunitCollect_WithActionFeatures(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(Grafting_MultiunitCollect_WithActionFeatures, self).__init__()
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        master_x = F.relu(self.conv_master(x))

        sub_x = F.relu(self.conv_sub(x))

        concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        spatial_vis = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, F.softmax(spatial_vis[0][0], dim=1)
        # return select_unit_prob, spatial_action_prob, value, spatial_vis[0][0]




class ExtendConv3Grafting_MultiunitCollect_WithActionFeatures(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        # grafting
        super(ExtendConv3Grafting_MultiunitCollect_WithActionFeatures, self).__init__()
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        master_x = F.relu(self.conv_master(x))
        sub_x = F.relu(self.conv_sub(x))

        concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))
        x = F.relu(self.conv3(x))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob =  nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        spatial_vis = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, F.softmax(spatial_vis[0][0])


class MultiInputSinglePolicyNet(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(MultiInputSinglePolicyNet, self).__init__()
        # grafting
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        master_x = F.relu(self.conv_master(x))
        sub_x = F.relu(self.conv_sub(x))

        concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        # spatial policy branch
        policy_branch = self.policy(x)
        spatial_vis = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)
        return spatial_action_prob, value, F.softmax(spatial_vis[0][0], dim=1)


class MultiInputSinglePolicyNetExtendConv3(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(MultiInputSinglePolicyNetExtendConv3, self).__init__()
        # grafting
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        master_x = F.relu(self.conv_master(x))
        sub_x = F.relu(self.conv_sub(x))

        concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))
        x = F.relu(self.conv3(x))

        # spatial policy branch
        policy_branch = self.policy(x)
        spatial_vis = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)
        return spatial_action_prob, value, F.softmax(spatial_vis[0][0], dim=1)


class CollectFiveBaselineWithActionFeatures(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(CollectFiveBaselineWithActionFeatures, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(17, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        x = F.relu(self.conv1(x))
        concat_feature_layers = torch.cat([x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob =  nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        spatial_vis = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, F.softmax(spatial_vis[0][0])


class CollectFiveBaselineWithActionFeaturesV2(nn.Module):
    """
    The difference between v1 and v2 is the # of channels in the first CNN.
    """
    def __init__(self, screen_channels, screen_resolution):
        super(CollectFiveBaselineWithActionFeaturesV2, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 32, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        x = F.relu(self.conv1(x))
        concat_feature_layers = torch.cat([x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob =  nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        temp_policy_branch = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, None


class CollectFiveBaselineWithActionFeaturesExtendConv3(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(CollectFiveBaselineWithActionFeaturesExtendConv3, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(17, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        x = F.relu(self.conv1(x))
        concat_feature_layers = torch.cat([x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))
        x = F.relu(self.conv3(x))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob =  nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        temp_policy_branch = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, None


class CollectFiveBaselineWithActionFeaturesExtendConv3V2(nn.Module):
    """
    the difference between original and v2 is self.conv1
    origin's input channel is 8, output is 16
    v2                        8            32
    """
    def __init__(self, screen_channels, screen_resolution):
        super(CollectFiveBaselineWithActionFeaturesExtendConv3V2, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 32, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x, action_features):
        x = F.relu(self.conv1(x))
        concat_feature_layers = torch.cat([x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))
        x = F.relu(self.conv3(x))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob =  nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        temp_policy_branch = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, None


class NewGraftingModel(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(NewGraftingModel, self).__init__()
        self.conv_master = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv_sub = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)

        # train from scratch
        self.conv2 = nn.Conv2d(33, 32, kernel_size=(3, 3), stride=1, padding=1)

        # grafting
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.select_unit = nn.Conv2d(32, 1, kernel_size=(1, 1))

        # grafting
        self.non_spatial_branch = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x1, x2, action_features):
        master_x = F.relu(self.conv_master(x1))
        sub_x = F.relu(self.conv_sub(x2))

        concat_feature_layers = torch.cat([master_x, sub_x, action_features], dim=1)
        x = F.relu(self.conv2(concat_feature_layers))

        select_unit_branch = self.select_unit(x)
        select_unit_branch = select_unit_branch.view(select_unit_branch.shape[0], -1)
        select_unit_prob = nn.functional.softmax(select_unit_branch, dim=1)

        # spatial policy branch
        policy_branch = self.spatial_policy(x)
        spatial_vis = policy_branch
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        spatial_action_prob = nn.functional.softmax(policy_branch, dim=1)

        # non spatial branch
        non_spatial_represenatation = F.relu(self.non_spatial_branch(x.view(-1)))  # flatten the state representation
        value = self.value(non_spatial_represenatation)

        return select_unit_prob, spatial_action_prob, value, F.softmax(spatial_vis[0][0], dim=1)