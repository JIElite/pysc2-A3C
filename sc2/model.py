import torch.nn as nn
import torch.nn.functional as F



class FullyConv(nn.Module):
    def __init__(self, screen_channels, screen_resolution):
        super(FullyConv, self).__init__()
        self.conv1 = nn.Conv2d(screen_channels, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.spatial_policy = nn.Conv2d(32, 1, kernel_size=(1, 1))

        self.v1 = nn.Linear(screen_resolution[0] * screen_resolution[1] * 32, 256)
        self.non_spatial_policy = nn.Linear(256, 2)
        self.value = nn.Linear(256, 1)

        # init weight
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.spatial_policy.weight.data)
        nn.utils.weight_norm(self.v1)
        nn.utils.weight_norm(self.value)
        self.v1.bias.data.fill_(0)
        self.value.bias.data.fill_(0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # flatten conv policy to (1, policy_branch.shape[0]) -> (1, 64*64)
        policy_branch = self.spatial_policy(x)
        policy_branch = policy_branch.view(policy_branch.shape[0], -1)
        action_prob = nn.functional.softmax(policy_branch, dim=1)
        log_action_prob = nn.functional.log_softmax(policy_branch, dim=1)

        v1 = F.relu(self.v1(x.view(-1)))
        non_spatial_policy = self.non_spatial_policy(v1).unsqueeze(0)
        non_spatial_policy_prob = F.softmax(non_spatial_policy, dim=1)
        non_spatial_policy_log_prob = F.log_softmax(non_spatial_policy, dim=1)
        value = self.value(v1)

        # return non_spatial_policy_prob,  non_spatial_policy_log_prob,  action_prob, log_action_prob, value
        return action_prob, log_action_prob, value
