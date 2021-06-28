from __future__ import absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from region_proposal_network import RegionProposalNetwork
from 